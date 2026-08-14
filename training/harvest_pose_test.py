#!/usr/bin/env python3
"""harvest_pose_test.py -- cosecha con la nueva estrategia (pose_pipeline +
labeling_strategy).

Mantiene la ESTRATEGIA TEMPORAL de harvest_v14.py sin cambios: centro
temporal del clip, ventana de +-2s (los "4 segundos" con los que trabaja
el modelo, WINDOW_S en config.py) -- ahora con el FPS REAL de la carpeta
(labeling_strategy.FPS_POR_CARPETA), no un valor fijo: la metadata del
contenedor no es confiable (confirmado para control_negativo/
doblar_sobreponer: dicen 11fps, son 8fps de verdad).

Nota: labeling_strategy.aplica_reglas_etiquetado() calcula tambien
`frame_inds` para algunas carpetas (control_negativo, doblar_sobreponer) --
una ventana de 32 frames consecutivos en una posicion aleatoria del clip,
del pipeline PoseC3D de referencia. Este harvester NO la usa para elegir
frames (se sigue usando la ventana +-2s centrada de harvest_v14.py, tal
como se pidio) -- frame_inds solo se calcula porque viene con las reglas de
etiquetado, no se descarta en silencio, pero tampoco se aplica aca.

Lo que cambia respecto a harvest_v14.py:
  - Deteccion de personas: en vez de YOLO deteccion+ByteTrack (bytetrack
    puede perder/cambiar de ID cuando la deteccion falla un frame), se usa
    YOLO-POSE en modo `predict` frame a frame, SIN tracking, y por cada
    frame se toma la deteccion de MAYOR CONFIANZA. Es exactamente el
    criterio de `2_pose_pkl_regenation.py::no_coords_flag=True` (su rama
    "no hace falta tracking porque solo hay una persona") -- aplica
    directo a estos videos, que tienen una unica persona.
  - Crop + resize: pose_pipeline.bbox_desde_keypoints (bbox de TODOS los
    keypoints validos de la ventana, marco proporcional) +
    recorta_y_redimensiona_gpu (UN solo resize, torch, nearest) en vez de
    make_tube (crop por deteccion + cv2.resize INTER_AREA).
  - Labels: YA NO SE RESUELVEN ACA. anotaciones.csv guarda dataset_tipo +
    clip_origen (crudos) pero NINGUNA columna "label" -- no sabemos todavia,
    al cosechar, si el entrenamiento final va a ser binario o multiclase
    (labeling_strategy_binaria.py vs labeling_strategy.py resuelven
    distinto, y algunas carpetas -- reingesta_*, manipulaciones_tiendas --
    solo son resolubles para binario, no para multiclase). Resolver el
    label a posteriori, a partir de dataset_tipo+clip_origen (alcanza con
    esos dos campos, ya estan en la tabla), es
    training/genera_anotaciones_por_cabeza.py -- corre sobre el .npy YA
    cosechado, sin volver a tocar video. fps_de_carpeta() SI se sigue
    llamando aca (una vez, al arrancar): si dataset_tipo no tiene FPS
    conocido, no hay forma de cosechar nada, eso no depende de la cabeza.

Uso:
  python3 training/harvest_pose_test.py <carpeta_videos> <out_dir> [--pose-model PATH]

<carpeta_videos> debe ser una carpeta cuyo nombre sea un dataset_tipo
conocido (labeling_strategy.FPS_POR_CARPETA) -- ej. ".../videos_for_train/control_negativo".
"""
import csv
import errno
import hashlib
import os
import sys
import glob
import random
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_pipeline as pp
import labeling_strategy as ls
from ultralytics import YOLO

DEVICE = "cuda"
PROB_VIDEO_QA = 0.01   # ~1% de los tubos, ademas de --guardar-videos (que fuerza el 100%)
MAX_LARGO_NOMBRE = 150   # algunos clips fuente tienen nombres absurdamente largos --
                         # visto en vivo: uno rompio el limite de 255 bytes del filesystem
                         # al construir "L<label>__<carpeta>__<nombre>.npy.tmp"


def nombre_seguro(base, max_len=MAX_LARGO_NOMBRE):
    """Trunca `base` si excede max_len caracteres, agregando un hash corto
    del nombre COMPLETO original para no perder unicidad entre dos nombres
    que truncados quedarian identicos."""
    if len(base) <= max_len:
        return base
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return base[:max_len] + "__" + h


def guarda_npy_atomico(path, arr):
    """np.save directo puede dejar un .npy TRUNCADO/corrupto si el disco se
    llena a mitad de la escritura (justo el escenario que puede pasar en una
    cosecha larga). Se escribe a un .tmp en el mismo directorio y se hace
    rename -- en Linux el rename es atomico, asi que nunca queda un .npy a
    medio escribir con el nombre final: o esta completo, o no existe."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:   # file handle, no un path -- si no, np.save AGREGA ".npy" solo
        np.save(f, arr)
    os.replace(tmp, path)


def clips_ya_procesados(anotaciones_path):
    """Lee anotaciones.csv si ya existe (de una corrida anterior interrumpida)
    y devuelve el set de clip_origen ya escritos -- para poder retomar sin
    reprocesar. Si el archivo no existe todavia, set vacio."""
    if not os.path.exists(anotaciones_path):
        return set()
    with open(anotaciones_path) as f:
        return {r["clip_origen"] for r in csv.DictReader(f)}


def frame_a_gpu(frame_bgr):
    t = torch.from_numpy(frame_bgr).to(DEVICE, non_blocking=True)
    return t.flip(-1).permute(2, 0, 1).contiguous()   # BGR->RGB, HWC->CHW


def keypoints_por_frame_max_confianza(modelo_pose, frame_bgr):
    """Un frame -> (17,2) en pixeles nativos. Sin tracking: si hay mas de una
    deteccion, se toma la de mayor confianza (criterio de
    2_pose_pkl_regenation.py para videos de una sola persona). Si no hay
    ninguna deteccion, (0,0) en los 17 keypoints -- mismo marcador de
    "invalido" que ya espera pose_pipeline.bbox_desde_keypoints."""
    res = modelo_pose.predict(frame_bgr, conf=0.2, verbose=False)[0]
    if res.boxes is None or res.boxes.conf.shape[0] == 0 or res.keypoints is None:
        return np.zeros((pp.NUM_KEYPOINTS, 2), dtype=np.float32)
    idx_mejor = int(res.boxes.conf.cpu().numpy().argmax())
    return res.keypoints.xy.cpu().numpy()[idx_mejor]   # (17,2)


def keypoints_por_ventana_batch(modelo_pose, frames_bgr):
    """Igual que aplicar keypoints_por_frame_max_confianza a cada frame de
    `frames_bgr` (lista de ~32 frames), pero en UNA sola llamada a
    modelo_pose.predict() con la lista completa -- ultralytics batchea
    internamente. Medido en vivo: llamar predict() por frame (32 llamadas
    Python+CUDA-launch separadas por clip) es el cuello de botella real de
    la cosecha, no el forward del modelo en si -- batchear da varias veces
    mas throughput sin cambiar el criterio de seleccion por frame (maxima
    confianza), que se sigue aplicando exactamente igual, solo que sobre el
    Results de cada imagen del batch en vez de una sola."""
    resultados = modelo_pose.predict(frames_bgr, conf=0.2, verbose=False)
    salida = np.zeros((len(frames_bgr), pp.NUM_KEYPOINTS, 2), dtype=np.float32)
    for i, res in enumerate(resultados):
        if res.boxes is None or res.boxes.conf.shape[0] == 0 or res.keypoints is None:
            continue
        idx_mejor = int(res.boxes.conf.cpu().numpy().argmax())
        salida[i] = res.keypoints.xy.cpu().numpy()[idx_mejor]
    return salida


FPS_OBJETIVO = 25 / 3   # ~8.33 -- "acercar todo a 8fps" pedido explicitamente


def extrae_tubo(modelo_pose, clip, marco, fps, n_frames=pp.N_FRAMES):
    cap = cv2.VideoCapture(clip)
    w, h = int(cap.get(3)), int(cap.get(4))
    frames_todos = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames_todos.append(fr)
    cap.release()
    if not frames_todos:
        return None, None, None

    # Acercar todo a ~8fps: las carpetas a 25fps corren pose sobre ~3x mas
    # frames por ventana que las de 8fps para la MISMA duracion (+-2s) --
    # 3x mas lento sin ganar nada (la ventana temporal sigue siendo ~4s de
    # video real). Subsamplear a 1 de cada `stride` frames antes de armar la
    # ventana homogeniza la densidad temporal entre carpetas Y achica el
    # costo de pose proporcionalmente. Con fps=25 -> stride=3 (~8.33fps,
    # exactamente lo pedido); con fps=8/9 -> stride=1 (no cambia nada).
    stride = max(1, round(fps / FPS_OBJETIVO))
    frames = frames_todos[::stride]
    fps_efectivo = fps / stride

    n = len(frames)
    center = n // 2
    radio = fps_efectivo * 2   # +-2s = ~4s, el WINDOW_S del modelo -- sobre el fps YA acercado a 8, no el original
    idx_ventana = [i for i in range(n) if abs(i - center) <= radio]
    if len(idx_ventana) < 4:
        idx_ventana = list(range(n))   # mismo fallback que harvest_v14.py: si la ventana quedo muy chica, usar todo el clip

    kp_clip = keypoints_por_ventana_batch(modelo_pose, [frames[i] for i in idx_ventana])
    ventana = pp.bbox_desde_keypoints(kp_clip, w, h, marco=marco)
    if ventana is None:
        return None, None, None

    frames_gpu = [frame_a_gpu(frames[i]) for i in idx_ventana]
    tubo_store = pp.recorta_y_redimensiona_gpu(frames_gpu, ventana, out_size=pp.STORE_SIZE, n_frames=n_frames)
    return tubo_store, ventana, len(frames_todos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta_videos")
    ap.add_argument("out_dir")
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--guardar-videos", action="store_true",
                     help="fuerza guardar un .mp4 por tubo (100%%), ademas del .npy -- por "
                          f"defecto solo se guarda para una muestra aleatoria del "
                          f"{PROB_VIDEO_QA*100:.0f}%% (inspeccion visual del recorte)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                     help="fraccion de clips que se van a split=val (aleatorio, ver aviso abajo)")
    ap.add_argument("--n-frames", type=int, default=32,
                     help="frames por tubo (default 32, antes 16 -- pedido explicito para poder "
                          "usar los checkpoints VideoMAE a 32 frames/mayor resolucion temporal, "
                          "ver docstring de pose_pipeline.N_FRAMES). training/train_binario.py "
                          "detecta solo cuantos frames tiene cada .npy y resamplea si hace falta, "
                          "asi que mezclar tubos de 16 y 32 frames en el mismo dataset no rompe "
                          "nada, pero conviene ser consistente por corrida de cosecha.")
    args = ap.parse_args()

    dataset_tipo = os.path.basename(os.path.normpath(args.carpeta_videos))
    fps = ls.fps_de_carpeta(dataset_tipo)   # se detiene aca mismo si dataset_tipo es desconocido -- falla rapido, antes de procesar nada
    print(f"dataset_tipo={dataset_tipo!r}  fps_real={fps}")

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    clips = sorted(glob.glob(os.path.join(args.carpeta_videos, "*.mp4")))
    print(f"{len(clips)} clips en {args.carpeta_videos}")

    # anotaciones.csv CRUDO: npy,split,marco,x0,y0,x1,y1,fps,clip_origen --
    # SIN "label" (ver docstring del modulo: se resuelve a posteriori, con
    # training/genera_anotaciones_por_cabeza.py, porque dataset_tipo ya esta
    # en la tabla -- ver harvest_muestra.py, que agrega esa columna -- y
    # clip_origen alcanza para recalcular cualquier label sin volver a tocar
    # video). "split" sigue siendo un sorteo simple (ALEATORIO) -- para el
    # dataset real conviene separar por clip/dia/camara de origen, no al
    # azar por tubo, para no filtrar el mismo evento entre train y val.
    print("AVISO: split en anotaciones.csv es un sorteo -- todavia no se separa por dia/camara de origen")

    anot_path = os.path.join(args.out_dir, "anotaciones.csv")
    ya_procesados = clips_ya_procesados(anot_path)   # RETOMAR: de una corrida anterior interrumpida (ej. disco lleno)
    if ya_procesados:
        print(f"RETOMANDO: {len(ya_procesados)} clips ya estan en {anot_path}, se saltan")
    anot = open(anot_path, "a" if ya_procesados or os.path.exists(anot_path) else "w", newline="")
    # csv.writer, no f-strings: "ventana" son 4 numeros con comas -- escrito a
    # mano (sin comillas) esas comas se confunden con separadores de columna
    # y corren todo lo que viene despues (fps, clip_origen). Se parte ademas
    # en 4 columnas (x0,y0,x1,y1) en vez de una tupla-string.
    escritor = csv.writer(anot)
    if not ya_procesados and anot.tell() == 0:
        escritor.writerow(["npy", "split", "marco", "x0", "y0", "x1", "y1", "fps", "clip_origen"])
        anot.flush()

    n_ok = n_skip = n_saltados_ya = 0
    for clip in clips:
        nombre_base_clip = os.path.basename(clip)
        if nombre_base_clip in ya_procesados:
            n_saltados_ya += 1
            continue
        nombre_clip = os.path.splitext(nombre_base_clip)[0]
        marco = round(random.uniform(pp.MARCO_MIN_TRAIN, pp.MARCO_MAX_TRAIN), 3)

        try:
            tubo, ventana, total_frames = extrae_tubo(modelo_pose, clip, marco, fps, n_frames=args.n_frames)
        except Exception as e:
            print("ERR", clip, e)
            tubo = None
        if tubo is None:
            n_skip += 1
            print(f"  [skip] {os.path.basename(clip)} -- sin keypoints validos")
            continue

        split = "val" if random.random() < args.val_frac else "train"   # ALEATORIO, ver aviso arriba
        base = nombre_seguro(f"{dataset_tipo}__{nombre_clip}")
        arr = tubo.permute(0, 2, 3, 1).cpu().numpy()   # T,C,H,W -> T,H,W,C, mismo formato que el dataset viejo

        try:
            guarda_npy_atomico(os.path.join(args.out_dir, base + ".npy"), arr)

            if args.guardar_videos or random.random() < PROB_VIDEO_QA:
                # video de la zona de recorte, SOLO para inspeccion visual --
                # no es parte de los datos de entrenamiento en si. Mismo
                # criterio (4fps) que guarda_alerta en el resto del repo.
                video_path = os.path.join(args.out_dir, base + ".mp4")
                h, w = arr.shape[1], arr.shape[2]
                vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
                for fr in arr:
                    vw.write(fr[:, :, ::-1])   # RGB -> BGR para escribir
                vw.release()
                print(f"         (QA) video guardado: {base}.mp4")
        except OSError as e:
            if e.errno == errno.ENOSPC:
                anot.close()
                print(f"\nDISCO LLENO al escribir {base}.npy -- {n_ok} tubos ya quedaron guardados "
                      f"y confirmados en {anot_path} (ninguno queda a medio escribir, el .npy se "
                      f"escribe atomico). Libera espacio y corre EXACTAMENTE el mismo comando de "
                      f"nuevo -- retoma solo, salta los {n_ok} que ya estan en anotaciones.csv.")
                sys.exit(1)
            raise

        # flush + fsync: la fila queda en disco YA, no solo en el buffer de
        # Python/OS -- si el proceso muere (disco lleno, kill -9, corte de luz)
        # en el clip siguiente, esta fila no se pierde y el retomado no la repite.
        x0, y0, x1, y1 = ventana
        escritor.writerow([f"{base}.npy", split, marco, x0, y0, x1, y1, fps, nombre_base_clip])
        anot.flush()
        os.fsync(anot.fileno())
        n_ok += 1
        print(f"  [ok]   {nombre_base_clip} -> {base}.npy  split={split}  marco={marco}  ventana={ventana}  tubo={arr.shape}")

    anot.close()
    print(f"\n{n_ok} tubos generados, {n_skip} saltados, {n_saltados_ya} ya estaban de una corrida "
          f"anterior -> {args.out_dir}/anotaciones.csv")


if __name__ == "__main__":
    main()

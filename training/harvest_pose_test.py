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
  - Labels: numerico multi-clase (labeling_strategy.py), no hurto/normal --
    ver el docstring de ese modulo. Si una carpeta no tiene reglas de
    etiquetado, o si el label queda sin resolver, el script SE DETIENE
    (no se atrapa esa excepcion) -- es la validacion que se pidio.

Uso:
  python3 training/harvest_pose_test.py <carpeta_videos> <out_dir> [--pose-model PATH]

<carpeta_videos> debe ser una carpeta cuyo nombre sea un dataset_tipo
conocido (labeling_strategy.FPS_POR_CARPETA) -- ej. ".../videos_for_train/control_negativo".
"""
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


def extrae_tubo(modelo_pose, clip, marco, fps):
    cap = cv2.VideoCapture(clip)
    w, h = int(cap.get(3)), int(cap.get(4))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None, None, None

    n = len(frames)
    center = n // 2
    radio = fps * 2   # +-2s = ~4s, el WINDOW_S del modelo -- fps REAL de la carpeta, no fijo
    idx_ventana = [i for i in range(n) if abs(i - center) <= radio]
    if len(idx_ventana) < 4:
        idx_ventana = list(range(n))   # mismo fallback que harvest_v14.py: si la ventana quedo muy chica, usar todo el clip

    kp_clip = np.stack([keypoints_por_frame_max_confianza(modelo_pose, frames[i]) for i in idx_ventana])
    ventana = pp.bbox_desde_keypoints(kp_clip, w, h, marco=marco)
    if ventana is None:
        return None, None, None

    frames_gpu = [frame_a_gpu(frames[i]) for i in idx_ventana]
    tubo_store = pp.recorta_y_redimensiona_gpu(frames_gpu, ventana, out_size=pp.STORE_SIZE)
    return tubo_store, ventana, n


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
    args = ap.parse_args()

    dataset_tipo = os.path.basename(os.path.normpath(args.carpeta_videos))
    fps = ls.fps_de_carpeta(dataset_tipo)   # se detiene aca mismo si dataset_tipo es desconocido -- falla rapido, antes de procesar nada
    print(f"dataset_tipo={dataset_tipo!r}  fps_real={fps}")

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    clips = sorted(glob.glob(os.path.join(args.carpeta_videos, "*.mp4")))
    print(f"{len(clips)} clips en {args.carpeta_videos}")

    # anotaciones.csv: UNA sola tabla (npy,label,split,marco,ventana,fps,clip_origen).
    # "label" es el codigo numerico multi-clase de labeling_strategy.py (NO
    # hurto/normal -- ver el docstring de ese modulo). "split" sigue siendo
    # un sorteo simple (ALEATORIO) -- para el dataset real conviene separar
    # por clip/dia/camara de origen, no al azar por tubo, para no filtrar el
    # mismo evento entre train y val.
    print("AVISO: split en anotaciones.csv es un sorteo -- todavia no se separa por dia/camara de origen")
    anot = open(os.path.join(args.out_dir, "anotaciones.csv"), "w")
    anot.write("npy,label,split,marco,ventana,fps,clip_origen\n")

    n_ok = n_skip = 0
    for clip in clips:
        nombre_clip = os.path.splitext(os.path.basename(clip))[0]
        marco = round(random.uniform(pp.MARCO_MIN_TRAIN, pp.MARCO_MAX_TRAIN), 3)

        try:
            tubo, ventana, total_frames = extrae_tubo(modelo_pose, clip, marco, fps)
        except (ls.DatasetTipoDesconocido, ls.LabelNoResuelto):
            raise   # ADVERTENCIA + PAUSA real: no se atrapa, se detiene todo el script
        except Exception as e:
            print("ERR", clip, e)
            tubo = None
        if tubo is None:
            n_skip += 1
            print(f"  [skip] {os.path.basename(clip)} -- sin keypoints validos")
            continue

        results = {
            "frame_dir": os.path.basename(clip),
            "dataset_tipo": dataset_tipo,
            "label": ls.extraer_label_inicial(os.path.basename(clip)),
            "total_frames": total_frames,
        }
        ls.aplica_reglas_etiquetado(results)   # in-place; puede raise DatasetTipoDesconocido/LabelNoResuelto (sin atrapar, ver arriba)
        label = results["label"]

        split = "val" if random.random() < args.val_frac else "train"   # ALEATORIO, ver aviso arriba
        base = f"L{label:02d}__{nombre_clip}"
        arr = tubo.permute(0, 2, 3, 1).cpu().numpy()   # T,C,H,W -> T,H,W,C, mismo formato que el dataset viejo
        np.save(os.path.join(args.out_dir, base + ".npy"), arr)

        if args.guardar_videos or random.random() < PROB_VIDEO_QA:
            # video de la zona de recorte, SOLO para inspeccion visual -- no
            # es parte de los datos de entrenamiento en si. Mismo criterio
            # (4fps) que guarda_alerta en el resto del repo.
            video_path = os.path.join(args.out_dir, base + ".mp4")
            h, w = arr.shape[1], arr.shape[2]
            vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
            for fr in arr:
                vw.write(fr[:, :, ::-1])   # RGB -> BGR para escribir
            vw.release()
            print(f"         (QA) video guardado: {base}.mp4")

        anot.write(f"{base}.npy,{label},{split},{marco},{ventana},{fps},{os.path.basename(clip)}\n")
        n_ok += 1
        print(f"  [ok]   {os.path.basename(clip)} -> {base}.npy  label={label}  split={split}  marco={marco}  ventana={ventana}  tubo={arr.shape}")

    anot.close()
    print(f"\n{n_ok} tubos generados, {n_skip} saltados -> {args.out_dir}/anotaciones.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""harvest_pose_streaming.py -- cosecha video->tubo en streaming, mientras los
.mp4 todavia estan LLEGANDO (ej. via scp desde otra maquina, en paralelo).

Reutiliza toda la logica de extraccion de harvest_pose_test.py (mismo
pose_pipeline + labeling_strategy, mismo criterio de ventana/crop/resize) --
lo unico nuevo aca es COMO SE ELIGEN LOS CLIPS A PROCESAR:

  - Recorre TODAS las subcarpetas de <videos_root> que sean un dataset_tipo
    conocido (labeling_strategy.FPS_POR_CARPETA), no una sola carpeta fija.
  - Antes de tocar un .mp4, chequea que su tamaño en disco este ESTABLE
    (mismo st_size visto en dos PASADAS de escaneo separadas por al menos
    --estable-seg -- ver RastreadorEstabilidad, SIN time.sleep() bloqueante
    por archivo, eso fue un error de la primera version que tapaba casi todo
    el throughput real) -- si todavia esta subiendo/escribiendose, se salta y
    se reintenta en la proxima pasada. Sin esto, leer un .mp4 a medio subir
    con cv2 puede dar un video corto/corrupto.
  - Si en una pasada completa no aparece NINGUN clip nuevo y estable, espera
    --poll-seg y vuelve a escanear (los nuevos archivos van apareciendo via
    upload externo) -- hasta --idle-max-seg de inactividad total, momento en
    el que asume que ya no van a llegar mas videos y corta.
  - Se detiene apenas se alcanza --target-tubos (default 67000) -- no hace
    falta que sea exacto, un poco mas esta bien (se corta despues de terminar
    el clip que lo cruza, no a mitad de uno).

anotaciones.csv combina TODAS las carpetas en un solo out_dir -- clip_origen
se guarda como "dataset_tipo/nombre_archivo.mp4" (no solo el nombre) porque
dos carpetas distintas podrian tener un clip con el mismo nombre.

Uso:
  python3 training/harvest_pose_streaming.py <videos_root> <out_dir> \
      [--target-tubos 67000] [--pose-model yolo11m-pose.pt] [--n-frames 32] \
      [--estable-seg 3] [--poll-seg 5] [--idle-max-seg 300]

<videos_root> ej. ".../videos_for_train" -- sus subcarpetas directas deben
llamarse como un dataset_tipo conocido (ver labeling_strategy.FPS_POR_CARPETA).
Subcarpetas con nombre desconocido se listan y se saltan (no se aborta todo
el barrido por una carpeta rara).
"""
import csv
import os
import sys
import glob
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_pipeline as pp
import labeling_strategy as ls
from ultralytics import YOLO

from harvest_pose_test import (
    nombre_seguro, guarda_npy_atomico, extrae_tubo, MAX_LARGO_NOMBRE, PROB_VIDEO_QA,
)

import cv2


def clips_ya_procesados(anotaciones_path):
    """clip_origen aca es 'dataset_tipo/archivo.mp4' (ver docstring del
    modulo) -- distinto del harvest_pose_test.py original (que guarda solo
    el nombre de archivo, valido porque el corre sobre UNA sola carpeta)."""
    if not os.path.exists(anotaciones_path):
        return set()
    with open(anotaciones_path) as f:
        return {r["clip_origen"] for r in csv.DictReader(f)}


def tubos_ya_generados(anotaciones_path):
    if not os.path.exists(anotaciones_path):
        return 0
    with open(anotaciones_path) as f:
        return sum(1 for _ in csv.DictReader(f))


class RastreadorEstabilidad:
    """Detecta ".mp4 ya termino de subir/escribirse" SIN bloquear con
    time.sleep() por archivo (eso fue un error de la primera version --
    con miles de archivos, dormir 3s por cada uno revisado tapaba
    practicamente todo el throughput real de decode+pose, sin importar
    cuanto se optimizara el resto).

    En vez de eso: se guarda (tamaño, timestamp) de la ULTIMA vez que se vio
    cada archivo. Cada pasada de escaneo separa naturalmente esas lecturas
    por el tiempo que toma procesar el resto de los clips de la pasada
    anterior (tipicamente varios segundos) -- si el tamaño no cambio Y ya
    paso al menos `espera_seg` desde que se vio por primera vez con ese
    tamaño, se considera estable. Un archivo nuevo (nunca visto) NUNCA se
    procesa en la misma pasada en que aparece -- como minimo espera a la
    pasada siguiente, que ya es tiempo mas que suficiente en la practica."""

    def __init__(self, espera_seg):
        self.espera_seg = espera_seg
        self._visto = {}   # path -> (size, primera_vez_con_ese_size)

    def es_estable(self, path):
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        ahora = time.time()
        prev = self._visto.get(path)
        if prev is None or prev[0] != size:
            self._visto[path] = (size, ahora)
            return False
        return size > 0 and (ahora - prev[1]) >= self.espera_seg

    def olvidar(self, path):
        self._visto.pop(path, None)


def descubrir_carpetas(videos_root):
    """Subcarpetas directas de videos_root que son un dataset_tipo conocido.
    Avisa (no aborta) por las que no lo son."""
    carpetas = []
    for nombre in sorted(os.listdir(videos_root)):
        ruta = os.path.join(videos_root, nombre)
        if not os.path.isdir(ruta):
            continue
        if nombre in ls.FPS_POR_CARPETA:
            carpetas.append(nombre)
        else:
            print(f"[AVISO] carpeta {nombre!r} no es un dataset_tipo conocido "
                  f"(labeling_strategy.FPS_POR_CARPETA) -- se salta.")
    return carpetas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_root")
    ap.add_argument("out_dir")
    ap.add_argument("--target-tubos", type=int, default=67000)
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--guardar-videos", action="store_true")
    ap.add_argument("--estable-seg", type=float, default=3.0,
                     help="segundos entre dos lecturas de tamaño para considerar un .mp4 'ya subido'")
    ap.add_argument("--poll-seg", type=float, default=5.0,
                     help="cuanto esperar antes de re-escanear si no hay ningun clip nuevo/estable")
    ap.add_argument("--idle-max-seg", type=float, default=300.0,
                     help="corta si pasa este tiempo sin encontrar NINGUN clip nuevo procesable "
                          "(asume que ya no van a llegar mas videos)")
    args = ap.parse_args()

    carpetas = descubrir_carpetas(args.videos_root)
    if not carpetas:
        sys.exit(f"No se encontro ninguna subcarpeta de dataset_tipo conocido en {args.videos_root}")
    print(f"Carpetas a monitorear: {carpetas}")

    fps_por_tipo = {c: ls.fps_de_carpeta(c) for c in carpetas}

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    anot_path = os.path.join(args.out_dir, "anotaciones.csv")
    ya_procesados = clips_ya_procesados(anot_path)
    n_ok = tubos_ya_generados(anot_path)
    if ya_procesados:
        print(f"RETOMANDO: {len(ya_procesados)} clips ya estan en {anot_path} ({n_ok} tubos), se saltan")

    anot = open(anot_path, "a" if os.path.exists(anot_path) else "w", newline="")
    escritor = csv.writer(anot)
    if anot.tell() == 0:
        escritor.writerow(["npy", "split", "marco", "x0", "y0", "x1", "y1", "fps", "clip_origen"])
        anot.flush()

    n_skip = 0
    idle_desde = None   # timestamp desde el que no se encuentra nada nuevo, None = recien encontramos algo
    rastreador = RastreadorEstabilidad(args.estable_seg)

    print(f"Objetivo: {args.target_tubos} tubos | ya tengo {n_ok}")

    while n_ok < args.target_tubos:
        encontro_algo_esta_pasada = False

        for dataset_tipo in carpetas:
            if n_ok >= args.target_tubos:
                break
            carpeta = os.path.join(args.videos_root, dataset_tipo)
            clips = sorted(glob.glob(os.path.join(carpeta, "*.mp4")))
            for clip in clips:
                if n_ok >= args.target_tubos:
                    break
                nombre_base_clip = os.path.basename(clip)
                clip_origen = f"{dataset_tipo}/{nombre_base_clip}"
                if clip_origen in ya_procesados:
                    continue
                if not rastreador.es_estable(clip):
                    continue   # todavia subiendo, o recien detectado -- se reintenta en la proxima pasada

                encontro_algo_esta_pasada = True
                rastreador.olvidar(clip)   # ya se va a marcar como procesado, no hace falta seguir rastreandolo
                idle_desde = None
                nombre_clip = os.path.splitext(nombre_base_clip)[0]
                marco = round(random.uniform(pp.MARCO_MIN_TRAIN, pp.MARCO_MAX_TRAIN), 3)
                fps = fps_por_tipo[dataset_tipo]

                try:
                    tubo, ventana, total_frames = extrae_tubo(modelo_pose, clip, marco, fps, n_frames=args.n_frames)
                except Exception as e:
                    print("ERR", clip, e)
                    tubo = None

                # marcar como procesado SIEMPRE (haya dado tubo o no) -- si no,
                # un clip que falla se reintenta en cada pasada para siempre
                ya_procesados.add(clip_origen)

                if tubo is None:
                    n_skip += 1
                    print(f"  [skip] {clip_origen} -- sin keypoints validos")
                    continue

                split = "val" if random.random() < args.val_frac else "train"
                base = nombre_seguro(f"{dataset_tipo}__{nombre_clip}")
                arr = tubo.permute(0, 2, 3, 1).cpu().numpy()

                guarda_npy_atomico(os.path.join(args.out_dir, base + ".npy"), arr)
                if args.guardar_videos or random.random() < PROB_VIDEO_QA:
                    video_path = os.path.join(args.out_dir, base + ".mp4")
                    h, w = arr.shape[1], arr.shape[2]
                    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
                    for fr in arr:
                        vw.write(fr[:, :, ::-1])
                    vw.release()

                x0, y0, x1, y1 = ventana
                escritor.writerow([f"{base}.npy", split, marco, x0, y0, x1, y1, fps, clip_origen])
                anot.flush()
                os.fsync(anot.fileno())
                n_ok += 1
                if n_ok % 25 == 0 or n_ok == args.target_tubos:
                    print(f"  [ok] {n_ok}/{args.target_tubos} tubos | ultimo={clip_origen} split={split}", flush=True)

        if not encontro_algo_esta_pasada and n_ok < args.target_tubos:
            if idle_desde is None:
                idle_desde = time.time()
            idle_transcurrido = time.time() - idle_desde
            if idle_transcurrido >= args.idle_max_seg:
                print(f"\n[FIN-IDLE] {idle_transcurrido:.0f}s sin clips nuevos/estables -- "
                      f"asumo que ya no van a llegar mas videos. {n_ok}/{args.target_tubos} tubos generados.")
                break
            print(f"[ESPERANDO] sin clips nuevos esta pasada (idle={idle_transcurrido:.0f}s/{args.idle_max_seg:.0f}s) "
                  f"-- durmiendo {args.poll_seg:.0f}s", flush=True)
            time.sleep(args.poll_seg)

    anot.close()
    print(f"\n{n_ok} tubos generados ({n_skip} clips saltados por falta de keypoints) -> {anot_path}")


if __name__ == "__main__":
    main()

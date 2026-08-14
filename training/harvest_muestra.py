#!/usr/bin/env python3
"""harvest_muestra.py -- cosecha una MUESTRA ALEATORIA de N videos tomados
de VARIAS carpetas de videos_for_train/ (a diferencia de harvest_pose_test.py,
que procesa una sola carpeta a la vez). Reusa toda la logica de extraccion y
resumibilidad de harvest_pose_test.py -- este script solo arma el pool
multi-carpeta, hace el sorteo, y agrega dataset_tipo a la tabla.

YA NO resuelve labels aca (ver harvest_pose_test.py y
training/genera_anotaciones_por_cabeza.py) -- por eso TODAS las carpetas de
videos_for_train/ entran al sorteo, incluidas manipulaciones_tiendas/
reingesta_dia1_socrates/reingesta_dia_platonv1 (antes CARPETAS_EXCLUIDAS
porque labeling_strategy.aplica_reglas_etiquetado() no tiene reglas
multiclase para esas y se hubiera detenido en seco; eso ya no aplica aca,
solo le importa a quien resuelva el label multiclase despues -- esas 3
carpetas van a quedar sin label multiclase valido, pero SI tienen uno
binario, ver labeling_strategy_binaria.py). Unica condicion real para
entrar al sorteo: que dataset_tipo tenga FPS conocido (ls.fps_de_carpeta),
eso no depende de la cabeza.

Uso:
  python3 training/harvest_muestra.py <videos_for_train_dir> <out_dir> [--n 500] [--seed 0]
"""
import csv
import errno
import os
import sys
import glob
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harvest_pose_test as hpt
import labeling_strategy as ls
from ultralytics import YOLO


def clips_ya_procesados_multi(anotaciones_path):
    """Version de hpt.clips_ya_procesados() para el esquema multi-carpeta:
    la clave es dataset_tipo+clip_origen (no solo clip_origen), porque dos
    carpetas distintas podrian, en teoria, compartir un nombre de archivo."""
    if not os.path.exists(anotaciones_path):
        return set()
    with open(anotaciones_path) as f:
        return {(r["dataset_tipo"], r["clip_origen"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_for_train_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=500,
                     help="TOTAL de tubos que debe tener anotaciones.csv al terminar -- si ya hay "
                          "N_previos de una corrida anterior en out_dir, se sortean solo los que faltan "
                          "(N - N_previos) de clips NUEVOS, sin reprocesar ni volver a sortear los que ya estan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--guardar-videos", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--n-frames", type=int, default=32,
                     help="frames por tubo -- ver --n-frames en harvest_pose_test.py")
    args = ap.parse_args()

    carpetas = sorted(
        d for d in os.listdir(args.videos_for_train_dir)
        if os.path.isdir(os.path.join(args.videos_for_train_dir, d))
    )
    print(f"carpetas incluidas en el sorteo ({len(carpetas)}): {carpetas}")

    anot_path = os.path.join(args.out_dir, "anotaciones.csv")
    ya_procesados = clips_ya_procesados_multi(anot_path)   # (dataset_tipo, clip_origen) de una corrida anterior -- AJUSTA EL POOL DE SORTEO, no solo el skip
    if ya_procesados:
        print(f"{len(ya_procesados)} tubos ya estan en {anot_path} -- se reusan, no se reprocesan")

    pool = []
    for carpeta in carpetas:
        fps = ls.fps_de_carpeta(carpeta)  # se detiene aca si a alguna le falta el fps -- antes de sortear nada
        clips = glob.glob(os.path.join(args.videos_for_train_dir, carpeta, "*.mp4"))
        pool += [(c, carpeta, fps) for c in clips if (carpeta, os.path.basename(c)) not in ya_procesados]
    print(f"pool disponible (ya excluidos los {len(ya_procesados)} ya hechos): {len(pool)} clips en {len(carpetas)} carpetas")

    n_faltantes = max(0, args.n - len(ya_procesados))
    print(f"objetivo: {args.n} tubos en total -> faltan {n_faltantes} NUEVOS por sortear/procesar")

    random.seed(args.seed)
    muestra = random.sample(pool, min(n_faltantes, len(pool)))
    print(f"muestra sorteada: {len(muestra)} clips nuevos (seed={args.seed})")

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    anot = open(anot_path, "a" if ya_procesados or os.path.exists(anot_path) else "w", newline="")
    escritor = csv.writer(anot)
    if not ya_procesados and anot.tell() == 0:
        # SIN "label" -- ver docstring del modulo y de harvest_pose_test.py:
        # se resuelve a posteriori (training/genera_anotaciones_por_cabeza.py)
        # a partir de dataset_tipo+clip_origen, ya en esta misma tabla.
        escritor.writerow(["npy", "split", "marco", "x0", "y0", "x1", "y1",
                            "fps", "dataset_tipo", "clip_origen"])
        anot.flush()

    n_ok = n_skip = n_saltados_ya = 0
    for clip, dataset_tipo, fps in muestra:
        nombre_base_clip = os.path.basename(clip)
        if (dataset_tipo, nombre_base_clip) in ya_procesados:
            n_saltados_ya += 1
            continue
        nombre_clip = os.path.splitext(nombre_base_clip)[0]
        marco = round(random.uniform(hpt.pp.MARCO_MIN_TRAIN, hpt.pp.MARCO_MAX_TRAIN), 3)

        try:
            tubo, ventana, total_frames = hpt.extrae_tubo(modelo_pose, clip, marco, fps, n_frames=args.n_frames)
        except Exception as e:
            print("ERR", clip, e)
            tubo = None
        if tubo is None:
            n_skip += 1
            print(f"  [skip] ({dataset_tipo}) {nombre_base_clip} -- sin keypoints validos")
            continue

        split = "val" if random.random() < args.val_frac else "train"
        base = hpt.nombre_seguro(f"{dataset_tipo}__{nombre_clip}")
        arr = tubo.permute(0, 2, 3, 1).cpu().numpy()

        try:
            hpt.guarda_npy_atomico(os.path.join(args.out_dir, base + ".npy"), arr)
            if args.guardar_videos or random.random() < hpt.PROB_VIDEO_QA:
                import cv2
                video_path = os.path.join(args.out_dir, base + ".mp4")
                h, w = arr.shape[1], arr.shape[2]
                vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
                for fr in arr:
                    vw.write(fr[:, :, ::-1])
                vw.release()
                print(f"         (QA) video guardado: {base}.mp4")
        except OSError as e:
            if e.errno == errno.ENOSPC:
                anot.close()
                print(f"\nDISCO LLENO al escribir {base}.npy -- {n_ok} tubos ya quedaron guardados "
                      f"y confirmados en {anot_path}. Libera espacio y corre EXACTAMENTE el mismo "
                      f"comando de nuevo -- retoma solo.")
                sys.exit(1)
            raise

        x0, y0, x1, y1 = ventana
        escritor.writerow([f"{base}.npy", split, marco, x0, y0, x1, y1, fps, dataset_tipo, nombre_base_clip])
        anot.flush()
        os.fsync(anot.fileno())
        n_ok += 1
        print(f"  [ok]   ({dataset_tipo}) {nombre_base_clip} -> {base}.npy  split={split}  tubo={arr.shape}")

    anot.close()
    print(f"\n{n_ok} tubos generados, {n_skip} saltados, {n_saltados_ya} ya estaban de una corrida "
          f"anterior -> {anot_path}")


if __name__ == "__main__":
    main()

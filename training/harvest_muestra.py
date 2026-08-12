#!/usr/bin/env python3
"""harvest_muestra.py -- cosecha una MUESTRA ALEATORIA de N videos tomados
de VARIAS carpetas de videos_for_train/ (a diferencia de harvest_pose_test.py,
que procesa una sola carpeta a la vez). Reusa toda la logica de extraccion,
etiquetado y resumibilidad de harvest_pose_test.py -- este script solo arma
el pool multi-carpeta, hace el sorteo, y agrega dataset_tipo a la tabla.

Carpetas EXCLUIDAS del sorteo: manipulaciones_tiendas, reingesta_dia1_socrates,
reingesta_dia_platonv1 -- labeling_strategy.py todavia no tiene reglas de
etiquetado para esas, aplica_reglas_etiquetado() se detendria ahi mismo.

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

CARPETAS_EXCLUIDAS = {"manipulaciones_tiendas", "reingesta_dia1_socrates", "reingesta_dia_platonv1"}


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
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--guardar-videos", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    carpetas = sorted(
        d for d in os.listdir(args.videos_for_train_dir)
        if os.path.isdir(os.path.join(args.videos_for_train_dir, d)) and d not in CARPETAS_EXCLUIDAS
    )
    print(f"carpetas incluidas en el sorteo ({len(carpetas)}): {carpetas}")
    print(f"carpetas EXCLUIDAS (sin reglas de labeling todavia): {sorted(CARPETAS_EXCLUIDAS)}")

    pool = []
    for carpeta in carpetas:
        fps = ls.fps_de_carpeta(carpeta)  # se detiene aca si a alguna le falta el fps -- antes de sortear nada
        clips = glob.glob(os.path.join(args.videos_for_train_dir, carpeta, "*.mp4"))
        pool += [(c, carpeta, fps) for c in clips]
    print(f"pool total: {len(pool)} clips en {len(carpetas)} carpetas")

    random.seed(args.seed)
    muestra = random.sample(pool, min(args.n, len(pool)))
    print(f"muestra sorteada: {len(muestra)} clips (seed={args.seed})")

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    anot_path = os.path.join(args.out_dir, "anotaciones.csv")
    ya_procesados = clips_ya_procesados_multi(anot_path)
    if ya_procesados:
        print(f"RETOMANDO: {len(ya_procesados)} clips ya estan en {anot_path}, se saltan")
    anot = open(anot_path, "a" if ya_procesados or os.path.exists(anot_path) else "w", newline="")
    escritor = csv.writer(anot)
    if not ya_procesados and anot.tell() == 0:
        escritor.writerow(["npy", "label", "split", "marco", "x0", "y0", "x1", "y1",
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
            tubo, ventana, total_frames = hpt.extrae_tubo(modelo_pose, clip, marco, fps)
        except (ls.DatasetTipoDesconocido, ls.LabelNoResuelto):
            raise   # ADVERTENCIA + PAUSA real: no se atrapa
        except Exception as e:
            print("ERR", clip, e)
            tubo = None
        if tubo is None:
            n_skip += 1
            print(f"  [skip] ({dataset_tipo}) {nombre_base_clip} -- sin keypoints validos")
            continue

        results = {
            "frame_dir": nombre_base_clip,
            "dataset_tipo": dataset_tipo,
            "label": ls.extraer_label_inicial(nombre_base_clip),
            "total_frames": total_frames,
        }
        ls.aplica_reglas_etiquetado(results)
        label = results["label"]

        split = "val" if random.random() < args.val_frac else "train"
        base = f"L{label:02d}__{dataset_tipo}__{nombre_clip}"
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
        escritor.writerow([f"{base}.npy", label, split, marco, x0, y0, x1, y1, fps, dataset_tipo, nombre_base_clip])
        anot.flush()
        os.fsync(anot.fileno())
        n_ok += 1
        print(f"  [ok]   ({dataset_tipo}) {nombre_base_clip} -> {base}.npy  label={label}  split={split}  tubo={arr.shape}")

    anot.close()
    print(f"\n{n_ok} tubos generados, {n_skip} saltados, {n_saltados_ya} ya estaban de una corrida "
          f"anterior -> {anot_path}")


if __name__ == "__main__":
    main()

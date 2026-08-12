#!/usr/bin/env python3
"""harvest_pose_test.py -- PRUEBA de cosecha con la nueva estrategia (pose_pipeline).

Procesa un grupo chico de videos para confirmar que el pipeline completo
funciona antes de correrlo sobre el dataset real. Etiquetas ALEATORIAS
(no hay labeling todavia -- solo se esta probando la mecanica).

Mantiene la ESTRATEGIA TEMPORAL de harvest_v14.py sin cambios: centro
temporal del clip, ventana de +-2s (los "4 segundos" con los que trabaja
el modelo, WINDOW_S en config.py). Lo que cambia es todo lo demas:

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

Uso:
  python3 training/harvest_pose_test.py <carpeta_videos> <out_dir> [--pose-model PATH]
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
from ultralytics import YOLO

FPS_REAL = 8   # metadata del contenedor puede decir otra cosa (11fps visto en ffprobe) -- IGNORAR, es 8
DEVICE = "cuda"


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


def extrae_tubo(modelo_pose, clip, marco):
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
        return None, None

    n = len(frames)
    center = n // 2
    radio = FPS_REAL * 2   # +-2s = ~4s, el WINDOW_S del modelo
    idx_ventana = [i for i in range(n) if abs(i - center) <= radio]
    if len(idx_ventana) < 4:
        idx_ventana = list(range(n))   # mismo fallback que harvest_v14.py: si la ventana quedo muy chica, usar todo el clip

    kp_clip = np.stack([keypoints_por_frame_max_confianza(modelo_pose, frames[i]) for i in idx_ventana])
    ventana = pp.bbox_desde_keypoints(kp_clip, w, h, marco=marco)
    if ventana is None:
        return None, None

    frames_gpu = [frame_a_gpu(frames[i]) for i in idx_ventana]
    tubo_store = pp.recorta_y_redimensiona_gpu(frames_gpu, ventana, out_size=pp.STORE_SIZE)
    return tubo_store, ventana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta_videos")
    ap.add_argument("out_dir")
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    modelo_pose = YOLO(args.pose_model)

    clips = sorted(glob.glob(os.path.join(args.carpeta_videos, "*.mp4")))
    print(f"{len(clips)} clips en {args.carpeta_videos}")

    man = open(os.path.join(args.out_dir, "manifest.csv"), "w")
    man.write("npy,clase_ALEATORIA,marco,ventana,clip\n")

    n_ok = n_skip = 0
    for clip in clips:
        marco = round(random.uniform(pp.MARCO_MIN_TRAIN, pp.MARCO_MAX_TRAIN), 3)
        try:
            tubo, ventana = extrae_tubo(modelo_pose, clip, marco)
        except Exception as e:
            print("ERR", clip, e)
            tubo = None
        if tubo is None:
            n_skip += 1
            print(f"  [skip] {os.path.basename(clip)} -- sin keypoints validos")
            continue

        clase = random.choice(["hurto", "normal"])   # ALEATORIO, solo para probar la mecanica
        nombre = f"{clase}__{os.path.basename(clip)}.npy"
        arr = tubo.permute(0, 2, 3, 1).cpu().numpy()   # T,C,H,W -> T,H,W,C, para que quede igual formato que el dataset viejo
        np.save(os.path.join(args.out_dir, nombre), arr)
        man.write(f"{nombre},{clase},{marco},{ventana},{os.path.basename(clip)}\n")
        n_ok += 1
        print(f"  [ok]   {os.path.basename(clip)} -> {nombre}  marco={marco}  ventana={ventana}  tubo={arr.shape}")

    man.close()
    print(f"\n{n_ok} tubos generados, {n_skip} saltados -> {args.out_dir}")


if __name__ == "__main__":
    main()

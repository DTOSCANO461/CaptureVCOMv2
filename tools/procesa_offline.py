#!/usr/bin/env python3
"""Corre el pipeline de deteccion (YOLO11+ByteTrack -> tubos 4s -> VideoMAE)
sobre ficheros de video locales (no RTSP), tratando cada fichero como una
camara independiente. Pensado para probar el modelo sobre grabaciones ya
existentes (p.ej. videos/<caso>/*.ts) sin necesitar config.py de tienda.

Uso:
  python3 tools/procesa_offline.py videos/paco_robo1/*.ts
  python3 tools/procesa_offline.py videos/paco_robo1/*.ts --umbral 0.45 --out salida/paco_robo1
"""
import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from camara_worker import make_tube
from modelo import ModeloV5

YOLO_MODEL = os.path.join(ROOT, "model", "yolo11s.pt")
CKPT = os.path.join(ROOT, "model", "videomae_v6_last.pt")

WINDOW_S = 4.0
STRIDE_S = 0.7
SCORE_EVERY_S = 1.5
MIN_TRACK_S = 2.0
SKIP_FRAMES = 2
ZOOM = 0.6
EMIT_UMBRAL = 0.45
EMIT_VALLE = 0.15
EMIT_UPGRADE = 0.25
EMIT_COOLDOWN_S = 8.0
REARM_QUIET_S = 4.0


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def guarda_alerta(cam_id, tube_rgb, score, track_id, t_s, alertas_dir):
    os.makedirs(alertas_dir, exist_ok=True)
    nombre = f"{cam_id}_t{t_s:07.1f}s_id{track_id}_p{score:.3f}.mp4"
    path = os.path.join(alertas_dir, nombre)
    h, w = tube_rgb.shape[1], tube_rgb.shape[2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for fr in tube_rgb:
        vw.write(fr[:, :, ::-1])
    vw.release()
    return nombre


def procesa_video(path, cam_id, det, modelo, alertas_dir, wcsv, fcsv, umbral):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log(f"[{cam_id}] no se pudo abrir {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"[{cam_id}] {os.path.basename(path)}: {w}x{h} @ {fps:.1f}fps, {n_total} frames")

    buf = int(WINDOW_S * fps / max(SKIP_FRAMES, 1)) + 4
    ring = deque(maxlen=buf)
    tracks = {}
    ultima_emision = {}
    ultimo_alto = {}
    bajo_valle = {}

    next_t, fi = 0.0, 0
    t0 = time.time()
    n_alertas = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        fi += 1
        if SKIP_FRAMES > 1 and (fi - 1) % SKIP_FRAMES:
            continue

        res = det.track(frame, classes=[0], conf=0.30, persist=True,
                         verbose=False, imgsz=960, tracker="bytetrack.yaml")[0]
        ring.append((t, frame))
        if res.boxes is not None and res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist()):
                tracks.setdefault(tid, deque(maxlen=buf)).append((t, tuple(b)))

        if t < next_t:
            continue
        next_t = t + STRIDE_S

        for tid, dq in list(tracks.items()):
            if dq and dq[-1][0] < t - 1.5:
                del tracks[tid]
                continue
            if len(dq) < 2 or (dq[-1][0] - dq[0][0]) < MIN_TRACK_S:
                continue
            k = (cam_id, tid)

            win = [(ft, fr) for (ft, fr) in ring if ft >= t - WINDOW_S]
            wb = [bx for (ft, bx) in dq if ft >= t - WINDOW_S]
            if len(win) < 4 or len(wb) < 2:
                continue
            frames_bgr = [f for (_, f) in win]
            tube = make_tube(frames_bgr, wb, w, h)
            p = float(modelo.score([tube])[0])

            if p < EMIT_VALLE:
                bajo_valle[k] = True
            if p >= umbral:
                gap = t - ultimo_alto.get(k, -1e9)
                ultimo_alto[k] = t
                pt, ps = ultima_emision.get(k, (-1e9, 0.0))
                if (gap >= REARM_QUIET_S or p >= ps + EMIT_UPGRADE
                        or bajo_valle.get(k, False) or t - pt >= EMIT_COOLDOWN_S):
                    clip = guarda_alerta(cam_id, tube, p, tid, t, alertas_dir)
                    wcsv.writerow({"camara": cam_id, "track": tid, "t_s": round(t, 1),
                                   "score": round(p, 4), "clip": clip})
                    fcsv.flush()
                    ultima_emision[k] = (t, p)
                    bajo_valle[k] = False
                    n_alertas += 1
                    log(f"[{cam_id}] ALERTA t={t:.1f}s track={tid} p={p:.3f} -> {clip}")

        if fi % 500 < SKIP_FRAMES:
            elapsed = time.time() - t0
            log(f"[{cam_id}] {fi}/{n_total} frames ({100*fi/max(n_total,1):.0f}%), "
                f"{elapsed:.0f}s transcurridos")

    cap.release()
    log(f"[{cam_id}] terminado: {n_alertas} alertas, {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="ficheros de video (uno por camara)")
    ap.add_argument("--out", default=os.path.join(ROOT, "salida_offline"))
    ap.add_argument("--umbral", type=float, default=EMIT_UMBRAL)
    args = ap.parse_args()

    if not os.path.exists(YOLO_MODEL) or not os.path.exists(CKPT):
        sys.exit(f"faltan pesos: {YOLO_MODEL} / {CKPT}")

    from ultralytics import YOLO
    log("cargando YOLO11s (tracking)...")
    det = YOLO(YOLO_MODEL)
    log("cargando VideoMAE v6 (puntuador)...")
    modelo = ModeloV5(CKPT, zoom=ZOOM)
    log(f"modelos cargados. umbral de emision = {args.umbral}")

    os.makedirs(args.out, exist_ok=True)
    alertas_dir = os.path.join(args.out, "alertas")
    csv_path = os.path.join(args.out, "alertas.csv")
    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    wcsv = csv.DictWriter(fcsv, fieldnames=["camara", "track", "t_s", "score", "clip"])
    if nuevo:
        wcsv.writeheader()
        fcsv.flush()

    for path in args.videos:
        cam_id = os.path.splitext(os.path.basename(path))[0]
        procesa_video(path, cam_id, det, modelo, alertas_dir, wcsv, fcsv, args.umbral)

    fcsv.close()
    log(f"listo. alertas y csv en {args.out}")


if __name__ == "__main__":
    main()

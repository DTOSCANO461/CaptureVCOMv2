#!/usr/bin/env python3
"""Corre v5 sobre TODOS los clips historicos de Family Cash A que tienen
etiqueta humana (video_data.json: 'si'=hurto confirmado, 'no'=falso positivo
confirmado) y todavia tienen el video en disco.

Para cada clip: recorta a la persona mas cercana a la coordenada del nombre
de fichero (formato OUTPUT_coord_X__Y, igual que extract_tubes.py), tubo de
16 frames en una ventana de 4s, puntua con v5.

Salida: cuantos 'no' (falsos positivos reales) v5 tambien descarta, y
cuantos 'si' (hurtos reales) v5 tambien detecta.
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from modelo import ModeloV5

BASE_UPLOADS = "/root/CAPTURE_VIDEOS/uploads"
VIDEO_DATA = "/root/CAPTURE_VIDEOS/video_data.json"
N_FRAMES = 16
WINDOW_S = 4.0


def pick_person(dets, w, h, target=None):
    if not dets:
        return None
    if target is not None:
        tx, ty = target
        return min(dets, key=lambda d: ((d[0]+d[2])/2 - tx)**2 + ((d[1]+d[3])/2 - ty)**2)
    cx, cy = w / 2, h / 2

    def score(d):
        area = (d[2]-d[0]) * (d[3]-d[1])
        dcx, dcy = (d[0]+d[2])/2, (d[1]+d[3])/2
        dist = np.hypot(dcx - cx, dcy - cy) / np.hypot(cx, cy)
        return area * (1.0 - 0.5 * dist) * d[4]
    return max(dets, key=score)


def sample_indices(n_total, fps, n=N_FRAMES, window_s=WINDOW_S):
    dur = n_total / fps if fps > 0 else 0
    if dur <= window_s:
        lo, hi = 0, max(n_total - 1, 1)
    else:
        c = n_total / 2
        half = window_s * fps / 2
        lo, hi = c - half, c + half - 1
    return np.linspace(lo, hi, n).round().astype(int).clip(0, n_total - 1)


def extrae_tubo(path, det):
    cap = cv2.VideoCapture(path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 11
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_total < 2 or w == 0 or h == 0:
        cap.release()
        return None

    idxs = sample_indices(n_total, fps)
    grabbed = {}
    i = 0
    needed = set(idxs.tolist())
    while cap.isOpened() and needed:
        ok, fr = cap.read()
        if not ok:
            break
        if i in needed:
            grabbed[i] = fr
            needed.discard(i)
        i += 1
    cap.release()
    if not grabbed:
        return None
    frames = [grabbed.get(int(ix), grabbed[min(grabbed, key=lambda k: abs(k-ix))])
              for ix in idxs]

    m = re.search(r"OUTPUT_coord_(\d+)__(\d+)", os.path.basename(path))
    target = (int(m.group(1)), int(m.group(2))) if m else None

    keys = [frames[4], frames[8], frames[12]]
    res = det(keys, classes=[0], conf=0.25, verbose=False, imgsz=640)
    dets_all = []
    for rr in res:
        dets_all.append([tuple(b.xyxy[0].tolist()) + (float(b.conf[0]),) for b in rr.boxes])

    boxes = [pick_person(d, w, h, target) for d in dets_all]
    boxes = [b for b in boxes if b is not None]
    if boxes:
        cx = float(np.median([(b[0]+b[2])/2 for b in boxes]))
        cy = float(np.median([(b[1]+b[3])/2 for b in boxes]))
        side = float(np.median([max(b[2]-b[0], b[3]-b[1]) for b in boxes])) * 1.9
        side = max(side, 200)
    else:
        cx, cy, side = w/2, h/2, min(w, h) * 0.7

    side = min(side, min(w, h))
    x1 = int(np.clip(cx - side/2, 0, w - side))
    y1 = int(np.clip(cy - side/2, 0, h - side))
    s = int(side)
    tube = np.stack([
        cv2.resize(fr[y1:y1+s, x1:x1+s], (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        for fr in frames
    ]).astype(np.uint8)
    return tube


def main():
    from ultralytics import YOLO
    det = YOLO(C.YOLO_MODEL)
    modelo = ModeloV5(C.CKPT, zoom=C.ZOOM)

    labels = json.load(open(VIDEO_DATA))
    casos = []
    for carpeta, info in labels.items():
        ev = info.get("evento")
        if ev not in ("si", "no"):
            continue
        d = os.path.join(BASE_UPLOADS, carpeta)
        clips = sorted(glob.glob(os.path.join(d, "*.mp4")))
        if clips:
            casos.append((carpeta, ev, clips))

    print(f"{len(casos)} carpetas etiquetadas con video disponible "
          f"({sum(1 for c in casos if c[1]=='si')} si / "
          f"{sum(1 for c in casos if c[1]=='no')} no)")
    print(f"umbral v5 = {C.UMBRAL}\n")

    resultados = []
    for i, (carpeta, ev, clips) in enumerate(casos):
        scores = []
        for clip in clips:
            try:
                tubo = extrae_tubo(clip, det)
                if tubo is not None:
                    p = float(modelo.score([tubo])[0])
                    scores.append(p)
            except Exception as e:
                print(f"  [error] {clip}: {e}")
        if scores:
            resultados.append((carpeta, ev, max(scores), scores))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(casos)} procesados...", flush=True)

    si = [(c, m, s) for c, ev, m, s in resultados if ev == "si"]
    no = [(c, m, s) for c, ev, m, s in resultados if ev == "no"]

    detectados = sum(1 for _, m, _ in si if m >= C.UMBRAL)
    eliminados = sum(1 for _, m, _ in no if m < C.UMBRAL)

    print(f"\n=== HURTOS CONFIRMADOS ('si'), n={len(si)} ===")
    print(f"v5 los detecta (score >= {C.UMBRAL}): {detectados}/{len(si)} "
          f"({100*detectados/max(len(si),1):.1f}%)")
    for c, m, s in sorted(si, key=lambda x: x[1]):
        marca = "MISS" if m < C.UMBRAL else "ok"
        print(f"  [{marca:4}] {c}  score={m:.4f}")

    print(f"\n=== FALSOS POSITIVOS CONFIRMADOS ('no'), n={len(no)} ===")
    print(f"v5 los elimina (score < {C.UMBRAL}): {eliminados}/{len(no)} "
          f"({100*eliminados/max(len(no),1):.1f}%)")
    for c, m, s in sorted(no, key=lambda x: -x[1]):
        marca = "SIGUE DISPARANDO" if m >= C.UMBRAL else "eliminado"
        print(f"  [{marca:16}] {c}  score={m:.4f}")

    with open("/opt/capturevcomv2/data/prueba_historica.json", "w") as f:
        json.dump({"si": [(c, m) for c, m, _ in si],
                  "no": [(c, m) for c, m, _ in no],
                  "umbral": C.UMBRAL}, f, indent=1)
    print(f"\nguardado en /opt/capturevcomv2/data/prueba_historica.json")


if __name__ == "__main__":
    main()

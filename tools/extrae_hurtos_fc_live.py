#!/usr/bin/env python3
"""Corre EN EL PC DE FAMILY CASH. Extrae tubos de hurto de las grabaciones
crudas reales (data/raw/camN/) de una ventana de tiempo concreta -- para la
sesion de Jose grabando hurtos en vivo. Sigue con YOLO+ByteTrack cada
fichero de 60s por separado (igual que hace el pipeline en vivo, cada raw
es una captura independiente), genera ventanas deslizantes WINDOW_S/STRIDE_S
igual que produccion, y anonimiza a cualquiera que NO sea la persona mas
centrada en cada ventana (asume que Jose es el protagonista de la accion).

Uso:
  /opt/capturevcomv2/venv/bin/python3 extrae_hurtos_fc_live.py cam0 "17:16" "17:18"
  (hora local del PC, formato HH:MM; coge los raw de esa camara con mtime
  entre 2 min antes del inicio y 1 min despues del fin -- el fin es
  obligatorio para no colar de mas si el script se corre tarde)
"""
import glob
import os
import sys
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

WINDOW_S = 4.0
STRIDE_S = 2.0
MIN_TRACK_S = 2.0
YOLO_MODEL = "/opt/capturevcomv2/model/yolo11s.pt"
RAW_BASE = "/opt/capturevcomv2/data/raw"
OUT_DIR = os.environ.get("OUT_DIR", "/opt/capturevcomv2/data/hurtos_fc_lunes_backup")
_pose = None


def _load_pose():
    global _pose
    if _pose is None:
        from ultralytics import YOLO
        _pose = YOLO("yolo11n-pose.pt")
    return _pose


def cara_mask(frame_rgb, kp, conf_min=0.3):
    h, w = frame_rgb.shape[:2]
    nariz, oj_i, oj_d, or_i, or_d = kp[0], kp[1], kp[2], kp[3], kp[4]
    ojos = [p for p in (oj_i, oj_d) if p[2] > conf_min]
    orejas = [p for p in (or_i, or_d) if p[2] > conf_min]
    if nariz[2] <= conf_min or not ojos:
        return None
    xs = [nariz[0]] + [p[0] for p in ojos]
    ys_ojos = [p[1] for p in ojos]
    cy_ojos = float(np.mean(ys_ojos))
    cx = float(np.mean(xs))
    if len(ojos) == 2:
        ancho = abs(ojos[0][0] - ojos[1][0])
    elif len(orejas) == 2:
        ancho = abs(orejas[0][0] - orejas[1][0]) * 0.6
    else:
        ancho = max(w, h) * 0.12
    ancho = max(ancho, 6)
    alto_arriba = ancho * 0.55
    alto_abajo = abs(nariz[1] - cy_ojos) * 2.3 + ancho * 0.35
    cy = cy_ojos + (alto_abajo - alto_arriba) / 2
    ax = int(round(ancho * 1.15))
    ay = int(round((alto_arriba + alto_abajo) / 2))
    if ax < 3 or ay < 3:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def anonimiza_frame_proteger_centro(frame_rgb, det):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    res = det(frame_bgr, verbose=False, conf=0.25)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return frame_rgb
    h, w = frame_rgb.shape[:2]
    ccx, ccy = w / 2, h / 2
    candidatos = []
    for kp in res.keypoints.data.cpu().numpy():
        m = cara_mask(frame_rgb, kp)
        if m is None:
            continue
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        dist = (cx - ccx) ** 2 + (cy - ccy) ** 2
        candidatos.append((dist, m))
    if not candidatos:
        return frame_rgb
    candidatos.sort(key=lambda t: t[0])
    candidatos = candidatos[1:]  # protege al mas centrado (Jose)
    if not candidatos:
        return frame_rgb
    mask_total = np.zeros(frame_bgr.shape[:2], np.uint8)
    for _, m in candidatos:
        mask_total |= m
    mask_total = cv2.dilate(mask_total, np.ones((3, 3), np.uint8))
    out_bgr = cv2.inpaint(frame_bgr, mask_total, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def make_tube(frames_bgr, boxes, w, h, n_frames=16, side_out=256):
    idx = np.linspace(0, len(frames_bgr) - 1, n_frames).round().astype(int)
    bs = np.array(boxes, dtype=float)
    cx = np.median((bs[:, 0] + bs[:, 2]) / 2)
    cy = np.median((bs[:, 1] + bs[:, 3]) / 2)
    side = max(float(np.median(np.maximum(bs[:, 2] - bs[:, 0], bs[:, 3] - bs[:, 1]))) * 1.9, 200)
    side = min(side, min(w, h))
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    s = int(side)
    out = []
    for i in idx:
        crop = frames_bgr[i][y1:y1 + s, x1:x1 + s]
        rz = cv2.resize(crop, (side_out, side_out), interpolation=cv2.INTER_AREA)
        out.append(rz[:, :, ::-1])
    return np.ascontiguousarray(np.stack(out))


def procesa_fichero(path, det):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not w or not h:
        cap.release()
        return []
    buf = int(WINDOW_S * fps) + 4
    ring = deque(maxlen=max(buf, 50))
    tracks = {}
    resultados = []
    next_t, fi, t = 0.0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        fi += 1
        res = det.track(frame, classes=[0], conf=0.30, persist=True,
                        verbose=False, imgsz=640, tracker="bytetrack.yaml")[0]
        ring.append((t, frame))
        if res.boxes is not None and res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist()):
                tracks.setdefault(tid, deque(maxlen=max(buf, 50))).append((t, tuple(b)))
        if t < next_t:
            continue
        next_t = t + STRIDE_S
        for tid, dq in list(tracks.items()):
            if len(dq) < 2 or (dq[-1][0] - dq[0][0]) < MIN_TRACK_S:
                continue
            win = [(ft, fr) for (ft, fr) in ring if ft >= t - WINDOW_S]
            wb = [bx for (ft, bx) in dq if ft >= t - WINDOW_S]
            if len(win) < 4 or len(wb) < 2:
                continue
            frames_bgr = [f for (_, f) in win]
            tubo = make_tube(frames_bgr, wb, w, h)
            resultados.append((f"tid{tid}_t{t:.1f}", tubo))
    cap.release()
    return resultados


def main():
    cam = sys.argv[1]
    hora_ini_str = sys.argv[2]
    hora_fin_str = sys.argv[3]
    hoy = datetime.now().date()
    hh, mm = map(int, hora_ini_str.split(":"))
    t_inicio = datetime.combine(hoy, datetime.min.time()).replace(hour=hh, minute=mm) - timedelta(minutes=2)
    hh2, mm2 = map(int, hora_fin_str.split(":"))
    t_fin = datetime.combine(hoy, datetime.min.time()).replace(hour=hh2, minute=mm2) + timedelta(minutes=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    det_track = None
    from ultralytics import YOLO
    det_track = YOLO(YOLO_MODEL)
    det_pose = _load_pose()

    camdir = f"{RAW_BASE}/{cam}"
    ficheros = sorted(glob.glob(f"{camdir}/*.mp4"))
    ficheros = [p for p in ficheros
                if t_inicio <= datetime.fromtimestamp(os.path.getmtime(p)) <= t_fin]
    print(f"{len(ficheros)} ficheros entre {t_inicio} y {t_fin} en {camdir}", flush=True)

    total = 0
    for p in ficheros:
        nombre = os.path.splitext(os.path.basename(p))[0]
        ventanas = procesa_fichero(p, det_track)
        for suf, tubo in ventanas:
            anon = np.empty_like(tubo)
            for i in range(tubo.shape[0]):
                anon[i] = anonimiza_frame_proteger_centro(tubo[i], det_pose)
            npy_name = f"hurtos_fc_lunes__{nombre}__{suf}.mp4.npy"
            np.save(os.path.join(OUT_DIR, npy_name), anon.astype(np.uint8))
            total += 1
        print(f"{nombre}: {len(ventanas)} ventanas", flush=True)

    print(f"\nOK: {total} tubos en {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

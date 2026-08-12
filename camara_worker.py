#!/usr/bin/env python3
"""Un HILO por cámara (no proceso): conecta al RTSP, sigue personas
(YOLO+ByteTrack, un YOLO11s pequeño y propio por hilo), graba en crudo, y
cuando tiene una ventana de 4s lista la mete en la cola compartida para que
el hilo de puntuación (un único modelo v5 en VRAM) la evalúe.

Por qué hilos y no procesos: v5 (VideoMAE-Base) pesa ~350MB + contexto CUDA
propio por proceso (otros ~400-600MB). Con 10 procesos independientes, cada
uno cargando su propia copia, se arriesga a agotar los 16GB de la GPU y que
una caida en cascada tumbe las 10 camaras a la vez. Con un solo proceso y
un solo modelo compartido, el coste de VRAM es ~1/10 y un fallo de lectura
de una camara no afecta a las demas ni al modelo.

El YOLO de tracking (pequeño, ~19MB) SI se duplica por hilo porque cada
camara necesita su propio estado de tracker (ByteTrack no se puede compartir
entre streams distintos sin mezclar los IDs).
"""
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config as C

MIN_MOVE_REL = 0.20


def dentro_de_horario():
    """Horario de apertura de la tienda (mismo que usaba CAPTURE). Fuera de
    horario no tiene sentido conectar a las camaras ni puntuar."""
    dia = datetime.now().isoweekday()  # 1=lunes ... 7=domingo
    hora = datetime.now().hour
    ini, fin = C.HORARIO.get(dia, (0, 24))
    return ini <= hora < fin


def track_se_mueve(dq, min_rel=MIN_MOVE_REL):
    if len(dq) < 3:
        return False
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b in dq])
    sizes = np.array([max(b[2] - b[0], b[3] - b[1]) for _, b in dq])
    recorrido = float(np.abs(np.diff(c, axis=0)).sum())
    return recorrido / max(float(np.median(sizes)), 1.0) >= min_rel


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
        out.append(rz[:, :, ::-1])  # BGR -> RGB
    return np.ascontiguousarray(np.stack(out))


class GrabadorSegmentado:
    """Reutiliza los frames que YA se leen para el tracking; no abre una
    segunda conexión a la cámara."""

    def __init__(self, cam_id, out_dir, segundo_s):
        self.cam_id = cam_id
        self.out_dir = out_dir
        self.segundo_s = segundo_s
        self.writer = None
        self.t_inicio = None
        os.makedirs(out_dir, exist_ok=True)

    def escribe(self, frame_bgr, fps, w, h):
        ahora = time.time()
        if self.writer is None or (ahora - self.t_inicio) >= self.segundo_s:
            if self.writer is not None:
                self.writer.release()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.out_dir, f"{self.cam_id}_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, max(fps, 1), (w, h))
            self.t_inicio = ahora
        self.writer.write(frame_bgr)

    def cierra(self):
        if self.writer is not None:
            self.writer.release()


def limpia_viejos(out_dir, retencion_h):
    limite = time.time() - retencion_h * 3600
    try:
        for f in os.listdir(out_dir):
            p = os.path.join(out_dir, f)
            if os.path.isfile(p) and os.path.getmtime(p) < limite:
                os.remove(p)
    except FileNotFoundError:
        pass


def log(cam_id, msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{cam_id}] {msg}"
    print(linea, flush=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, f"{cam_id}.log"), "a") as f:
        f.write(linea + "\n")


def hilo_camara(cam_id, rtsp_url, cola_puntuar, parar_evt):
    """Corre en su propio hilo. Lee, sigue, graba, y ENCOLA candidatos.
    No toca el modelo v5 directamente: eso lo hace hilo_puntuador."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    from ultralytics import YOLO

    log(cam_id, "arrancando (hilo), cargando YOLO propio de seguimiento...")
    det = YOLO(C.YOLO_MODEL)
    grabador = GrabadorSegmentado(cam_id, os.path.join(C.RAW_DIR, cam_id), C.SEGMENTO_S)
    reintentos = 0
    ultima_limpieza = 0.0
    ultimo_check_horario = 0.0

    while not parar_evt.is_set():
        if not dentro_de_horario():
            log(cam_id, "fuera de horario de tienda, en espera...")
            while not parar_evt.is_set() and not dentro_de_horario():
                time.sleep(60)
            if parar_evt.is_set():
                break
            log(cam_id, "horario de tienda abierto, conectando...")

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()   # sin esto, cada intento fallido filtra hilos FFMPEG
            reintentos += 1
            espera = min(5 * reintentos, 60)
            log(cam_id, f"no conecta (intento {reintentos}), reintento en {espera}s")
            time.sleep(espera)
            continue
        reintentos = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 11
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        log(cam_id, f"conectado: {w}x{h} @ {fps:.1f}fps")

        buf = int(C.WINDOW_S * fps / max(C.SKIP_FRAMES, 1)) + 4
        ring = deque(maxlen=buf)
        tracks, last_alarm = {}, {}
        next_t, fi, t_conexion = 0.0, 0, time.time()

        try:
            while not parar_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    log(cam_id, "stream cortado, reconectando...")
                    break

                ahora = time.time()
                if ahora - ultimo_check_horario > 60:
                    ultimo_check_horario = ahora
                    if not dentro_de_horario():
                        log(cam_id, "cerro el horario de tienda, desconectando")
                        break

                if ahora - ultima_limpieza > 3600:
                    limpia_viejos(os.path.join(C.RAW_DIR, cam_id), C.RETENCION_RAW_H)
                    ultima_limpieza = ahora

                grabador.escribe(frame, fps, w, h)

                t = time.time() - t_conexion
                fi += 1
                if C.SKIP_FRAMES > 1 and (fi - 1) % C.SKIP_FRAMES:
                    continue

                res = det.track(frame, classes=[0], conf=0.30, persist=True,
                                verbose=False, imgsz=960, tracker="bytetrack.yaml")[0]
                ring.append((t, frame))
                if res.boxes is not None and res.boxes.id is not None:
                    for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist()):
                        tracks.setdefault(tid, deque(maxlen=buf)).append((t, tuple(b)))

                if t < next_t:
                    continue
                next_t = t + C.STRIDE_S

                for tid, dq in list(tracks.items()):
                    if dq and dq[-1][0] < t - 1.5:
                        del tracks[tid]
                        continue
                    if len(dq) < 2 or (dq[-1][0] - dq[0][0]) < C.MIN_TRACK_S:
                        continue
                    if t - last_alarm.get(tid, -1e9) < C.SCORE_EVERY_S:
                        continue
                    win = [(ft, fr) for (ft, fr) in ring if ft >= t - C.WINDOW_S]
                    wb = [bx for (ft, bx) in dq if ft >= t - C.WINDOW_S]
                    if len(win) < 4 or len(wb) < 2:
                        continue
                    frames_bgr = [f for (_, f) in win]
                    tube = make_tube(frames_bgr, wb, w, h)
                    last_alarm[tid] = t
                    try:
                        cola_puntuar.put_nowait((cam_id, tid, tube))
                    except queue.Full:
                        log(cam_id, "cola de puntuacion llena, se descarta 1 ventana")

        except Exception:
            log(cam_id, "excepcion en el bucle:\n" + traceback.format_exc())
        finally:
            cap.release()
        time.sleep(2)

    grabador.cierra()
    log(cam_id, "hilo detenido")

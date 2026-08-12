#!/usr/bin/env python3
"""Prueba de carga: 10 'camaras' en paralelo usando clips reales ya existentes
en el PC, en bucle, ritmadas al fps real de la fuente (no a la velocidad de
lectura de disco, que seria un test optimista e irrealista).

Reutiliza EXACTAMENTE la logica de produccion (camara_worker.hilo_camara +
modelo.ModeloV5), solo cambia el origen (fichero local en vez de RTSP) y
anade el ritmo de fps para simular una camara real.
"""
import glob
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
from camara_worker import track_se_mueve, make_tube, log
from modelo import ModeloV5

FPS_OBJETIVO = 11.0  # el fps tipico de las camaras de Family Cash


def hilo_camara_archivo(cam_id, path, cola, parar_evt):
    from ultralytics import YOLO
    det = YOLO(C.YOLO_MODEL)
    log(cam_id, f"arrancando prueba de carga con {os.path.basename(path)}")

    while not parar_evt.is_set():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log(cam_id, "no se pudo abrir el fichero de prueba")
            time.sleep(5)
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        buf = int(C.WINDOW_S * FPS_OBJETIVO / max(C.SKIP_FRAMES, 1)) + 4
        ring = deque(maxlen=buf)
        tracks, last_alarm = {}, {}
        next_t, fi, t_inicio = 0.0, 0, time.time()
        periodo = 1.0 / FPS_OBJETIVO

        while not parar_evt.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break  # fin del clip -> reabrir (bucle)

            t = time.time() - t_inicio
            fi += 1
            if C.SKIP_FRAMES > 1 and (fi - 1) % C.SKIP_FRAMES:
                dormir = periodo - (time.time() - t0)
                if dormir > 0:
                    time.sleep(dormir)
                continue

            res = det.track(frame, classes=[0], conf=0.30, persist=True,
                            verbose=False, imgsz=640, tracker="bytetrack.yaml")[0]
            ring.append((t, frame))
            if res.boxes is not None and res.boxes.id is not None:
                for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist()):
                    tracks.setdefault(tid, deque(maxlen=buf)).append((t, tuple(b)))

            if t >= next_t:
                next_t = t + C.STRIDE_S
                for tid, dq in list(tracks.items()):
                    if dq and dq[-1][0] < t - 1.5:
                        del tracks[tid]; continue
                    if len(dq) < 2 or (dq[-1][0] - dq[0][0]) < C.MIN_TRACK_S:
                        continue
                    if t - last_alarm.get(tid, -1e9) < C.COOLDOWN_S:
                        continue
                    if not track_se_mueve(dq):
                        continue
                    win = [(ft, fr) for (ft, fr) in ring if ft >= t - C.WINDOW_S]
                    wb = [bx for (ft, bx) in dq if ft >= t - C.WINDOW_S]
                    if len(win) < 4 or len(wb) < 2:
                        continue
                    frames_bgr = [f for (_, f) in win]
                    tube = make_tube(frames_bgr, wb, w, h)
                    last_alarm[tid] = t
                    try:
                        cola.put_nowait((cam_id, tid, tube))
                    except queue.Full:
                        pass

            dormir = periodo - (time.time() - t0)
            if dormir > 0:
                time.sleep(dormir)
        cap.release()


def hilo_puntuador_prueba(cola, modelo, parar_evt, contador):
    while not parar_evt.is_set():
        lote = []
        try:
            lote.append(cola.get(timeout=1.0))
        except queue.Empty:
            continue
        while len(lote) < 32:
            try:
                lote.append(cola.get_nowait())
            except queue.Empty:
                break
        tubes = [t for (_, _, t) in lote]
        scores = modelo.score(tubes)
        contador[0] += len(tubes)


def monitor(parar_evt, minutos):
    import subprocess
    fin = time.time() + minutos * 60
    print(f"\n{'t(s)':>6} {'GPU%':>6} {'VRAM(MB)':>10} {'CPU%':>7}")
    while time.time() < fin and not parar_evt.is_set():
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True
        ).stdout.strip()
        cpu = subprocess.run(["bash", "-c", "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"],
                             capture_output=True, text=True).stdout.strip()
        gpu_util, vram = gpu.split(",")
        print(f"{minutos*60-(fin-time.time()):6.0f} {gpu_util.strip():>6} {vram.strip():>10} {cpu:>7}")
        time.sleep(10)


def main():
    minutos = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    clips = sorted(glob.glob("/root/CAPTURE_VIDEOS/uploads/*/*.mp4"))
    random.seed(7)
    elegidos = random.sample(clips, min(10, len(clips)))
    print(f"prueba de carga: {len(elegidos)} clips en bucle, {FPS_OBJETIVO}fps, {minutos} min")
    for c in elegidos:
        print(" ", c)

    modelo = ModeloV5(C.CKPT, zoom=C.ZOOM)
    cola = queue.Queue(maxsize=256)
    parar_evt = threading.Event()
    contador = [0]

    hilos = [threading.Thread(target=hilo_puntuador_prueba,
                              args=(cola, modelo, parar_evt, contador), daemon=True)]
    for i, clip in enumerate(elegidos):
        h = threading.Thread(target=hilo_camara_archivo,
                             args=(f"test{i}", clip, cola, parar_evt), daemon=True)
        hilos.append(h)
    for h in hilos:
        h.start()
        time.sleep(0.5)

    monitor(parar_evt, minutos)
    parar_evt.set()
    print(f"\nventanas puntuadas en total: {contador[0]}")
    time.sleep(2)


if __name__ == "__main__":
    main()

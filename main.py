#!/usr/bin/env python3
"""CaptureVCOMv2 — proceso único, un modelo v5 compartido en VRAM.

Un hilo por cámara (lectura RTSP + tracking + grabación cruda), todos
metiendo ventanas candidatas en una cola compartida. Un único hilo
puntuador consume la cola, agrupa lo que haya en cada instante (batch) y
puntúa con el UNICO modelo cargado — así el coste de VRAM no se multiplica
por 10 cámaras.
"""
import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
from camara_worker import hilo_camara, log
from modelo import ModeloV5


def guarda_alerta(cam_id, tube_rgb, score, track_id, alertas_dir):
    os.makedirs(alertas_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"{cam_id}_{ts}_id{track_id}_p{score:.3f}.mp4"
    path = os.path.join(alertas_dir, nombre)
    import cv2
    h, w = tube_rgb.shape[1], tube_rgb.shape[2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for fr in tube_rgb:
        vw.write(fr[:, :, ::-1])
    vw.release()
    return nombre


def hilo_puntuador(cola, modelo, parar_evt):
    alertas_csv = os.path.join(C.OUT_DIR, "alertas.csv")
    nuevo = not os.path.exists(alertas_csv)
    fcsv = open(alertas_csv, "a", newline="")
    wcsv = csv.DictWriter(fcsv, fieldnames=["timestamp", "camara", "track", "score", "clip"])
    if nuevo:
        wcsv.writeheader()
        fcsv.flush()

    log("puntuador", "arrancado, esperando ventanas...")
    n_puntuadas = 0
    ultima_emision = {}   # (cam_id, tid) -> (t_wall, score) para el cooldown de EMISION
    ultimo_alto = {}      # (cam_id, tid) -> ultima vez que el score estuvo activo (para detectar huecos)
    bajo_valle = {}       # (cam_id, tid) -> el score cayo bajo EMIT_VALLE desde la ultima emision (re-arme por valle)
    while not parar_evt.is_set():
        lote = []
        try:
            lote.append(cola.get(timeout=1.0))
        except queue.Empty:
            continue
        # arrastra lo que haya llegado mientras tanto, sin esperar mas
        while len(lote) < 32:
            try:
                lote.append(cola.get_nowait())
            except queue.Empty:
                break

        tubes = [t for (_, _, t) in lote]
        try:
            scores = modelo.score(tubes)
        except Exception as e:
            log("puntuador", f"ERROR puntuando lote de {len(tubes)}: {e}")
            continue

        n_puntuadas += len(tubes)
        for (cam_id, tid, tube), p in zip(lote, scores):
            k = (cam_id, tid)
            # Un clip nuevo solo si hubo un HUECO real antes (REARM_QUIET_S de calma):
            # accion continua (empaquetar / hurto largo) -> 1 clip, no una rafaga de
            # casi-duplicados; dos hurtos separados por un hueco siguen siendo 2 clips.
            if p < C.EMIT_VALLE:
                bajo_valle[k] = True
            if p >= getattr(C, "EMIT_UMBRAL_POR_CAM", {}).get(cam_id, C.EMIT_UMBRAL):
                ahora = time.time()
                gap = ahora - ultimo_alto.get(k, -1e9)
                ultimo_alto[k] = ahora
                pt, ps = ultima_emision.get(k, (-1e9, 0.0))
                if (gap >= C.REARM_QUIET_S or p >= ps + C.EMIT_UPGRADE
                        or bajo_valle.get(k, False) or ahora - pt >= C.EMIT_COOLDOWN_S):
                    clip = guarda_alerta(cam_id, tube, float(p), tid, C.ALERTAS_DIR)
                    wcsv.writerow({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "camara": cam_id, "track": tid,
                        "score": round(float(p), 4), "clip": clip})
                    fcsv.flush()
                    ultima_emision[k] = (ahora, float(p))
                    bajo_valle[k] = False
                    log(cam_id, f"ALERTA track={tid} p={p:.3f} -> {clip}")
        if n_puntuadas % 200 < len(tubes):
            log("puntuador", f"{n_puntuadas} ventanas puntuadas hasta ahora")


def main():
    os.makedirs(C.RAW_DIR, exist_ok=True)
    os.makedirs(C.ALERTAS_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    print(f"CaptureVCOMv2 — {len(C.CAMARAS)} camaras, umbral={C.UMBRAL}", flush=True)

    log("main", "cargando el modelo v5 (una sola vez, compartido)...")
    modelo = ModeloV5(C.CKPT, zoom=C.ZOOM)
    log("main", "modelo cargado.")

    cola = queue.Queue(maxsize=256)
    parar_evt = threading.Event()

    hilos = [threading.Thread(target=hilo_puntuador, args=(cola, modelo, parar_evt),
                              daemon=True, name="puntuador")]
    for cam_id, url in C.CAMARAS:
        h = threading.Thread(target=hilo_camara, args=(cam_id, url, cola, parar_evt),
                             daemon=True, name=cam_id)
        hilos.append(h)

    for h in hilos:
        h.start()
        time.sleep(1)  # escalonar conexiones RTSP

    try:
        while True:
            time.sleep(30)
            vivos = [h.name for h in hilos if h.is_alive()]
            if len(vivos) < len(hilos):
                muertos = [h.name for h in hilos if not h.is_alive()]
                log("main", f"AVISO: hilos caidos: {muertos}")
    except KeyboardInterrupt:
        log("main", "parando...")
        parar_evt.set()
        for h in hilos:
            h.join(timeout=10)


if __name__ == "__main__":
    main()

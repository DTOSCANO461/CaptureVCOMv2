#!/usr/bin/env python3
"""tools/procesa_offline_binario.py -- equivalente de tools/procesa_offline.py
(el script original del commit inicial: inferencia offline + un .mp4 por
alarma) pero sobre el pipeline NUEVO (pose_pipeline.py + inferencia_pose_comun.py,
checkpoint binario NUM_CLASES=2 de training/train_binario.py) en vez del
viejo (camara_worker.make_tube + ModeloV5, deteccion YOLO+ByteTrack sin
pose). No modifica tools/procesa_offline.py ni ningun archivo de
inferencia_pose_* existente -- reusa inferencia_pose_comun.procesa_video/
ModeloPoseGPU y LectorTSCv2/sonda_cv2 de inferencia_pose_test.py tal cual.

Por cada ventana con P(hurto) >= --umbral, guarda un clip .mp4 (el tubo tal
como lo vio el modelo, RECONSTRUIDO invirtiendo la normalizacion ImageNet --
procesa_video() solo expone el tensor ya normalizado via cb_ventana, no hace
falta tocar ese archivo para esto) y una fila en alertas.csv -- mismo
esquema de nombre/columnas que guarda_alerta() del script original.

Uso:
  python3 tools/procesa_offline_binario.py videos/paco_robo1/*.ts
  python3 tools/procesa_offline_binario.py videos/paco_robo1/*.ts --umbral 0.5 --out salida_offline/alertas_binario
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import inferencia_pose_comun as ic
import config as C
from inferencia_pose_test import LectorTSCv2, sonda_cv2

CKPT_BINARIO_DEFAULT = os.path.join(ROOT, "salida_offline", "dataset_5000_binario", "runs", "videomae", "best.pt")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def desnormaliza_tubo(x):
    """x: (3,T,H,W) float normalizado ImageNet, GPU -- inverso EXACTO de
    inferencia_pose_comun.normaliza_tubo() (misma IMAGENET_MEAN_T/STD_T),
    para reconstruir el tubo visual (T,H,W,3) uint8 RGB tal como lo vio el
    modelo, sin tocar ese modulo."""
    arr = (x * ic.IMAGENET_STD_T + ic.IMAGENET_MEAN_T).clamp(0, 1) * 255.0
    return arr.permute(1, 2, 3, 0).round().byte().cpu().numpy()  # T,H,W,3 RGB


def guarda_alerta(cam_id, tubo_rgb, score, track_id, t_s, alertas_dir, fps_clip=4):
    """fps_clip=4: N_FRAMES=16 sobre WINDOW_S=4.0s -- mismo criterio que
    guarda_alerta() del procesa_offline.py original (reproduce el tubo a
    velocidad real)."""
    os.makedirs(alertas_dir, exist_ok=True)
    nombre = f"{cam_id}_t{t_s:07.1f}s_id{track_id}_p{score:.3f}.mp4"
    path = os.path.join(alertas_dir, nombre)
    h, w = tubo_rgb.shape[1], tubo_rgb.shape[2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps_clip, (w, h))
    for fr in tubo_rgb:
        vw.write(fr[:, :, ::-1])  # RGB -> BGR
    vw.release()
    return nombre


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", help="ficheros .ts/.mp4 (uno por camara)")
    ap.add_argument("--pose-model", default=os.path.join(ROOT, "yolo11m-pose.pt"))
    ap.add_argument("--ckpt", default=CKPT_BINARIO_DEFAULT)
    ap.add_argument("--conf-pose", type=float, default=0.30)
    ap.add_argument("--umbral", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "salida_offline", "alertas_binario"))
    ap.add_argument("--max-segundos", type=float, default=45.0,
                     help="corta cada video a los primeros N segundos -- 0 = video entero")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    alertas_dir = os.path.join(args.out, "alertas")
    csv_path = os.path.join(args.out, "alertas.csv")
    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    wcsv = csv.DictWriter(fcsv, fieldnames=["camara", "track", "t_s", "score", "clip"])
    if nuevo:
        wcsv.writeheader()
        fcsv.flush()

    log(f"checkpoint binario: {args.ckpt}  umbral={args.umbral}")
    modelo_score = ic.ModeloPoseGPU(ckpt_path=args.ckpt, num_clases=2)

    n_alertas_total = 0
    for path in args.videos:
        cam_id = os.path.splitext(os.path.basename(path))[0]
        fps, w, h = sonda_cv2(path)
        log(f"[{cam_id}] {w}x{h} @ {fps:.1f}fps")

        modelo_pose = YOLO(args.pose_model)   # tracker propio por camara, mismo motivo que inferencia_pose_test.py
        lector_v = LectorTSCv2([path])
        limite_frames = None if args.max_segundos <= 0 else int(args.max_segundos * fps)
        contador = [0]

        def obtener_frame(lector_v=lector_v, limite_frames=limite_frames, contador=contador):
            if limite_frames is not None and contador[0] >= limite_frames:
                raise StopIteration("limite --max-segundos alcanzado")
            contador[0] += 1
            return lector_v.get_frame()

        n_alertas_video = [0]

        def cb_ventana(tid, t, ventana, tubo_norm, probs, cam_id=cam_id, n_alertas_video=n_alertas_video):
            p_hurto = probs[1].item()
            if p_hurto < args.umbral:
                return
            tubo_rgb = desnormaliza_tubo(tubo_norm)
            clip = guarda_alerta(cam_id, tubo_rgb, p_hurto, tid, t, alertas_dir)
            wcsv.writerow({"camara": cam_id, "track": tid, "t_s": round(t, 2),
                           "score": round(p_hurto, 4), "clip": clip})
            fcsv.flush()
            n_alertas_video[0] += 1
            log(f"[{cam_id}] ALERTA t={t:.2f}s track={tid} p={p_hurto:.3f} -> {clip}")

        t0 = time.time()
        stats = ic.procesa_video(
            obtener_frame, modelo_pose, modelo_score,
            fps=fps, w=w, h=h,
            window_s=C.WINDOW_S, stride_s=C.STRIDE_S,
            min_track_s=C.MIN_TRACK_S, skip_frames=C.SKIP_FRAMES,
            cb_ventana=cb_ventana, conf_pose=args.conf_pose,
        )
        log(f"[{cam_id}] {stats['n_ventanas']} ventanas, {n_alertas_video[0]} alertas, {time.time()-t0:.1f}s")
        n_alertas_total += n_alertas_video[0]

    fcsv.close()
    log(f"listo: {n_alertas_total} alertas -- clips en {alertas_dir}, log en {csv_path}")


if __name__ == "__main__":
    main()

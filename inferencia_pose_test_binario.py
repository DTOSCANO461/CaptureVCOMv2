#!/usr/bin/env python3
"""inferencia_pose_test_binario.py -- variante de inferencia_pose_test.py
para el checkpoint BINARIO (hurto=1/normal=0, training/train_binario.py),
NO el multiclase de 70 acciones. No modifica inferencia_pose_test.py ni
inferencia_pose_comun.py -- los reusa tal cual (LectorTSCv2/sonda_cv2 se
importan directo del primero, procesa_video/ModeloPoseGPU del segundo, que
ya acepta `num_clases` como parametro sin necesitar ningun cambio).

Unica diferencia real con inferencia_pose_test.py: el checkpoint por
defecto, num_clases=2 al construir ModeloPoseGPU, y el callback de ventana
(reporta P(hurto) directo en vez de un top5 de 70 clases -- top5 no tiene
sentido con solo 2 salidas).

Uso:
  python3 inferencia_pose_test_binario.py [--carpeta videos/paco_robo1] [--max-segundos 45]
"""
import argparse
import glob
import os
import time

import torch
from ultralytics import YOLO

import inferencia_pose_comun as ic
import pose_pipeline as pp
import config as C
from inferencia_pose_test import LectorTSCv2, sonda_cv2

CKPT_BINARIO_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "salida_offline", "dataset_5000_binario", "runs", "videomae", "best.pt",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta", default="videos/paco_robo1")
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--ckpt", default=CKPT_BINARIO_DEFAULT)
    ap.add_argument("--conf-pose", type=float, default=0.30)
    ap.add_argument("--umbral", type=float, default=0.5, help="P(hurto) >= umbral -> se marca HURTO en el log")
    ap.add_argument("--max-segundos", type=float, default=45.0,
                     help="corta cada video a los primeros N segundos -- los 4 clips duran "
                          "~201s cada uno; --max-segundos 0 procesa el video entero.")
    args = ap.parse_args()

    archivos = sorted(glob.glob(os.path.join(args.carpeta, "*.ts")))
    if not archivos:
        raise FileNotFoundError(f"no hay .ts en {args.carpeta}")
    print(f"{len(archivos)} archivos .ts en {args.carpeta}: {[os.path.basename(a) for a in archivos]}")
    print(f"checkpoint BINARIO: {args.ckpt}")
    print(f"limite por video: {'sin limite (video entero)' if args.max_segundos <= 0 else f'{args.max_segundos:.0f}s'}\n")

    modelo_score = ic.ModeloPoseGPU(ckpt_path=args.ckpt, num_clases=2)

    resumen_por_video = []
    total_ventanas = total_hurto = 0
    todos_los_track_ids = set()
    t_inicio_total = time.time()

    for path in archivos:
        fps, w, h = sonda_cv2(path)
        nombre = os.path.basename(path)
        print(f"\n--- {nombre} (fps={fps:.2f} {w}x{h}) ---")

        modelo_pose = YOLO(args.pose_model)   # tracker propio por "camara", mismo motivo que inferencia_pose_test.py
        lector_v = LectorTSCv2([path])

        limite_frames = None if args.max_segundos <= 0 else int(args.max_segundos * fps)
        contador_frames = [0]

        def obtener_frame(lector_v=lector_v, limite_frames=limite_frames, contador=contador_frames):
            if limite_frames is not None and contador[0] >= limite_frames:
                raise StopIteration("limite de --max-segundos alcanzado")
            contador[0] += 1
            return lector_v.get_frame()

        n_hurto_video = [0]

        def cb_ventana(tid, t, ventana, tubo_norm, probs, nombre=nombre, n_hurto_video=n_hurto_video):
            p_hurto = probs[1].item()
            marca = "HURTO" if p_hurto >= args.umbral else "normal"
            if marca == "HURTO":
                n_hurto_video[0] += 1
            print(f"  t={t:6.2f}s  track={tid:>3}  ventana={ventana}  P(hurto)={p_hurto:.4f}  -> {marca}")

        t_video_ini = time.time()
        stats = ic.procesa_video(
            obtener_frame, modelo_pose, modelo_score,
            fps=fps, w=w, h=h,
            window_s=C.WINDOW_S, stride_s=C.STRIDE_S,
            min_track_s=C.MIN_TRACK_S, skip_frames=C.SKIP_FRAMES,
            cb_ventana=cb_ventana, conf_pose=args.conf_pose,
        )
        dt_video = time.time() - t_video_ini
        print(f"  -> {stats['n_ventanas']} ventanas puntuadas ({n_hurto_video[0]} marcadas HURTO), "
              f"{len(stats['track_ids'])} track_ids distintos, {dt_video:.1f}s de proceso")

        resumen_por_video.append((nombre, stats["n_ventanas"], n_hurto_video[0], len(stats["track_ids"]), dt_video))
        total_ventanas += stats["n_ventanas"]
        total_hurto += n_hurto_video[0]
        todos_los_track_ids.update((nombre, tid) for tid in stats["track_ids"])

    dt_total = time.time() - t_inicio_total

    print("\n" + "=" * 78)
    print("RESUMEN FINAL (modelo BINARIO)")
    print("=" * 78)
    for nombre, n_vent, n_hurto, n_tracks, dt in resumen_por_video:
        pct = 100 * n_hurto / n_vent if n_vent else 0.0
        print(f"  {nombre}: {n_vent} ventanas, {n_hurto} HURTO ({pct:.1f}%), {n_tracks} track_ids, {dt:.1f}s")
    pct_total = 100 * total_hurto / total_ventanas if total_ventanas else 0.0
    print(f"\nTOTAL: {total_ventanas} ventanas, {total_hurto} marcadas HURTO ({pct_total:.1f}%, umbral={args.umbral}), "
          f"{len(todos_los_track_ids)} track_ids distintos (namespaced por video), {len(archivos)} videos")
    print(f"tiempo total: {dt_total:.1f}s ({dt_total/60:.2f} min)")
    if modelo_score.checkpoint_cargado is None:
        print("\nAVISO: el modelo corrio SOLO con el backbone preentrenado (sin checkpoint fine-tuneado) "
              "-- los P(hurto) de arriba son RUIDO.")


if __name__ == "__main__":
    main()

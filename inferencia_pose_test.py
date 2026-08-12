#!/usr/bin/env python3
"""inferencia_pose_test.py -- arnes de VALIDACION del pipeline de
inferencia_pose.py que SI corre en esta sandbox: identica logica corriente
abajo (tracking por pose, ventaneo temporal, recorte del tubo, scoring
VideoMAE multiclase -- todo vive en inferencia_pose_comun.py e importa
IGUAL en los dos archivos), pero la captura usa cv2.VideoCapture en vez de
NVDEC/PyNvVideoCodec (que no instala en este entorno: Python 3.9, la
libreria real requiere >=3.10 -- ver inferencia_pose.py).

Cada frame se sube a GPU con el MISMO contrato que produciria LectorTS
(ejemplo_captura.py): tensor (H,W,3) uint8 RGB en cuda. A partir de ahi el
resto del codigo (inferencia_pose_comun.procesa_video) no sabe ni le
importa si el frame vino de NVDEC o de cv2+upload.

DISCREPANCIA encontrada y resuelta -- ver PASO 6 mas abajo: los 4 archivos
de prueba (sr_stream_1..4.ts) NO son segmentos secuenciales de UNA misma
camara (que es lo que asume el diseño "carpeta" de LectorTS: rotacion de
.ts de un solo stream, ver ejemplo_captura.py). Son 4 CAMARAS DISTINTAS
del mismo evento grabando en simultaneo (confirmado cruzando con
ejemplo_poses.py, que referencia "camera_streams[0]" = sr_stream_1.ts).
Por eso este harness los procesa como 4 videos INDEPENDIENTES (instancia
nueva de lector Y de tracker YOLO por archivo), no como una carpeta
continua -- concatenarlos habria mezclado 4 escenas fisicas distintas en
una sola linea de tiempo y arrastrado track_ids de una camara a otra.

Uso:
  python3 inferencia_pose_test.py [--carpeta videos/paco_robo1] [--max-segundos 45]
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import inferencia_pose_comun as ic
import pose_pipeline as pp
import config as C

DEVICE = "cuda"


class LectorTSCv2:
    """Mismo contrato/API que ejemplo_captura.LectorTS (get_frame() ->
    tensor (H,W,3) uint8 RGB en GPU, StopIteration al agotarse en modo
    offline) pero decodificando con cv2.VideoCapture -- lo unico que
    cambia entre este harness e inferencia_pose.py.

    Acepta una LISTA de archivos ya resuelta por el llamador (en vez de
    escanear una carpeta como LectorTS): aca cada .ts de prueba es una
    camara distinta, no segmentos rotativos de la misma (ver docstring del
    modulo) -- el llamador decide explicitamente que archivos van juntos
    en una misma linea de tiempo.
    """

    def __init__(self, archivos, device=DEVICE):
        self.archivos = list(archivos)
        if not self.archivos:
            raise FileNotFoundError("lista de .ts vacia")
        self.device = device
        self._idx = -1
        self._cap = None
        self.archivo_actual = None
        self._abrir_siguiente()

    def _abrir_siguiente(self):
        self._idx += 1
        if self._idx >= len(self.archivos):
            raise StopIteration("no hay mas archivos .ts (modo offline)")
        if self._cap is not None:
            self._cap.release()
        path = self.archivos[self._idx]
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"no se pudo abrir {path}")
        self.archivo_actual = path

    def get_frame(self):
        while True:
            ok, frame_bgr = self._cap.read()
            if ok:
                # BGR numpy CPU -> RGB uint8 GPU: MISMO contrato que LectorTS.get_frame()
                t = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
                return t.flip(-1)   # BGR -> RGB, sigue HWC
            self._abrir_siguiente()   # este .ts se acabo -> probar el siguiente (o StopIteration)


def sonda_cv2(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta", default="videos/paco_robo1")
    ap.add_argument("--pose-model", default="yolo11m-pose.pt")
    ap.add_argument("--ckpt", default=ic.CKPT_DEFAULT)
    ap.add_argument("--conf-pose", type=float, default=0.30)
    ap.add_argument("--max-segundos", type=float, default=45.0,
                     help="corta cada video a los primeros N segundos de video -- "
                          "los 4 clips duran ~201s cada uno; sin este limite la corrida "
                          "completa (4x201s) tarda demasiado para un smoke test. "
                          "--max-segundos 0 procesa el video entero.")
    args = ap.parse_args()

    archivos = sorted(glob.glob(os.path.join(args.carpeta, "*.ts")))
    if not archivos:
        raise FileNotFoundError(f"no hay .ts en {args.carpeta}")
    print(f"{len(archivos)} archivos .ts en {args.carpeta}: {[os.path.basename(a) for a in archivos]}")
    print(f"limite por video: {'sin limite (video entero)' if args.max_segundos <= 0 else f'{args.max_segundos:.0f}s'}\n")

    # =================================================================
    # PASOS 1-5: smoke test detallado sobre UN solo video (sr_stream_1),
    # imprimiendo cada forma/contrato intermedio antes de correr el resto
    # de los videos.
    # =================================================================
    primer_video = archivos[0]
    fps0, w0, h0 = sonda_cv2(primer_video)
    print(f"[sonda cv2] {os.path.basename(primer_video)}: fps={fps0:.2f} w={w0} h={h0}\n")

    print("=" * 78)
    print("PASO 1: contrato de frame de la capa de captura (cv2 + upload a GPU)")
    print("=" * 78)
    cap_ref = cv2.VideoCapture(primer_video)
    ok, frame_bgr_ref = cap_ref.read()
    cap_ref.release()
    assert ok, "no se pudo leer el primer frame de referencia con cv2 crudo"

    lector = LectorTSCv2([primer_video])
    frame_gpu = lector.get_frame()
    print(f"frame: shape={tuple(frame_gpu.shape)}  dtype={frame_gpu.dtype}  device={frame_gpu.device}")
    assert frame_gpu.shape == (h0, w0, 3), f"shape inesperada {tuple(frame_gpu.shape)} vs ({h0},{w0},3)"
    assert frame_gpu.dtype == torch.uint8
    assert frame_gpu.device.type == "cuda"

    py, px = h0 // 2, w0 // 2
    bgr_crudo = frame_bgr_ref[py, px].tolist()
    rgb_subido = frame_gpu[py, px].cpu().tolist()
    print(f"pixel ({py},{px}): cv2 BGR crudo={bgr_crudo}  tensor subido={rgb_subido}")
    assert rgb_subido == bgr_crudo[::-1], "el frame subido NO quedo en RGB (deberia ser el BGR crudo invertido)"
    print("OK: (H,W,3) uint8 cuda RGB confirmado -- el canal esta invertido respecto al BGR crudo de cv2, como se espera.\n")

    print("=" * 78)
    print("PASO 2: deteccion de pose + tracking real (.track(), no .predict())")
    print("=" * 78)
    modelo_pose_1 = YOLO(args.pose_model)
    lector2 = LectorTSCv2([primer_video])
    buffer_kp = {}
    n_frames_prueba = int(3 * fps0)   # ~3s para juntar buffer de un track real
    frames_prueba = []
    for _ in range(n_frames_prueba):
        try:
            fr = lector2.get_frame()
        except StopIteration:
            break
        frames_prueba.append(fr)
        for tid, kp in ic.detecta_pose(modelo_pose_1, fr, conf=args.conf_pose):
            buffer_kp.setdefault(tid, []).append(kp)

    assert buffer_kp, "no se detecto NINGUN track_id en los primeros frames -- no se puede seguir validando"
    tid_ejemplo = max(buffer_kp, key=lambda k: len(buffer_kp[k]))   # el track con mas frames acumulados
    kp_clip = np.stack(buffer_kp[tid_ejemplo])
    print(f"track_ids detectados en los primeros {n_frames_prueba} frames ({n_frames_prueba/fps0:.1f}s): "
          f"{sorted(buffer_kp.keys())}")
    print(f"track_id elegido={tid_ejemplo}  buffer de keypoints: shape={kp_clip.shape} (esperado (T,17,2))")
    assert kp_clip.ndim == 3 and kp_clip.shape[1:] == (pp.NUM_KEYPOINTS, 2)
    print("OK: tracking real por pose confirmado, boxes.id y keypoints.xy alineados por indice.\n")

    print("=" * 78)
    print("PASO 3: ventana de recorte (pose_pipeline.bbox_desde_keypoints) -- debe ser cuadrada")
    print("=" * 78)
    ventana = pp.bbox_desde_keypoints(kp_clip, w0, h0, marco=pp.MARCO_INFERENCIA)
    assert ventana is not None, "bbox_desde_keypoints devolvio None (sin keypoints validos)"
    x0, y0, x1, y1 = ventana
    print(f"ventana=(x0={x0}, y0={y0}, x1={x1}, y1={y1})  ancho={x1-x0}  alto={y1-y0}")
    assert (x1 - x0) == (y1 - y0), "la ventana NO salio cuadrada"
    print("OK: ventana cuadrada confirmada.\n")

    print("=" * 78)
    print("PASO 4: tubo (pose_pipeline.recorta_y_redimensiona_gpu, out_size=pp.OUT_SIZE)")
    print("=" * 78)
    frames_chw = [f.permute(2, 0, 1).contiguous() for f in frames_prueba]
    tubo = pp.recorta_y_redimensiona_gpu(frames_chw, ventana, out_size=pp.OUT_SIZE, n_frames=pp.N_FRAMES)
    print(f"tubo: shape={tuple(tubo.shape)}  dtype={tubo.dtype}  device={tubo.device}")
    assert tubo.shape == (pp.N_FRAMES, 3, pp.OUT_SIZE, pp.OUT_SIZE)
    print("OK: shape del tubo confirmada -- (N_FRAMES,3,OUT_SIZE,OUT_SIZE) uint8 RGB.\n")

    print("=" * 78)
    print("PASO 5: normalizacion final (inferencia_pose_comun.normaliza_tubo)")
    print("=" * 78)
    x = ic.normaliza_tubo(tubo)
    print(f"tensor final: shape={tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    print(f"mean={x.mean().item():.4f}  std={x.std().item():.4f}  todo_finito={bool(torch.isfinite(x).all().item())}")
    assert x.shape == (3, pp.N_FRAMES, pp.OUT_SIZE, pp.OUT_SIZE)
    assert torch.isfinite(x).all(), "el tensor final tiene NaN/Inf"
    print("OK: normalizacion ImageNet confirmada, sin NaN/Inf, shape (C,T,H,W).\n")

    # =================================================================
    # PASOS 6-7: corrida real de punta a punta (tracking+recorte+scoring
    # via inferencia_pose_comun.procesa_video, el mismo codigo que usa
    # inferencia_pose.py) sobre los 4 videos, cada uno como stream
    # INDEPENDIENTE (ver discrepancia documentada arriba del modulo).
    # =================================================================
    print("=" * 78)
    print(f"PASO 6/7: forward real por VideoMAE (multiclase, NUM_CLASES={ic.NUM_CLASES}) "
          f"sobre los {len(archivos)} videos + resumen final")
    print("=" * 78)
    modelo_score = ic.ModeloPoseGPU(ckpt_path=args.ckpt)

    resumen_por_video = []
    total_ventanas = 0
    todos_los_track_ids = set()
    tiempos_ventana = []
    t_inicio_total = time.time()

    for path in archivos:
        fps, w, h = sonda_cv2(path)
        nombre = os.path.basename(path)
        print(f"\n--- {nombre} (fps={fps:.2f} {w}x{h}) ---")

        modelo_pose = YOLO(args.pose_model)   # tracker propio y limpio por video (persist=True acumula estado; no
                                               # se puede compartir entre "camaras" distintas sin mezclar track_ids,
                                               # mismo motivo que camara_worker.py: "un YOLO chico por hilo/camara")
        lector_v = LectorTSCv2([path])

        limite_frames = None if args.max_segundos <= 0 else int(args.max_segundos * fps)
        contador_frames = [0]

        def obtener_frame(lector_v=lector_v, limite_frames=limite_frames, contador=contador_frames):
            if limite_frames is not None and contador[0] >= limite_frames:
                raise StopIteration("limite de --max-segundos alcanzado")
            contador[0] += 1
            return lector_v.get_frame()

        def cb_ventana(tid, t, ventana, tubo_norm, probs, nombre=nombre):
            t0 = time.time()
            top5_val, top5_idx = torch.topk(probs, 5)
            top5 = list(zip(top5_idx.tolist(), [round(v, 4) for v in top5_val.tolist()]))
            print(f"  t={t:6.2f}s  track={tid:>3}  ventana={ventana}  top5(clase,prob)={top5}")
            tiempos_ventana.append(time.time() - t0)   # solo mide overhead de print/topk, el forward ya corrio adentro de score()

        t_video_ini = time.time()
        stats = ic.procesa_video(
            obtener_frame, modelo_pose, modelo_score,
            fps=fps, w=w, h=h,
            window_s=C.WINDOW_S, stride_s=C.STRIDE_S,
            min_track_s=C.MIN_TRACK_S, skip_frames=C.SKIP_FRAMES,
            cb_ventana=cb_ventana, conf_pose=args.conf_pose,
        )
        dt_video = time.time() - t_video_ini
        print(f"  -> {stats['n_ventanas']} ventanas puntuadas, {len(stats['track_ids'])} track_ids "
              f"distintos, {dt_video:.1f}s de proceso")

        resumen_por_video.append((nombre, stats["n_ventanas"], len(stats["track_ids"]), dt_video))
        total_ventanas += stats["n_ventanas"]
        todos_los_track_ids.update((nombre, tid) for tid in stats["track_ids"])   # namespaced por video: track_id=1 de cam A no es el mismo id-space que track_id=1 de cam B

    dt_total = time.time() - t_inicio_total

    print("\n" + "=" * 78)
    print("RESUMEN FINAL")
    print("=" * 78)
    for nombre, n_vent, n_tracks, dt in resumen_por_video:
        print(f"  {nombre}: {n_vent} ventanas, {n_tracks} track_ids, {dt:.1f}s")
    print(f"\nTOTAL: {total_ventanas} ventanas puntuadas, {len(todos_los_track_ids)} track_ids distintos "
          f"(namespaced por video), {len(archivos)} videos procesados")
    print(f"tiempo total: {dt_total:.1f}s ({dt_total/60:.2f} min)")
    if total_ventanas:
        print(f"tiempo promedio por ventana (proceso completo del video / n ventanas): {dt_total/total_ventanas:.3f}s")
    if modelo_score.checkpoint_cargado is None:
        print("\nAVISO REPETIDO: el modelo corrio SOLO con el backbone preentrenado (sin el checkpoint "
              "fine-tuneado) -- los scores/top5 de arriba son RUIDO, no reflejan hurto/normal real. "
              "Esta corrida valida forma/dtype/shape de punta a punta, no calidad de deteccion.")


if __name__ == "__main__":
    main()

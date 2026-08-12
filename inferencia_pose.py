#!/usr/bin/env python3
"""inferencia_pose.py -- pipeline de PRODUCCION: captura NVDEC (PyNvVideoCodec,
sin cv2) + tracking por pose + construccion del tubo (pose_pipeline.py) +
scoring con VideoMAE multiclase (inferencia_pose_comun.py).

Este archivo NO se puede ejecutar en la sandbox de desarrollo: PyNvVideoCodec
requiere Python >=3.10 y el entorno de esta sandbox tiene Python 3.9.18 (los
paquetes que "pip install PyNvVideoCodec" encuentra ahi son stubs rotos,
version 0.0.0, metadata inconsistente). Esta escrito para correr en la
maquina real (GPU RTX 4080, driver 580.159.04 confirmado) -- tiene que ser
correcto por lectura/diseño, validado indirectamente por inferencia_pose_test.py
(identica logica corriente abajo, solo cambia la fuente de frames a
cv2.VideoCapture + upload GPU, ver ese archivo para los resultados reales
de la corrida de validacion).

Toda la logica de tracking por pose + ventaneo temporal + construccion del
tubo + normalizacion + scoring vive en inferencia_pose_comun.py e importa
IGUAL aca que en inferencia_pose_test.py -- la UNICA diferencia real entre
los dos archivos es de donde salen los frames: aca, `ejemplo_captura.LectorTS`
(NVDEC); en el otro, cv2.VideoCapture + upload manual con el mismo contrato
de tensor. No se reimplementa LectorTS aca, se importa directo (pedido
explicito: "el archivo es 100% autocontenido... para inferencia_pose.py,
importá LectorTS DIRECTO de ejemplo_captura.py, no la reimplementes").

Deteccion de pose: ic.detecta_pose() baja el frame a CPU numpy BGR ANTES de
.track() (ver el porque en el docstring de inferencia_pose_comun.py -- en
resumen: Ultralytics necesita numpy/PIL para hacer su propio letterbox +
reescalado de keypoints a resolucion nativa; pasarle un tensor GPU crudo
exige un tamano fijo multiplo de 32 y devuelve las coordenadas en ESE
espacio, no en pixeles nativos, lo que obligaria a reescalar a mano -- ver
ejemplo_poses.py de otro proyecto, que sí hace ese camino con un imgsz fijo
de export a TensorRT). Esa es la UNICA cesion real al ideal "cero CPU" de
ejemplo_captura.py: un frame chico, solo en los frames que sobreviven
SKIP_FRAMES, nunca el tubo que ve VideoMAE (ese sale del tensor GPU
original, sin tocar CPU en ningun punto). Si mas adelante se necesita
eliminar tambien esa copia, el camino esta documentado (imgsz fijo +
reescalado manual de keypoints.xy), pero no se aplico aca para no duplicar
la logica de deteccion entre este archivo y el de test (ver restriccion:
"la UNICA diferencia real entre los dos debe ser la fuente de frames").

Uso:
  python3 inferencia_pose.py --carpeta-ts ./rstp_ejemplo/cam0 --gpu-id 0
  python3 inferencia_pose.py --carpeta-ts ./rstp_ejemplo/cam0 --en-vivo
"""
import argparse
import json
import os
import subprocess
import time

from ejemplo_captura import LectorTS   # captura NVDEC -- NO se reimplementa, se importa tal cual (ver docstring)
from ultralytics import YOLO

import inferencia_pose_comun as ic
import config as C

GPU_ID_DEFAULT = 0
POSE_MODEL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11m-pose.pt")


def sonda_video_ffprobe(path_ts):
    """fps y resolucion nativa via ffprobe (mismo binario que ya usa
    ejemplo_captura.grabar_rtsp_a_ts para leer el RTSP -- no agrega una
    dependencia nueva al proyecto). LectorTS/PyNvVideoCodec no expone esta
    metadata en la API que usa ejemplo_captura.py (ni el demuxer ni el
    decoder tienen un getter de fps/resolucion ahi), asi que se lee aparte,
    UNA sola vez al arrancar, del primer .ts que haya en la carpeta."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path_ts)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return fps, int(info["width"]), int(info["height"])


def _primer_ts(carpeta):
    from pathlib import Path
    archivos = sorted(Path(carpeta).glob("*.ts"))
    if not archivos:
        raise FileNotFoundError(f"no hay .ts en {carpeta} -- hace falta al menos uno para sondear fps/resolucion con ffprobe")
    return str(archivos[0])


def procesa_camara_nvdec(carpeta_ts, modelo_pose, modelo_score, gpu_id=GPU_ID_DEFAULT,
                          en_vivo=False, cb_ventana=None):
    """Corre el pipeline completo (captura NVDEC + tracking pose + tubo +
    scoring) sobre UNA carpeta de .ts -- la MISMA semantica de carpeta que
    LectorTS/grabar_rtsp_a_ts: segmentos ROTATIVOS de UNA sola camara, en
    orden cronologico (no varias camaras distintas mezcladas -- si hace
    falta seguir varias camaras a la vez, se llama a esta funcion una vez
    por camara, cada una con su propia LectorTS/YOLO-pose/carpeta, igual
    que camara_worker.py hace un hilo por camara; el modelo de scoring
    (VideoMAE, pesado) SI puede compartirse entre camaras, pasando la MISMA
    instancia de `modelo_score` -- mismo motivo que camara_worker.py:
    ~350MB+contexto CUDA por proceso es demasiado para duplicar por camara).

    fps/w/h se sondean con ffprobe sobre el primer .ts de la carpeta (ver
    sonda_video_ffprobe) porque LectorTS no expone esa metadata.
    """
    fps, w, h = sonda_video_ffprobe(_primer_ts(carpeta_ts))
    print(f"[inferencia_pose] {carpeta_ts}: fps={fps:.2f} w={w} h={h} (via ffprobe)")

    lector = LectorTS(carpeta_ts, gpu_id=gpu_id, en_vivo=en_vivo)

    return ic.procesa_video(
        lector.get_frame, modelo_pose, modelo_score,
        fps=fps, w=w, h=h,
        window_s=C.WINDOW_S, stride_s=C.STRIDE_S,
        min_track_s=C.MIN_TRACK_S, skip_frames=C.SKIP_FRAMES,
        cb_ventana=cb_ventana,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta-ts", required=True, help="carpeta con .ts de UNA camara (rotativos, ver docstring)")
    ap.add_argument("--gpu-id", type=int, default=GPU_ID_DEFAULT)
    ap.add_argument("--pose-model", default=POSE_MODEL_DEFAULT)
    ap.add_argument("--ckpt", default=ic.CKPT_DEFAULT)
    ap.add_argument("--en-vivo", action="store_true",
                     help="si la carpeta sigue recibiendo .ts nuevos (camara en vivo grabando). "
                          "Sin este flag: offline, se corta cuando no quedan mas .ts.")
    args = ap.parse_args()

    modelo_pose = YOLO(args.pose_model)
    modelo_score = ic.ModeloPoseGPU(ckpt_path=args.ckpt)

    def cb_ventana(tid, t, ventana, tubo_norm, probs):
        # Produccion todavia NO tiene un mapeo de las NUM_CLASES=70 salidas a
        # una decision binaria hurto/normal (problema abierto, ver
        # inferencia_pose_comun.ModeloPoseGPU.score()) -- por ahora se loguea
        # el vector completo (top1). Cuando ese mapeo exista, aca es donde se
        # conecta la logica de EMIT_UMBRAL/cooldown equivalente a la de
        # camara_worker.py (fuera del alcance de este archivo).
        prob_top1, clase_top1 = probs.max(0)
        print(f"[score] t={t:.2f}s track={tid} ventana={ventana} "
              f"clase_top1={clase_top1.item()} prob={prob_top1.item():.4f}")

    t0 = time.time()
    stats = procesa_camara_nvdec(args.carpeta_ts, modelo_pose, modelo_score,
                                  gpu_id=args.gpu_id, en_vivo=args.en_vivo,
                                  cb_ventana=cb_ventana)
    dt = time.time() - t0
    print(f"\n{stats['n_ventanas']} ventanas puntuadas, {len(stats['track_ids'])} track_ids, {dt:.1f}s")


if __name__ == "__main__":
    main()

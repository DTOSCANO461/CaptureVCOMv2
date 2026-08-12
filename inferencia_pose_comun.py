#!/usr/bin/env python3
"""inferencia_pose_comun.py

Logica COMPARTIDA entre inferencia_pose.py (captura NVDEC/PyNvVideoCodec,
produccion) e inferencia_pose_test.py (captura cv2.VideoCapture, arnes de
validacion que SI corre en esta sandbox). La UNICA diferencia real entre
esos dos archivos tiene que ser de donde salen los frames -- todo lo que
pasa DESPUES (tracking por pose con .track(), ventaneo temporal estilo
camara_worker.py, construccion del tubo con pose_pipeline.py, normalizacion,
scoring con VideoMAE multiclase) vive ACA, una sola vez, para que los dos
archivos no puedan desincronizarse.

Contrato de frame que este modulo asume en TODAS sus funciones: tensor
torch (H,W,3) uint8 RGB, ya en GPU -- exactamente lo que entrega
`ejemplo_captura.LectorTS.get_frame()` (NVDEC, HWC confirmado leyendo ese
archivo: `torch.from_dlpack(frame_gpu).clone()` sobre el buffer que devuelve
el decoder, sin permute) y lo que arma `inferencia_pose_test.py` a partir
de un frame BGR de cv2 (`torch.from_numpy(frame_bgr).to("cuda").flip(-1)`).
Ninguna funcion de aca de para adelante sabe ni le importa cual de los dos
lo genero.

Por que la deteccion de pose (YOLO-pose .track()) SI baja el frame a CPU:
Ultralytics necesita numpy/PIL para poder hacer su propio letterbox
(preserva aspect ratio + reescala keypoints.xy de vuelta a la resolucion
NATIVA del frame). Probado en la practica contra un frame real de
videos/paco_robo1/sr_stream_1.ts: pasarle un tensor torch directo exige que
YA venga en un tamano multiplo de 32 (LoadTensor no hace letterbox) y
ademas `Results.orig_shape` queda fijado al tamano DEL TENSOR, no al frame
original -- keypoints.xy sale en el espacio reescalado (con aspecto
distorsionado si el resize no preserva ratio), no en pixeles nativos, y
hay que reescalar todo a mano (ver ejemplo_poses.py de otro proyecto, que
sí hace ese camino con un imgsz fijo de export a engine). Para no duplicar
esa complejidad/riesgo en dos archivos que tienen que compartir la misma
logica de tracking, se opta por UNA sola conversion (barata: un frame,
solo en los frames que pasan SKIP_FRAMES) a numpy BGR antes de .track().
El tubo que efectivamente ve VideoMAE NUNCA pasa por esta conversion: se
recorta directo del tensor GPU original (ver `procesa_video`).
"""
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_pipeline as pp                              # bbox_desde_keypoints, recorta_y_redimensiona_gpu, MARCO_INFERENCIA, OUT_SIZE, N_FRAMES
from modelo import _remap_attn_bias, _Wrap, IMAGENET_MEAN, IMAGENET_STD  # mismo checkpoint format/wrapper que gpu_pipeline.ModeloV5GPU
from training.train import NUM_CLASES                   # 70 -- el mismo numero que uso el checkpoint entrenado, NO se duplica el valor aca

DEVICE = "cuda"

# Checkpoint multiclase mas reciente (ver training/train.py, entrenado con
# NUM_CLASES=70). NO es config.CKPT: ese apunta al checkpoint BINARIO viejo
# (videomae_v6_last.pt, cabeza de 2 clases) que usa el pipeline de deteccion
# de camara_worker.py/gpu_pipeline.py -- son dos checkpoints incompatibles
# (distinto tamano de cabeza), no se puede reusar esa constante aca.
CKPT_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "salida_offline", "dataset_500", "runs", "videomae", "best.pt",
)

# Track sin actualizaciones en los ultimos 1.5s se da por perdido -- mismo
# valor hardcodeado que camara_worker.py (no esta en config.py).
TRACK_MUERTO_S = 1.5

IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(3, 1, 1, 1)
IMAGENET_STD_T = torch.tensor(IMAGENET_STD, device=DEVICE).view(3, 1, 1, 1)


# =====================================================================
# Deteccion de pose + tracking (YOLO-pose .track(), NO .predict()): en
# produccion hay VARIAS personas a la vez en camara, hace falta un
# track_id persistente por persona. El truco de
# training/harvest_pose_test.py::keypoints_por_frame_max_confianza
# (predict() sin tracking, se queda con la deteccion de mayor confianza)
# NO aplica aca: ese criterio solo es valido porque cada clip de cosecha
# tiene garantizada una unica persona en escena.
# =====================================================================
def frame_gpu_a_bgr_numpy(frame_gpu):
    """(H,W,3) uint8 RGB en GPU -> (H,W,3) uint8 BGR en CPU numpy, para
    alimentar YOLO-pose (ver docstring del modulo para el porque)."""
    return frame_gpu.flip(-1).contiguous().cpu().numpy()


def detecta_pose(modelo_pose, frame_gpu, conf=0.30, tracker="bytetrack.yaml"):
    """Un frame GPU -> lista de (track_id:int, keypoints (17,2) numpy en
    pixeles nativos). classes=[0] (persona) por paridad con camara_worker.py,
    aunque los pesos *-pose de Ultralytics ya solo detectan persona.

    Confirmado en la practica (yolo11m-pose.pt, frame real de
    videos/paco_robo1/sr_stream_1.ts, agosto 2026): con .track() (persist=
    True) `Results.boxes.id` trae el track_id de ByteTrack y
    `Results.keypoints.xy` los keypoints, EN EL MISMO ORDEN/INDICE que
    `boxes` -- no hace falta cruzar nada por IoU, van alineados 1 a 1.
    Si no hay ninguna deteccion, boxes.id es None (no una lista vacia).
    """
    frame_bgr = frame_gpu_a_bgr_numpy(frame_gpu)
    res = modelo_pose.track(frame_bgr, classes=[0], conf=conf, persist=True,
                             tracker=tracker, verbose=False)[0]
    if res.boxes is None or res.boxes.id is None or res.keypoints is None:
        return []
    ids = res.boxes.id.int().cpu().tolist()
    kps = res.keypoints.xy.cpu().numpy()   # (N,17,2) en pixeles nativos
    return list(zip(ids, kps))


# =====================================================================
# Estado de ventaneo temporal: ring buffer de frames + buffer de
# keypoints por track_id -- misma dinamica que camara_worker.hilo_camara
# (ring/tracks/next_t), con pose en vez de deteccion+bbox como fuente.
# Constantes: WINDOW_S/STRIDE_S/MIN_TRACK_S/SKIP_FRAMES de config.py
# (no se duplican valores: si el caller no pasa nada, se leen de ahi).
# =====================================================================
class EstadoVentaneo:
    def __init__(self, fps, window_s, stride_s, min_track_s, skip_frames):
        self.fps = fps
        self.window_s = window_s
        self.stride_s = stride_s
        self.min_track_s = min_track_s
        self.skip_frames = max(skip_frames, 1)

        # +4 de margen igual que camara_worker.py (redondeos de fps/skip)
        self.buf = int(window_s * fps / self.skip_frames) + 4
        self.ring = deque(maxlen=self.buf)    # (t, frame_gpu HWC RGB uint8)
        self.tracks = {}                      # track_id -> deque(maxlen=buf) de (t, kp (17,2) np)
        self.fi = 0
        self.next_t = 0.0


def normaliza_tubo(tubo):
    """tubo: (T,3,H,W) uint8 RGB en GPU, salida DIRECTA de
    pp.recorta_y_redimensiona_gpu(..., out_size=pp.OUT_SIZE) -- ya en
    224x224, SIN la holgura de STORE_SIZE=256 que usa la cosecha de
    entrenamiento. Por eso, a diferencia de TubeDataset.__getitem__
    (training/train.py), ACA NO hace falta el paso de `_ventanea_np`
    (ese slice 256->224 es augmentation de posicion, solo tiene sentido
    si el tubo se guardo con holgura). Devuelve (3,T,224,224) float32
    normalizado ImageNet, listo para VideoMAEForVideoClassification
    (via _Wrap, que permuta a B,T,C,H,W)."""
    x = tubo.permute(1, 0, 2, 3).float() / 255.0   # T,C,H,W -> C,T,H,W
    x = (x - IMAGENET_MEAN_T) / IMAGENET_STD_T
    return x


def procesa_video(obtener_frame, modelo_pose, modelo_score, fps, w, h,
                   window_s, stride_s, min_track_s, skip_frames,
                   cb_ventana=None, conf_pose=0.30):
    """Bucle principal, IDENTICO para NVDEC y cv2: el UNICO parametro que
    distingue produccion de la prueba es `obtener_frame` (callable sin
    argumentos que devuelve el siguiente frame GPU-tensor (H,W,3) uint8
    RGB, o lanza StopIteration cuando se acaba la fuente).

    Replica la dinamica de camara_worker.hilo_camara (ring buffer +
    buffer de keypoints por track_id + STRIDE_S/MIN_TRACK_S/WINDOW_S)
    pero con pose en vez de deteccion+bbox. NO replica SCORE_EVERY_S/
    last_alarm/EMIT_* (eso es la dinamica de ALARMA sobre un score binario
    p(hurto) -- con un vector de 70 clases sin mapeo a hurto/normal
    definido, esa parte no tiene sentido todavia: ver docstring de
    ModeloPoseGPU.score()). Cada track que cumple MIN_TRACK_S se re-puntua
    cada STRIDE_S, sin throttle adicional.

    `t` es tiempo de VIDEO (fi/fps), no tiempo de pared: en offline
    (LectorTS con en_vivo=False, o cv2 leyendo un .ts ya grabado) el
    decoder corre mas rapido que tiempo real, y WINDOW_S/STRIDE_S son
    conceptos de "segundos de video", no de wall-clock. Para una camara
    EN VIVO esto coincide con wall-clock igual (un frame nuevo llega
    aprox. cada 1/fps segundos reales), asi que la misma formula sirve
    para los dos modos sin ramificar el codigo.

    `cb_ventana(track_id, t, ventana, tubo_normalizado, probs)` se llama
    por cada ventana puntuada, si no es None -- inferencia_pose_test.py lo
    usa para loguear/validar; inferencia_pose.py, para lo que decida hacer
    produccion con el score (todavia no hay logica de alarma definida
    sobre 70 clases, ver arriba).
    """
    st = EstadoVentaneo(fps, window_s, stride_s, min_track_s, skip_frames)
    n_ventanas = 0
    track_ids_vistos = set()

    while True:
        try:
            frame_gpu = obtener_frame()
        except StopIteration:
            break

        st.fi += 1   # 1-based, igual que camara_worker.py
        t = (st.fi - 1) / fps
        if st.skip_frames > 1 and (st.fi - 1) % st.skip_frames:
            continue

        detecciones = detecta_pose(modelo_pose, frame_gpu, conf=conf_pose)
        st.ring.append((t, frame_gpu))
        for tid, kp in detecciones:
            track_ids_vistos.add(tid)
            st.tracks.setdefault(tid, deque(maxlen=st.buf)).append((t, kp))

        if t < st.next_t:
            continue
        st.next_t = t + stride_s

        for tid, dq in list(st.tracks.items()):
            if dq and dq[-1][0] < t - TRACK_MUERTO_S:
                del st.tracks[tid]
                continue
            if len(dq) < 2 or (dq[-1][0] - dq[0][0]) < min_track_s:
                continue

            kp_win = [kp for (ft, kp) in dq if ft >= t - window_s]
            win_frames = [fr for (ft, fr) in st.ring if ft >= t - window_s]
            if len(kp_win) < 2 or len(win_frames) < 4:
                continue

            kp_clip = np.stack(kp_win)   # (T,17,2)
            ventana = pp.bbox_desde_keypoints(kp_clip, w, h, marco=pp.MARCO_INFERENCIA)
            if ventana is None:
                continue

            frames_chw = [f.permute(2, 0, 1).contiguous() for f in win_frames]  # HWC -> CHW, lo que pide recorta_y_redimensiona_gpu
            tubo = pp.recorta_y_redimensiona_gpu(frames_chw, ventana, out_size=pp.OUT_SIZE, n_frames=pp.N_FRAMES)
            x = normaliza_tubo(tubo)
            probs = modelo_score.score([x])[0]   # (NUM_CLASES,) en GPU

            n_ventanas += 1
            if cb_ventana is not None:
                cb_ventana(tid, t, ventana, x, probs)

    return {"n_ventanas": n_ventanas, "track_ids": track_ids_vistos}


# =====================================================================
# Scoring: mismo patron de carga que gpu_pipeline.ModeloV5GPU (mismo
# checkpoint format, mismo _remap_attn_bias/_Wrap de modelo.py) pero
# MULTICLASE -- NUM_CLASES=70 (training/train.py), no 2.
# =====================================================================
class ModeloPoseGPU:
    """score() devuelve el vector COMPLETO de NUM_CLASES probabilidades
    softmax, no un solo p(hurto): todavia no existe un mapeo definido de
    esas 70 clases a una decision binaria hurto/normal (problema abierto
    del proyecto, no se inventa aca -- con el argmax + su probabilidad
    alcanza para un reporte de validacion)."""

    def __init__(self, ckpt_path=CKPT_DEFAULT, device=DEVICE, num_clases=NUM_CLASES):
        self.device = device
        self.num_clases = num_clases
        base = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_clases, ignore_mismatched_sizes=True,
        )
        self.m = _Wrap(base)

        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = _remap_attn_bias(sd)
            self.m.load_state_dict(sd)
            self.checkpoint_cargado = ckpt_path
            print(f"[ModeloPoseGPU] checkpoint fine-tuneado cargado: {ckpt_path}")
        else:
            self.checkpoint_cargado = None
            print(
                f"[ModeloPoseGPU] AVISO: no se encontro checkpoint fine-tuneado "
                f"({ckpt_path!r}) -- cargando SOLO el backbone preentrenado en "
                f"Kinetics, con una cabeza de {num_clases} clases SIN ENTRENAR "
                f"(pesos random). Los scores que salgan de aca NO SON CONFIABLES: "
                f"esto sirve UNICAMENTE para validar que el pipeline corre de "
                f"punta a punta, no para juzgar calidad de deteccion."
            )

        self.m.to(device).eval()

    @torch.no_grad()
    def score(self, tubos_gpu):
        """tubos_gpu: lista de tensores (3,T,224,224) normalizados, en GPU.
        Devuelve (B,NUM_CLASES) softmax en GPU."""
        batch = torch.stack(tubos_gpu).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.m(batch)
        return torch.softmax(out.float(), 1)

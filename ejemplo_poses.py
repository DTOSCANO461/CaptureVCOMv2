"""
ejemplo_poses.py

Ejemplo AUTOCONTENIDO (no importa nada de custom_utils/, bytetrack/, pyskl/
ni de ningun otro .py del repo) para mostrarle a otro algoritmo, paso a
paso, como se arma HOY el "tubo" espacio-temporal de una persona (dado su
track_id) que consume PoseC3D. Todos los valores hardcodeados abajo salen
LITERALMENTE de custom_utils/config_tienda.py (bloque activo "test paco",
que es el ultimo que se ejecuta en ese archivo y por lo tanto el que queda
vigente) y de configs/posec3d/slowonly_r50_ntu60_xview/joint.py -- se
copian los numeros, no se importan los modulos.

Que es fiel al pipeline real y que NO:

  - Carga de video directo a GPU (PyNvVideoCodec, sin pasar por .ts
    intermedio): IDENTICO a m1_ip_capturador.py::_open_video/get_frame.
    Como el objetivo es procesar offline un .ts que ya existe, no se hace
    ninguna conversion previa -- production tampoco la hace, el demuxer de
    PyNvVideoCodec abre el .ts tal cual.
  - Downsample del frame nativo a la resolucion del motor de pose: IDENTICO
    a yolo_pose_hilo en custom_utils/ram_manipulating.py -- es un indice
    nearest-neighbor (np.linspace + index_select), NO un cv2.resize/letterbox.
  - Reescalado de cajas/keypoints de vuelta a resolucion nativa: IDENTICO
    (misma formula ancho_nativo/ancho_modelo, alto_nativo/alto_modelo).
  - Carga del motor YOLO-pose + generacion del .engine si falta: IDENTICO a
    m0_engines_pos.py (mismo imgsz, half=True).
  - Calculo del "tubo" (bbox de recorte) dado un track_id: IDENTICO a
    PoseCompact + Resize(56,56) en pyskl/datasets/pipelines/augmentations.py
    (padding=0.25, hw_ratio=1.0 forzando ventana cuadrada, clamp a los
    bordes del frame, traslado de keypoints, reescalado por eje a 56x56) y
    al render gaussiano de GeneratePoseTarget (sigma=0.6).

  - Tracking (asignacion de IDs entre frames): el repo usa un ByteTrack
    nativo propio (bytetrack/utils/native_bytetrack.py -> BYTETracker de
    bytetrack/ByteTrack/, ~58k lineas). Traerlo entero rompe la condicion
    de "un solo archivo, sin depender de otros .py del repo", asi que ACA
    se usa el tracker ByteTrack que trae Ultralytics de fabrica
    (tracker="bytetrack.yaml"). No es el mismo binario que en produccion,
    pero el objeto de este ejemplo -- como se arma el crop dado un ID -- no
    depende de que tracker asigno ese ID.

Requiere: torch, ultralytics, opencv-python, numpy, PyNvVideoCodec (misma
libreria que ya usa el repo para decodificar en GPU).
"""

import os
import numpy as np
import torch
import cv2
from collections import deque
from ultralytics import YOLO
import PyNvVideoCodec as nvvc


# ============================================================
# CONFIG -- copiada literal desde custom_utils/config_tienda.py,
# bloque "#region test paco" (el que queda activo: es el ultimo que se
# ejecuta en ese archivo) y custom_utils/global_info.py.
# ============================================================

GPU_ID = 0

# camera_streams[0] del bloque "test paco" (config_tienda.py linea ~359)
VIDEO_PATH = "evidencia_utiles/paco_robo1/sr_stream_1.ts"

# h, w del bloque "test paco" (config_tienda.py linea 366) -- resolucion
# NATIVA del video/camara
NATIVE_H, NATIVE_W = 1080, 1920

# rendersize_pose del bloque "test paco" (config_tienda.py linea 367):
# (544-192, 960-192) -- resolucion de ENTRADA del motor TensorRT de pose.
# get_engine_yolopose_imgsz() en global_info.py devuelve esta tupla tal
# cual, y yolo_pose_hilo la desempaqueta como (model_height, model_width).
MODEL_H, MODEL_W = 544 - 192, 960 - 192  # (352, 768)

# pose_level del bloque "test paco" (config_tienda.py linea 365)
POSE_LEVEL = "m"

# get_yolopose_conf('textil') en global_info.py
YOLO_CONF = 0.2

# num_cams=4 en produccion (4 streams del bloque "test paco"), pero este
# ejemplo procesa UN solo video -> el engine se genera/usa con batch=1.
ENGINE_BATCH = 1

PT_PATH = f"./inference_models/yolo11{POSE_LEVEL}-pose.pt"
ENGINE_PATH = f"./inference_models/yolo11{POSE_LEVEL}-pose_{MODEL_H}a{MODEL_W}__b{ENGINE_BATCH}.engine"

# ---- PoseC3D: configs/posec3d/slowonly_r50_ntu60_xview/joint.py::test_pipeline ----
CLIP_LEN = 32              # UniformSampleFrames(clip_len=32)
NUM_KEYPOINTS = 17         # COCO-17 (salida de yolo*-pose)
COMPACT_PADDING = 0.25     # PoseCompact(padding=...) -- default de la clase, usado tal cual
COMPACT_HW_RATIO = 1.0     # PoseCompact(hw_ratio=1.) en el pipeline
TARGET_SIZE = 56           # Resize(scale=(56, 56), keep_ratio=False)
HEATMAP_SIGMA = 0.6        # GeneratePoseTarget(sigma=...) -- default de la clase, usado tal cual

# Track_id al que se le arma el tubo. Si es None, el script usa el primer
# track_id que junte CLIP_LEN muestras (utilizalo para ver el ejemplo sin
# tener que mirar antes el video).
TARGET_TRACK_ID = None

OUT_DIR = "./ejemplo_poses_out"


# ============================================================
# 1) Motor YOLO-pose: cargar .engine, generarlo si no existe.
#    Identico a m0_engines_pos.py::ensure_pose_engine.
# ============================================================

def asegurar_engine():
    if os.path.exists(ENGINE_PATH):
        return

    if not os.path.exists(PT_PATH):
        raise FileNotFoundError(f"No existe el modelo PT: {PT_PATH}")

    print(f"Generando engine desde {PT_PATH} con imgsz=({MODEL_H},{MODEL_W}), batch={ENGINE_BATCH}")

    modelo_pt = YOLO(PT_PATH)
    modelo_pt.export(format="engine", imgsz=(MODEL_H, MODEL_W), half=True, batch=ENGINE_BATCH)

    generado = f"./inference_models/yolo11{POSE_LEVEL}-pose.engine"
    os.rename(generado, ENGINE_PATH)

    print(f"Engine generado: {ENGINE_PATH}")


# ============================================================
# 2) Video directo a GPU. Identico a
#    m1_ip_capturador.py::_open_video / get_frame (sin conversion previa a
#    .ts: el demuxer abre el archivo original tal cual).
# ============================================================

def abrir_video_gpu(path):
    demuxer = nvvc.CreateDemuxer(path)
    decoder = nvvc.CreateDecoder(
        gpuid=GPU_ID,
        codec=demuxer.GetNvCodecId(),
        usedevicememory=True,
        outputColorType=nvvc.OutputColorType.RGB,
    )
    return demuxer, decoder


def iterar_frames_gpu(demuxer, decoder):
    """
    Cada yield es un tensor (H, W, 3) uint8 RGB en GPU -- mismo formato que
    entrega torch.from_dlpack(frame) en get_frame() de m1_ip_capturador.py.
    """
    for packet in demuxer:
        for frame in decoder.Decode(packet):
            yield torch.from_dlpack(frame)


# ============================================================
# 3) Preproceso para el motor: normalizar + downsample nearest-neighbor.
#    Identico a yolo_pose_hilo en custom_utils/ram_manipulating.py:
#        idx1 = np.linspace(0, height-1, model_height).astype(int)
#        idx2 = np.linspace(0, width-1, model_width).astype(int)
#        tensor[:, [0,1,2]][:, :, idx1][:, :, :, idx2]
# ============================================================

def construir_indices_nearest_neighbor():
    idx_h = np.linspace(0, NATIVE_H - 1, MODEL_H).astype(int)
    idx_w = np.linspace(0, NATIVE_W - 1, MODEL_W).astype(int)
    return idx_h, idx_w


def preprocesar_frame(frame_hwc_uint8, idx_h, idx_w):
    # (H,W,3) uint8 -> (1,3,H,W) float16 [0,1], igual que la region
    # "reestructuro y envio el tensor de frames" en m1_ip_capturador.py
    frame_chw = frame_hwc_uint8.permute(2, 0, 1).to(torch.float16).div(255.0).unsqueeze(0)

    # downsample nearest-neighbor a la resolucion del motor -- NO es un resize
    frame_engine = frame_chw[:, :, idx_h][:, :, :, idx_w]

    return frame_engine


def frame_a_bgr_numpy(frame_hwc_uint8):
    """Solo para dibujar/guardar con cv2 -- no es parte del pipeline de inferencia."""
    rgb = frame_hwc_uint8.cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 4) El "tubo": PoseCompact + Resize(56,56) para UN track_id, tal cual
#    pyskl/datasets/pipelines/augmentations.py::PoseCompact + Resize.
# ============================================================

def calcular_tubo_para_id(buffer_keypoints_nativos, frame_h, frame_w):
    """
    buffer_keypoints_nativos: array (T, 17, 2) en pixeles NATIVOS, con
    (0,0) en los frames/joints donde el track no fue visto o el keypoint
    tuvo score bajo (mismo criterio que production: x<=0.01 o y<=0.01 se
    tratan como invalidos).

    Devuelve:
        (x0, y0, x1, y1): ventana de recorte en pixeles nativos (el "tubo").
        keypoints_56: mismos keypoints, trasladados a esa ventana y
                      reescalados a 56x56 (lo que realmente ve el modelo,
                      vuelto heatmap despues).
    """
    kp = buffer_keypoints_nativos
    kp_x = kp[..., 0]
    kp_y = kp[..., 1]

    validos = (kp_x > 0.01) & (kp_y > 0.01)
    if not validos.any():
        return None, None

    # --- PoseCompact: bbox ajustado a TODOS los keypoints validos del clip completo ---
    min_x = kp_x[validos].min()
    max_x = kp_x[validos].max()
    min_y = kp_y[validos].min()
    max_y = kp_y[validos].max()

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    half_w = (max_x - min_x) / 2 * (1 + COMPACT_PADDING)
    half_h = (max_y - min_y) / 2 * (1 + COMPACT_PADDING)

    # --- hw_ratio=1.0: fuerza la ventana a CUADRADA (toma el lado mayor) ---
    half_h = max(COMPACT_HW_RATIO * half_w, half_h)
    half_w = max((1 / COMPACT_HW_RATIO) * half_h, half_w)

    x0 = center_x - half_w
    x1 = center_x + half_w
    y0 = center_y - half_h
    y1 = center_y + half_h

    # --- clamp contra los bordes del frame (allow_imgpad=True en el config,
    #     pero igual se clampea: ver PoseCompact.__call__ real) ---
    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(frame_w, x1))
    y1 = int(min(frame_h, y1))

    ventana_w = max(x1 - x0, 1)
    ventana_h = max(y1 - y0, 1)

    # --- traslada los keypoints al origen de la ventana (NO se recorta imagen) ---
    kp_trasladado = kp.copy()
    kp_trasladado[..., 0] -= x0
    kp_trasladado[..., 1] -= y0

    # --- Resize(scale=(56,56), keep_ratio=False): reescala cada eje por separado ---
    scale_x = TARGET_SIZE / ventana_w
    scale_y = TARGET_SIZE / ventana_h
    keypoints_56 = kp_trasladado.copy()
    keypoints_56[..., 0] *= scale_x
    keypoints_56[..., 1] *= scale_y

    return (x0, y0, x1, y1), keypoints_56


def renderizar_heatmap_56(keypoints_56, validos):
    """
    GeneratePoseTarget(with_kp=True, sigma=0.6): un gaussiano 2D por
    keypoint y por frame sobre un canvas de 56x56. Acumula los 17 canales
    en una sola imagen (suma) solo para poder visualizarlo en un jpg.
    """
    T = keypoints_56.shape[0]
    acumulado = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

    yy, xx = np.mgrid[0:TARGET_SIZE, 0:TARGET_SIZE].astype(np.float32)

    for t in range(T):
        for k in range(NUM_KEYPOINTS):
            if not validos[t, k]:
                continue
            mu_x, mu_y = keypoints_56[t, k, 0], keypoints_56[t, k, 1]
            patch = np.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * HEATMAP_SIGMA ** 2))
            acumulado = np.maximum(acumulado, patch)

    return acumulado


# ============================================================
# 5) Loop principal
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    asegurar_engine()

    print(f"Cargando engine: {ENGINE_PATH}")
    modelo = YOLO(ENGINE_PATH)

    idx_h, idx_w = construir_indices_nearest_neighbor()

    demuxer, decoder = abrir_video_gpu(VIDEO_PATH)

    # buffer circular por track_id: ultimos CLIP_LEN frames, (17,2) en
    # pixeles nativos, (0,0) si el id no aparecio en ese frame -- mismo rol
    # que los "cola_dict_poses" de m2_background.py
    buffers = {}

    objetivo = TARGET_TRACK_ID
    ultimo_frame_bgr = None
    numero_frame = 0

    for frame_gpu in iterar_frames_gpu(demuxer, decoder):
        if frame_gpu.shape[0] != NATIVE_H or frame_gpu.shape[1] != NATIVE_W:
            raise ValueError(
                f"El video decodifico ({frame_gpu.shape[0]},{frame_gpu.shape[1]}) "
                f"pero NATIVE_H/NATIVE_W estan hardcodeados a ({NATIVE_H},{NATIVE_W}). "
                f"Actualiza esas constantes arriba para que coincidan con el video real."
            )

        entrada_engine = preprocesar_frame(frame_gpu, idx_h, idx_w)

        resultados = modelo.track(
            entrada_engine,
            imgsz=(MODEL_H, MODEL_W),
            conf=YOLO_CONF,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        resultado = resultados[0]

        ids_presentes = {}

        if resultado.boxes.id is not None and resultado.keypoints is not None:
            track_ids = resultado.boxes.id.int().cpu().numpy()
            keypoints_xy = resultado.keypoints.xy.cpu().numpy()  # (N,17,2) en espacio (MODEL_W,MODEL_H)

            # --- reescala de espacio del motor a pixeles nativos: misma
            #     formula que sub_yolo_hilo en custom_utils/ram_manipulating.py.
            #     No se toca la confianza ni se fuerza nada a (0,0) aca: los
            #     keypoints no visibles ya llegan como (0,0) desde el propio
            #     modelo YOLO-pose, y es ese valor el que PoseCompact filtra
            #     mas abajo (kp_x>0.01 and kp_y>0.01), igual que en produccion. ---
            keypoints_xy = keypoints_xy.copy()
            keypoints_xy[..., 0] *= NATIVE_W / MODEL_W
            keypoints_xy[..., 1] *= NATIVE_H / MODEL_H

            for i, tid in enumerate(track_ids):
                ids_presentes[int(tid)] = keypoints_xy[i]

        # --- avanza el buffer circular de TODOS los ids que ya existian,
        #     agregando (0,0) para los que no aparecieron este frame ---
        for tid in buffers:
            buffers[tid].append(ids_presentes.get(tid, np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)))

        # --- crea buffer nuevo para ids que aparecen por primera vez ---
        for tid, kp_frame in ids_presentes.items():
            if tid not in buffers:
                buffers[tid] = deque(maxlen=CLIP_LEN)
                buffers[tid].append(kp_frame)

        ultimo_frame_bgr = frame_a_bgr_numpy(frame_gpu)
        numero_frame += 1

        if objetivo is None:
            listos = [tid for tid, buf in buffers.items() if len(buf) == CLIP_LEN]
            if listos:
                objetivo = listos[0]
                print(f"TARGET_TRACK_ID no fijado -> se eligio automaticamente id={objetivo}")

        if objetivo is not None and objetivo in buffers and len(buffers[objetivo]) == CLIP_LEN:
            break

    if objetivo is None or objetivo not in buffers:
        print("No se junto ningun track_id con un clip completo en este video.")
        return

    buffer_kp = np.stack(buffers[objetivo], axis=0)  # (32, 17, 2) pixeles nativos

    print(f"\n--- Tubo para track_id={objetivo}, frame final={numero_frame} ---")

    ventana, keypoints_56 = calcular_tubo_para_id(buffer_kp, NATIVE_H, NATIVE_W)

    if ventana is None:
        print("El track no tuvo ningun keypoint valido en la ventana -> no hay tubo.")
        return

    x0, y0, x1, y1 = ventana
    print(f"ventana (x0,y0,x1,y1) en pixeles nativos: {ventana}")
    print(f"tamano ventana: {x1 - x0} x {y1 - y0}")
    print(f"keypoints_56 shape: {keypoints_56.shape}  (lo que realmente entra al modelo, ya en 56x56)")

    # ---- visualizaciones (esto NO es parte del pipeline real, solo para mirar) ----
    frame_anotado = ultimo_frame_bgr.copy()
    cv2.rectangle(frame_anotado, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(frame_anotado, f"id={objetivo}", (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "frame_con_ventana.jpg"), frame_anotado)

    crop_visual = ultimo_frame_bgr[y0:y1, x0:x1]
    if crop_visual.size > 0:
        cv2.imwrite(os.path.join(OUT_DIR, "crop_visual.jpg"), crop_visual)

    validos_56 = (buffer_kp[..., 0] > 0.01) & (buffer_kp[..., 1] > 0.01)
    heatmap = renderizar_heatmap_56(keypoints_56, validos_56)
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    heatmap_vis = cv2.resize(heatmap_vis, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, "heatmap_56_ampliado.jpg"), heatmap_vis)

    print(f"\nGuardado en {OUT_DIR}/: frame_con_ventana.jpg, crop_visual.jpg, heatmap_56_ampliado.jpg")


if __name__ == "__main__":
    main()

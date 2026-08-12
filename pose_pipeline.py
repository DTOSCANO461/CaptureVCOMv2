"""pose_pipeline.py

Construccion de "tubos" a partir de tracking por POSE (YOLO-pose + geometria
de recorte estilo PoseCompact), en vez de deteccion+bbox como hacen hoy
camara_worker.py / gpu_pipeline.py. Es la nueva referencia -- pensada para
usarse TANTO en inferencia COMO al generar el dataset de entrenamiento, con
la misma funcion de recorte en los dos casos.

Se basa en ejemplo_poses.py (que documenta el pipeline de otro repo/proyecto
usando PoseC3D), pero con las diferencias que pidio el usuario:
  - n_frames=16 (VideoMAE), no 32 (PoseC3D).
  - No genera heatmap de keypoints a 56x56: el tubo son los PIXELES RGB del
    recorte (lo que consume VideoMAE), no una representacion de esqueleto.
  - Decodifica con cv2.VideoCapture (PyNvVideoCodec, usado en ejemplo_poses.py,
    no esta disponible en este entorno) -- el resto del pipeline (deteccion de
    pose, calculo de la ventana, resize) es igual de GPU-nativo.
  - UN SOLO resize, con torch (`F.interpolate`, modo "nearest" -- el mas
    barato: sin pesos que promediar, solo indexado).

Filosofia del recorte (ver bbox_desde_keypoints / recorta_y_redimensiona_gpu /
ventanea): el "marco" (padding) alrededor del bbox de la persona queda FIJO
al construir cada tubo -- es lo unico que congela el zoom, tanto en inferencia
como al generar el dataset (una vez guardado un .npy, ese zoom no se puede
recuperar sin volver al video crudo). La data augmentation de POSICION
(offset) se resuelve aparte, con un slice puro (sin resize) sobre un tubo
guardado a STORE_SIZE=256 en vez de OUT_SIZE=224 -- 32px de holgura, numeros
enteros limpios. En inferencia no hace falta holgura: se resuelve directo a
224 en el unico resize.
"""
import numpy as np
import torch
import torch.nn.functional as F

N_FRAMES = 16                  # VideoMAE (el ejemplo de referencia usa 32, para PoseC3D)
NUM_KEYPOINTS = 17             # COCO-17, salida de yolo*-pose
PADDING = 0.5                  # margen proporcional sobre el bbox de keypoints. Calibrado
                                # contra videos/paco_robo1/sr_stream_1.ts para dar un lado de
                                # recorte parecido al metodo viejo (deteccion, mediana*1.9):
                                # con padding=0.5 salio ratio 0.96 (ver registro de calibracion
                                # en el historial de cambios). Ajustable, es una sola constante.
HW_RATIO = 1.0                 # fuerza ventana CUADRADA (se usa el lado mayor de los dos)
OUT_SIZE = 224                 # resolucion de entrada de VideoMAE
STORE_SIZE = 256               # tamano al que se guardan los tubos de ENTRENAMIENTO:
                                # 224 + 32 de holgura -> el offset de la ventana (Paso de
                                # augmentation) desliza en [0,32] en cada eje, sin resize extra.


def bbox_desde_keypoints(kp_clip, frame_w, frame_h, padding=PADDING, hw_ratio=HW_RATIO):
    """kp_clip: (T,17,2) en pixeles nativos, (0,0) donde el keypoint no es valido
    (mismo criterio que el motor YOLO-pose / ejemplo_poses.py: x<=0.01 o y<=0.01
    se tratan como invalidos).

    Igual que PoseCompact: bbox ajustado a TODOS los keypoints validos del
    clip completo (no por frame), con padding proporcional, forzado a
    cuadrado (hw_ratio=1.0) y clampeado a los bordes del frame.

    Devuelve (x0,y0,x1,y1) enteros, o None si no hubo ningun keypoint valido
    en todo el clip.
    """
    kp_x, kp_y = kp_clip[..., 0], kp_clip[..., 1]
    validos = (kp_x > 0.01) & (kp_y > 0.01)
    if not validos.any():
        return None

    min_x, max_x = float(kp_x[validos].min()), float(kp_x[validos].max())
    min_y, max_y = float(kp_y[validos].min()), float(kp_y[validos].max())

    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_w = (max_x - min_x) / 2 * (1 + padding)
    half_h = (max_y - min_y) / 2 * (1 + padding)

    half_h = max(hw_ratio * half_w, half_h)
    half_w = max((1 / hw_ratio) * half_h, half_w)

    x0, x1 = cx - half_w, cx + half_w
    y0, y1 = cy - half_h, cy + half_h

    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(frame_w, x1)), int(min(frame_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def recorta_y_redimensiona_gpu(frames_gpu, ventana, out_size, n_frames=N_FRAMES):
    """frames_gpu: lista de tensores (3,H,W) uint8 RGB en GPU -- la ventana
    temporal cruda de la que se muestrean n_frames (igual criterio que
    gp.make_tube_gpu: linspace uniforme sobre los frames disponibles).
    ventana: (x0,y0,x1,y1) en pixeles nativos, de bbox_desde_keypoints.

    UN SOLO resize (nearest) del recorte crudo -> (out_size,out_size).
    out_size=STORE_SIZE (256) para generar tubos de entrenamiento;
    out_size=OUT_SIZE (224) para inferencia, directo a la red -- misma
    funcion, un solo parametro distinto.
    """
    x0, y0, x1, y1 = ventana
    idx = torch.linspace(0, len(frames_gpu) - 1, n_frames).round().long()

    salida = []
    for i in idx.tolist():
        recorte = frames_gpu[i][:, y0:y1, x0:x1].float().unsqueeze(0)   # 1,3,h,w
        redim = F.interpolate(recorte, size=(out_size, out_size), mode="nearest")
        salida.append(redim.squeeze(0).clamp(0, 255).to(torch.uint8))
    return torch.stack(salida)     # (n_frames,3,out_size,out_size) uint8 RGB


def ventanea(tubo_store, out_size=OUT_SIZE, offset=None):
    """tubo_store: (T,3,STORE_SIZE,STORE_SIZE) uint8 en GPU (ya generado con
    recorta_y_redimensiona_gpu(out_size=STORE_SIZE)).

    Data augmentation de POSICION: slice puro, sin resize.
    offset=None  -> sortea uno aleatorio en [0,holgura] por eje (entrenamiento).
    offset=(oy,ox) -> ventana fija (eval/inferencia: holgura//2 = centrado).
    """
    store_size = tubo_store.shape[-1]
    holgura = store_size - out_size
    if offset is None:
        oy = int(torch.randint(0, holgura + 1, (1,)).item()) if holgura > 0 else 0
        ox = int(torch.randint(0, holgura + 1, (1,)).item()) if holgura > 0 else 0
    else:
        oy, ox = offset
    return tubo_store[:, :, oy:oy + out_size, ox:ox + out_size]


def offset_centrado(store_size=STORE_SIZE, out_size=OUT_SIZE):
    """Offset determinista (centrado) para eval/inferencia -- mismo criterio
    que un center-crop clasico."""
    h = (store_size - out_size) // 2
    return (h, h)

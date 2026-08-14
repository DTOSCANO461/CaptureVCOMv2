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
  - UN SOLO resize, con torch (`F.interpolate`, modo "bilinear" -- promedia
    pixeles vecinos, mas fiel que "nearest" aunque un poco mas caro).

Filosofia del recorte (ver bbox_desde_keypoints / recorta_y_redimensiona_gpu /
ventanea): el "marco" (multiplicador sobre el bbox de keypoints) queda FIJO
al construir cada tubo -- es lo unico que congela el zoom, tanto en inferencia
(MARCO_INFERENCIA=1.05, siempre igual) como al generar el dataset (ahi el
llamador sortea un marco entre MARCO_MIN_TRAIN=1.0 y MARCO_MAX_TRAIN=1.1 por
tubo -- la augmentation de zoom in/out -- y una vez guardado el .npy ese zoom
ya no se puede recuperar sin volver al video crudo). La data augmentation de
POSICION (offset) se resuelve aparte, con un slice puro (sin resize) sobre un
tubo guardado a STORE_SIZE=256 en vez de OUT_SIZE=224 -- 32px de holgura,
numeros enteros limpios. En inferencia no hace falta holgura: se resuelve
directo a 224 en el unico resize.
"""
import numpy as np
import torch
import torch.nn.functional as F

N_FRAMES = 16                  # VideoMAE (el ejemplo de referencia usa 32, para PoseC3D)
NUM_KEYPOINTS = 17             # COCO-17, salida de yolo*-pose

# MARCO: multiplicador DIRECTO sobre el lado mayor del bbox tight de keypoints
# (side = lado_mayor_del_esqueleto * marco). marco=1.0 -> ventana pegada al
# esqueleto, sin margen. marco=1.05 -> 5% mas grande que el esqueleto.
MARCO_INFERENCIA = 1.05        # fijo, siempre el mismo recorte en produccion
MARCO_MIN_TRAIN = 1.0          # rango de sorteo al generar el dataset (zoom in/out,
MARCO_MAX_TRAIN = 1.1          # se congela por tubo al guardar el .npy -- ver docstring del modulo)

OUT_SIZE = 224                 # resolucion de entrada de VideoMAE
STORE_SIZE = 256               # tamano al que se guardan los tubos de ENTRENAMIENTO:
                                # 224 + 32 de holgura -> el offset de la ventana (Paso de
                                # augmentation) desliza en [0,32] en cada eje, sin resize extra.


def bbox_desde_keypoints(kp_clip, frame_w, frame_h, marco=MARCO_INFERENCIA):
    """kp_clip: (T,17,2) en pixeles nativos, (0,0) donde el keypoint no es valido
    (mismo criterio que el motor YOLO-pose / ejemplo_poses.py: x<=0.01 o y<=0.01
    se tratan como invalidos).

    bbox TIGHT ajustado a TODOS los keypoints validos del clip completo (no
    por frame): de mano a mano, de cabeza a pies visibles -- sin margen
    todavia. `marco` es el multiplicador DIRECTO sobre el lado mayor de ese
    bbox (side = lado_mayor * marco; marco=1.0 = pegado al esqueleto,
    marco=1.05 = 5% mas grande) -- ventana SIEMPRE cuadrada, el chico de
    VideoMAE necesita 224x224, no un rectangulo.

    En inferencia `marco` es fijo (MARCO_INFERENCIA). Al generar el dataset
    de entrenamiento, el llamador sortea `marco` entre MARCO_MIN_TRAIN y
    MARCO_MAX_TRAIN por tubo (augmentation de zoom in/out) y lo pasa aca --
    esta funcion no sortea nada, solo aplica el valor que le dan.

    Garantiza devolver una ventana de exactamente `side x side` (side =
    round(lado_mayor*marco), clampado a min(frame_w,frame_h) si hiciera
    falta): en vez de centrar y despues recortar cada borde contra el frame
    por separado (eso rompe la simetria y puede achicar el lado cerca de un
    borde), primero se fija el tamano y LUEGO se desliza la posicion para
    que quepa entera -- mismo criterio que camara_worker.make_tube.

    Devuelve (x0,y0,x1,y1) enteros con x1-x0 == y1-y0 siempre, o None si no
    hubo ningun keypoint valido en todo el clip.
    """
    kp_x, kp_y = kp_clip[..., 0], kp_clip[..., 1]
    validos = (kp_x > 0.01) & (kp_y > 0.01)
    if not validos.any():
        return None

    min_x, max_x = float(kp_x[validos].min()), float(kp_x[validos].max())
    min_y, max_y = float(kp_y[validos].min()), float(kp_y[validos].max())
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    lado_mayor = max(max_x - min_x, max_y - min_y)
    side = lado_mayor * marco
    side = min(side, min(frame_w, frame_h))    # nunca mas grande que el frame -> el deslizamiento de abajo siempre alcanza
    side_i = int(round(side))
    if side_i <= 0:
        return None

    x0 = int(np.clip(cx - side / 2, 0, frame_w - side_i))
    y0 = int(np.clip(cy - side / 2, 0, frame_h - side_i))
    return x0, y0, x0 + side_i, y0 + side_i


def recorta_y_redimensiona_gpu(frames_gpu, ventana, out_size, n_frames=N_FRAMES):
    """frames_gpu: lista de tensores (3,H,W) uint8 RGB en GPU -- la ventana
    temporal cruda de la que se muestrean n_frames (igual criterio que
    gp.make_tube_gpu: linspace uniforme sobre los frames disponibles).
    ventana: (x0,y0,x1,y1) en pixeles nativos, de bbox_desde_keypoints.

    UN SOLO resize (bilinear) del recorte crudo -> (out_size,out_size).
    out_size=STORE_SIZE (256) para generar tubos de entrenamiento;
    out_size=OUT_SIZE (224) para inferencia, directo a la red -- misma
    funcion, un solo parametro distinto.

    bilinear en vez de nearest (pedido explicito): promedia los pixeles
    vecinos en vez de solo indexar el mas cercano -- bordes/texturas quedan
    suavizados en vez de "escalonados", mas parecido a como VideoMAE fue
    preentrenado/evaluado rio arriba (resize bilineal es el estandar en casi
    todo pipeline de vision, nearest era la opcion mas barata en compute, no
    la de mejor calidad). align_corners=False es el default recomendado por
    PyTorch para bilinear (evita el sesgo de medio pixel en los bordes).
    """
    x0, y0, x1, y1 = ventana
    idx = torch.linspace(0, len(frames_gpu) - 1, n_frames).round().long()

    salida = []
    for i in idx.tolist():
        recorte = frames_gpu[i][:, y0:y1, x0:x1].float().unsqueeze(0)   # 1,3,h,w
        redim = F.interpolate(recorte, size=(out_size, out_size), mode="bilinear", align_corners=False)
        salida.append(redim.squeeze(0).clamp(0, 255).to(torch.uint8))
    return torch.stack(salida)     # (n_frames,3,out_size,out_size) uint8 RGB


def ventanea(tubo_store, out_size=OUT_SIZE, offset=None):
    """tubo_store: (T,3,H,W) uint8 en GPU (ya generado con
    recorta_y_redimensiona_gpu, en teoria siempre H==W==STORE_SIZE porque
    bbox_desde_keypoints garantiza una ventana cuadrada -- pero esto no
    ASUME esa garantia: si H y/o W llegan mas chicos que out_size (H!=W
    incluido), primero se extiende el borde opuesto (replicate, sin
    reescalar nada) hasta alcanzar out_size en cada eje, y recien ahi se
    desliza. Nunca falla por tubo chico, nunca hace un segundo resize.

    Data augmentation de POSICION: slice puro, sin resize.
    offset=None  -> sortea uno aleatorio en [0,holgura] por eje (entrenamiento).
    offset=(oy,ox) -> ventana fija (eval/inferencia: holgura//2 = centrado).
    """
    _, _, H, W = tubo_store.shape
    falta_h, falta_w = max(0, out_size - H), max(0, out_size - W)
    if falta_h or falta_w:
        # F.pad opera sobre las 2 ultimas dims: (izq,der,arriba,abajo).
        # Todo el faltante se agrega del lado derecho/abajo -- "ampliar el
        # borde opuesto" en vez de perder resolucion o volver a resize.
        tubo_store = F.pad(tubo_store.float(), (0, falta_w, 0, falta_h),
                            mode="replicate").to(tubo_store.dtype)
        H, W = tubo_store.shape[-2:]

    holgura_h, holgura_w = H - out_size, W - out_size
    if offset is None:
        oy = int(torch.randint(0, holgura_h + 1, (1,)).item()) if holgura_h > 0 else 0
        ox = int(torch.randint(0, holgura_w + 1, (1,)).item()) if holgura_w > 0 else 0
    else:
        oy, ox = offset
        oy, ox = min(oy, holgura_h), min(ox, holgura_w)   # por si el offset centrado se calculo contra STORE_SIZE nominal
    return tubo_store[:, :, oy:oy + out_size, ox:ox + out_size]


def offset_centrado(store_size=STORE_SIZE, out_size=OUT_SIZE):
    """Offset determinista (centrado) para eval/inferencia -- mismo criterio
    que un center-crop clasico."""
    h = (store_size - out_size) // 2
    return (h, h)

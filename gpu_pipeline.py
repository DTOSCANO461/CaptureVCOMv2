#!/usr/bin/env python3
"""Version GPU-nativa del pipeline de tubos.

`camara_worker.make_tube` + `modelo.ModeloV5._prep_uno` hacen todo el recorte,
muestreo, zoom y normalizacion con numpy/cv2 en CPU, y solo al final suben un
tensor ya listo a la GPU (`modelo.ModeloV5.score`, linea `.to(self.device)`).

Aqui se invierte eso: el frame de cv2 (numpy BGR, HWC, en CPU porque asi lo
entrega el decodificador de video) se sube a la GPU como tensor de torch
INMEDIATAMENTE, y todo lo demas -- recorte centrado en la persona, muestreo
de 16 frames, zoom 0.6, margen 6.25%, resize a 224x224, normalizacion
ImageNet -- se hace con operaciones explicitas de torch sobre ese tensor,
sin volver a pasar por numpy/CPU en ningun punto intermedio.

Las tres funciones replican EXACTAMENTE la aritmetica de camara_worker.py y
modelo.py (mismos parametros, mismo orden de operaciones); solo cambia el
motor (torch/GPU en vez de numpy+cv2/CPU).
"""
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def frame_a_gpu(frame_bgr, device="cuda"):
    """cv2 frame (H,W,3) BGR uint8 numpy -> tensor (3,H,W) RGB uint8 en GPU.

    Se sube como uint8 (no float) para no multiplicar por 4 la memoria del
    ring buffer en VRAM; la conversion a float solo ocurre en los 16 frames
    que de verdad se usan para un tubo (make_tube_gpu), no en cada frame leido.
    """
    t = torch.from_numpy(frame_bgr).to(device, non_blocking=True)  # H,W,3 uint8 BGR
    t = t.flip(-1)                                                  # BGR -> RGB
    t = t.permute(2, 0, 1).contiguous()                             # C,H,W uint8
    return t


def make_tube_gpu(frames_gpu, boxes, w, h, n_frames=16, side_out=256, device="cuda"):
    """Equivalente GPU de camara_worker.make_tube.

    frames_gpu: lista de tensores (3,H,W) uint8 en GPU (la ventana temporal
                de frames crudos, ya en orden).
    boxes:      lista de (x1,y1,x2,y2) del track en esa misma ventana.
    Devuelve:   tensor (n_frames,3,side_out,side_out) float32 en GPU, RGB,
                valores en [0,1].
    """
    idx = torch.linspace(0, len(frames_gpu) - 1, n_frames).round().long()

    # torch.median() NO es np.median(): con conteo par devuelve el menor de
    # los dos centrales en vez de promediarlos (desplaza el recorte). Se usa
    # torch.quantile(..., 0.5), que si interpola igual que numpy.
    bs = torch.tensor(boxes, dtype=torch.float32, device=device)       # (N,4)
    cx = torch.quantile((bs[:, 0] + bs[:, 2]) / 2, 0.5)
    cy = torch.quantile((bs[:, 1] + bs[:, 3]) / 2, 0.5)
    lado_mayor = torch.maximum(bs[:, 2] - bs[:, 0], bs[:, 3] - bs[:, 1])
    side = torch.quantile(lado_mayor, 0.5) * 1.9
    side = torch.clamp(side, min=200.0)                                # nunca mas pequeño que 200px
    side = torch.clamp(side, max=float(min(w, h)))                     # nunca mayor que el frame

    x1 = int(torch.clamp(cx - side / 2, 0, w - side).item())
    y1 = int(torch.clamp(cy - side / 2, 0, h - side).item())
    s = int(side.item())

    # Este resize casi siempre reduce mucho (side tipico 400-1080 -> 256), asi que el
    # filtro importa: probado contra cv2.INTER_AREA (la referencia de camara_worker.py)
    # sobre casos reales, bicubic+antialias es el que mas se acerca (diff de score
    # ~0.006 en el caso mas sensible probado, frente a ~0.19 con bilinear+antialias o
    # ~0.04 con area) - ver la seccion "que tan sensible es el resize" del notebook.
    salida = []
    for i in idx.tolist():
        recorte = frames_gpu[i][:, y1:y1 + s, x1:x1 + s]                # C,s,s uint8, en GPU
        recorte = recorte.float().unsqueeze(0) / 255.0                  # 1,C,s,s float32 [0,1]
        redim = F.interpolate(recorte, size=(side_out, side_out),
                               mode="bicubic", align_corners=False, antialias=True)
        redim = redim.clamp(0.0, 1.0)     # bicubic puede overshoot fuera de [0,1] (ringing)
        salida.append(redim.squeeze(0))
    tubo = torch.stack(salida)                                          # T,C,H,W en GPU
    return tubo


def preprocesa_tubo_gpu(tubo, zoom=0.6, device="cuda"):
    """Equivalente GPU de modelo.ModeloV5._prep_uno.

    tubo: (T,C,H,W) float32 [0,1] RGB en GPU (salida de make_tube_gpu).
    Devuelve: (C,T,224,224) normalizado ImageNet, en GPU, listo para VideoMAE.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1, 1)

    x = tubo.permute(1, 0, 2, 3)                                        # C,T,H,W

    # 1) recorte centrado por zoom (sin jitter: eso es solo de entrenamiento)
    if zoom < 1.0:
        side = int(round(x.shape[-1] * zoom))
        o = (x.shape[-1] - side) // 2
        x = x[:, :, o:o + side, o:o + side]

    # 2) margen fijo 6.25% (equivalente al recorte de 16px sobre 256 del dataset)
    L = x.shape[-1]
    m = int(L * 0.0625)
    x = x[:, :, m:L - m, m:L - m]

    # 3) resize bilinear a 224x224 si hace falta
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    # 4) normalizacion ImageNet
    x = (x - mean) / std
    return x


class ModeloV5GPU:
    """Como modelo.ModeloV5, pero score() espera tubos que YA llegaron
    preprocesados en GPU (via make_tube_gpu + preprocesa_tubo_gpu): no
    recibe numpy, no toca la CPU en ningun punto del forward."""

    def __init__(self, ckpt_path, device="cuda"):
        from transformers import VideoMAEForVideoClassification
        from modelo import _remap_attn_bias, _Wrap  # mismo checkpoint, mismo wrapper

        self.device = device
        base = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=2, ignore_mismatched_sizes=True,
        )
        self.m = _Wrap(base)
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _remap_attn_bias(sd)
        self.m.load_state_dict(sd)
        self.m.to(device).eval()

    @torch.no_grad()
    def score(self, tubos_gpu):
        """tubos_gpu: lista de tensores (C,T,224,224) normalizados, en GPU.
        Devuelve un tensor (B,) en GPU con p(hurto) -- el llamador decide si
        y cuando lo baja a CPU/numpy (p.ej. solo para el track que supera el
        umbral, no para todos)."""
        batch = torch.stack(tubos_gpu).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.m(batch)
        return torch.softmax(out.float(), 1)[:, 1]

"""Envoltorio de inferencia para el checkpoint v5 (VideoMAE-Base).

El preprocesado replica EXACTAMENTE la rama eval (self.train=False) de
TubeDataset.__getitem__ en train/train.py:
  1. recorte centrado por zoom (sin jitter, eso es solo de train)
  2. margen fijo 6.25% (equivalente al recorte de 16px sobre 256 del dataset)
  3. tensor (C,T,H,W), resize bilinear a 224x224 si hace falta
  4. normalizacion ImageNet
Un desajuste aqui puntua basura sin dar ningun error.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _remap_attn_bias(sd):
    """Checkpoints antiguos guardan bias separado en query/key/value; las
    versiones actuales de VideoMAE en transformers usan query/key/value sin
    bias (bias=False) + q_bias/v_bias sueltos (key.bias se fija a cero
    porque no afecta al softmax: se suma igual a todas las keys de una
    query, así que no cambia la distribucion de atencion). Sin este remap,
    load_state_dict falla por claves faltantes/sobrantes."""
    out = dict(sd)
    for k in list(out.keys()):
        if k.endswith(".attention.attention.query.bias"):
            out[k[: -len("query.bias")] + "q_bias"] = out.pop(k)
        elif k.endswith(".attention.attention.value.bias"):
            out[k[: -len("value.bias")] + "v_bias"] = out.pop(k)
        elif k.endswith(".attention.attention.key.bias"):
            out.pop(k)
    return out


class _Wrap(nn.Module):
    """Igual que build_model() en train.py: HF espera (B,T,C,H,W)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):  # x: B,C,T,H,W
        return self.m(pixel_values=x.permute(0, 2, 1, 3, 4)).logits


class ModeloV5:
    def __init__(self, ckpt_path, zoom=0.6, device="cuda"):
        self.device = device
        self.zoom = zoom
        base = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=2, ignore_mismatched_sizes=True,
        )
        self.m = _Wrap(base)
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _remap_attn_bias(sd)
        self.m.load_state_dict(sd)
        self.m.to(device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1, 1)

    def _prep_uno(self, arr):
        """arr: (16,H,W,3) uint8 RGB -> tensor (3,16,224,224) normalizado."""
        if self.zoom < 1.0:
            side = int(round(arr.shape[1] * self.zoom))
            o = (arr.shape[1] - side) // 2
            arr = arr[:, o:o + side, o:o + side]
        L = arr.shape[1]
        m = int(L * 0.0625)
        arr = arr[:, m:L - m, m:L - m]

        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0  # T,H,W,C
        x = x.permute(3, 0, 1, 2)  # C,T,H,W
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean.cpu()) / self.std.cpu()
        return x

    @torch.no_grad()
    def score(self, tubes):
        """tubes: lista de arrays (16,H,W,3) uint8 RGB. Devuelve p(hurto)."""
        batch = torch.stack([self._prep_uno(t) for t in tubes]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.m(batch)
        return torch.softmax(out.float(), 1)[:, 1].cpu().numpy()

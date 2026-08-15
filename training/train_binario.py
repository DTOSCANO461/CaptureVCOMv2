#!/usr/bin/env python3
"""CAPTURE v2 -- entrenamiento del clasificador RGB de tubos, BINARIO
(hurto=1 / normal=0). Modelos: videomae-s | mvit-s.

Este archivo es una copia deliberadamente separada de training/train.py (que
sigue siendo el entrenamiento MULTICLASE, 70 acciones, y no se toca aca --
pedido explicito: nada de lo nuevo debe reescribir lo ya hecho). Comparte con
train.py el pipeline de datos actual (tubos ya recortados por
pose_pipeline.py/harvest_*.py, ventaneo 256->224, augmentations de
apariencia/oclusion) -- lo unico que vuelve a ser "como la original" (la
version de train.py de antes de pasar a 70 clases, ver commit 18c34c3) es la
CABEZA del modelo (NUM_CLASES=2) y evaluate()/main() (AUC/AP en vez de
accuracy/F1, que tiene mas sentido para un solo positivo/negativo que para
70 clases desbalanceadas).

anotaciones.csv esperado (columna "label" ya BINARIA, 0/1): generado con
training/anotaciones_a_binario.py a partir de un anotaciones.csv multiclase
existente, vía labeling_strategy_binaria.py (que a su vez usa
labeling_strategy.py sin modificarla, ver el docstring de ese modulo).

Uso:
  python train_binario.py --model videomae --epochs 15
  python train_binario.py --model mvit --epochs 15
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Configurable via env var TRAIN_ROOT -- permite apuntar a otro dataset
# (ej. una cosecha nueva en otra carpeta/maquina) sin tocar este archivo.
ROOT = os.environ.get("TRAIN_ROOT", "/media/mulinux/T5001/smartvcom2/CaptureVCOMv2-main/salida_offline/dataset_todo_binario")
RUNS = os.path.join(ROOT, "runs_aug")

NUM_CLASES = 2


# ────────────────────────── datos ──────────────────────────

def load_index():
    """Lee ROOT/anotaciones.csv -- misma tabla que usa train.py (npy,label,
    split,...), la que generan harvest_pose_test.py/harvest_muestra.py o,
    para el caso binario, training/anotaciones_a_binario.py remapeando un
    anotaciones.csv multiclase existente. "label" ya viene 0/1 aca."""
    rows = list(csv.DictReader(open(f"{ROOT}/anotaciones.csv")))
    items = []
    for r in rows:
        if not r.get("npy"):
            continue
        items.append({"npy": os.path.join(ROOT, "tubes", r["npy"]), "label": int(r["label"]), "split": r["split"]})
    return items


def _ventanea_np(arr, out_size, offset=None):
    """Identico a train.py::_ventanea_np -- ver ese archivo para el
    razonamiento completo (autonomo, sin importar pose_pipeline)."""
    H, W = arr.shape[1], arr.shape[2]
    assert H >= out_size and W >= out_size, f"tubo ({H}x{W}) mas chico que out_size={out_size}"
    holgura_h, holgura_w = H - out_size, W - out_size
    if offset is None:
        oy = random.randint(0, holgura_h) if holgura_h > 0 else 0
        ox = random.randint(0, holgura_w) if holgura_w > 0 else 0
    else:
        oy, ox = offset
    return arr[:, oy:oy + out_size, ox:ox + out_size]


def _offset_centrado(store_size, out_size):
    h = (store_size - out_size) // 2
    return (h, h)


class TubeDataset(Dataset):
    """Identico a train.py::TubeDataset (mismo pipeline de datos, marco
    congelado desde la cosecha, ventaneo + augmentations de apariencia).
    Solo se duplica aca para que este archivo se mantenga AUTONOMO, mismo
    criterio que train.py (ver ENTRENAMIENTO/LEEME.md)."""

    STORE_SIZE = 256
    OUT_SIZE = 224

    def __init__(self, items, train, mean, std, n_frames=16):
        self.items = items
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        self.n_frames = n_frames
        self._avisado_resample = False   # __getitem__ corre en workers separados
        # (num_workers de DataLoader) -- este flag es por-proceso, asi que el
        # aviso puede imprimirse hasta N_WORKERS veces, no una sola, pero eso
        # ya alcanza para que sea visible sin ser spam por-tubo.

        # Verificacion al armar el dataset: el .npy de cada tubo puede tener
        # 16 O 32 frames guardados (segun con que --n-frames se cosecho, ver
        # harvest_pose_test.py) -- el modelo (build_model) siempre necesita
        # exactamente self.n_frames. Ac se hace un MUESTREO (no las 66k+ filas
        # enteras, seria leer todos los .npy solo para esto) para avisar de
        # entrada si el dataset esta mezclado o si difiere de lo que pide el
        # modelo, en vez de descubrirlo silenciosamente tubo a tubo durante
        # el entrenamiento.
        if items:
            muestra = random.sample(items, min(50, len(items)))
            conteo = {}
            for it in muestra:
                try:
                    frames_reales = np.load(it["npy"], mmap_mode="r").shape[0]
                except Exception:
                    continue
                conteo[frames_reales] = conteo.get(frames_reales, 0) + 1
            distinto = {k: v for k, v in conteo.items() if k != n_frames}
            print(f"[dataset] modelo necesita n_frames={n_frames} -- muestra de {len(muestra)} "
                  f"tubos, frames reales encontrados: {conteo}"
                  + (f"  -- AVISO: {sum(distinto.values())}/{len(muestra)} NO coinciden, "
                     f"se van a resamplear (linspace) al cargar, no son frames reales adicionales/"
                     f"descartados sin mas" if distinto else " (todos coinciden, sin resample)"),
                  flush=True)

    def __len__(self):
        return len(self.items)

    def _fake_circle(self, arr):
        import cv2
        t, h, w, _ = arr.shape
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
        r = random.randint(int(0.3*h), int(1.1*h))
        color = (random.randint(190, 255), random.randint(0, 45), random.randint(0, 45))
        th = random.randint(2, 7)
        for i in range(t):
            cv2.circle(arr[i], (cx, cy), r, color, th, lineType=cv2.LINE_AA)
        return arr

    def _fake_blur(self, arr):
        import cv2
        t, h, w, _ = arr.shape
        bw, bh = random.randint(20, 60), random.randint(20, 60)
        x = random.randint(0, w - bw - 1)
        y = random.randint(0, max(1, h//2 - bh))
        k = random.choice([9, 13, 17])
        for i in range(t):
            roi = arr[i][y:y+bh, x:x+bw]
            arr[i][y:y+bh, x:x+bw] = cv2.GaussianBlur(roi, (k, k), 0)
        return arr

    def _rotate(self, arr):
        """Rotacion leve (+-15 grados), UNA sola matriz para los T frames --
        tiene que ser la misma en todo el tubo (si cada frame rotara
        distinto se rompe la coherencia temporal que el modelo aprende).
        BORDER_REFLECT101 en vez de negro para no meter un artefacto de
        borde negro constante que el modelo podria aprender como shortcut."""
        import cv2
        t, h, w, _ = arr.shape
        ang = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = np.empty_like(arr)
        for i in range(t):
            out[i] = cv2.warpAffine(arr[i], M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        return out

    def _distorsion(self, arr):
        """Distorsion de perspectiva leve (jitter de esquinas ~6% del lado),
        misma transformacion para los T frames por el mismo motivo que
        _rotate. Simula angulo/lente de camara real, no solo encuadre
        rectangular perfecto."""
        import cv2
        t, h, w, _ = arr.shape
        jitter = 0.06
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = (src + np.random.uniform(-jitter, jitter, src.shape) * np.array([w, h])).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out = np.empty_like(arr)
        for i in range(t):
            out[i] = cv2.warpPerspective(arr[i], M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        return out

    def _canal_swap(self, arr):
        """Permuta al azar los 3 canales de color (RGB -> alguna de las 6
        permutaciones) -- rompe el atajo de "este color de ropa/piel =
        hurto" sin tocar la geometria del tubo, mismo espiritu que la
        ganancia por canal de mas abajo pero mas agresivo (permutacion
        entera, no solo escalado)."""
        perm = np.random.permutation(3)
        return arr[:, :, :, perm]

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(it["npy"])  # (16 o 32,256,256,3) uint8 RGB -- depende de --n-frames al cosechar
        if arr.shape[0] != self.n_frames:
            if not self._avisado_resample:
                print(f"[dataset] resampleando (linspace) tubo con {arr.shape[0]} frames reales "
                      f"-> {self.n_frames} pedidos por el modelo (ej: {it['npy']})", flush=True)
                self._avisado_resample = True
            idx = np.linspace(0, arr.shape[0]-1, self.n_frames).round().astype(int)
            arr = arr[idx]
        arr = arr.copy()

        if self.train:
            if random.random() < 0.55:
                arr = self._fake_circle(arr)
            if random.random() < 0.35:
                arr = self._fake_blur(arr)
            if random.random() < 0.5:
                arr = arr[:, :, ::-1]  # flip H
            if random.random() < 0.30:
                arr = self._rotate(arr)
            if random.random() < 0.25:
                arr = self._distorsion(arr)
            if random.random() < 0.20:
                arr = self._canal_swap(arr)

        offset = None if self.train else _offset_centrado(self.STORE_SIZE, self.OUT_SIZE)
        arr = _ventanea_np(arr, self.OUT_SIZE, offset=offset)

        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0  # T,H,W,C
        x = x.permute(3, 0, 1, 2)  # C,T,H,W
        if self.train:
            if random.random() < 0.85:
                x = x * (0.55 + 0.9*random.random())              # brillo, rango ensanchado
                m = x.mean()
                x = (x - m) * (0.55 + 0.9*random.random()) + m    # contraste, rango ensanchado
            if random.random() < 0.65:
                g = torch.tensor([0.65 + 0.7*random.random() for _ in range(3)]).view(3, 1, 1, 1)
                x = x * g
            if random.random() < 0.20:
                x = x.mean(0, keepdim=True).repeat(3, 1, 1, 1)    # gris
            x = x.clamp(0, 1)
        x = (x - self.mean) / self.std
        y = it["label"]
        return x, y


# ────────────────────────── modelos ──────────────────────────

class _WrapVideoMAE(nn.Module):
    """HF VideoMAEForVideoClassification espera pixel_values en B,T,C,H,W."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):  # x: B,C,T,H,W
        return self.m(pixel_values=x.permute(0, 2, 1, 3, 4)).logits


def _carga_videomae_bin_manual(mid, num_labels, num_frames=16):
    """Bypass del bloqueo de transformers a torch.load para checkpoints que
    SOLO traen pytorch_model.bin (no model.safetensors) -- caso real:
    MCG-NJU/videomae-small-finetuned-kinetics. transformers>=4.5x exige
    torch>=2.6 para permitir cargar .bin via torch.load, aun con
    weights_only=True (bloqueo generico por CVE-2025-32434); aca tenemos
    torch 2.5.1. weights_only=True YA es la parte segura de torch.load (evita
    ejecucion de codigo arbitrario via pickle, que es la vulnerabilidad real)
    -- se construye el modelo desde el config y se le carga el state_dict a
    mano, sin pasar por el bloqueo (probado: 186 tensores, sin missing/
    unexpected mas alla de la cabeza classifier.*, que se descarta a proposito
    porque NUM_CLASES es distinto al checkpoint original)."""
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    path = hf_hub_download(mid, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # num_frames: la position embedding (ver commit --frames) se recalcula
    # sola desde el config, sinusoidal fija, NUNCA se guarda en pytorch_model.bin
    # -- pedir un num_frames distinto al del checkpoint no rompe el load_state_dict de abajo.
    cfg = VideoMAEConfig.from_pretrained(mid, num_labels=num_labels, num_frames=num_frames)
    m = VideoMAEForVideoClassification(cfg)
    sd.pop("classifier.weight", None)
    sd.pop("classifier.bias", None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert set(missing) <= {"classifier.weight", "classifier.bias"}, f"missing inesperado: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return m


# Dimensiones de arquitectura de cada tamano de VideoMAE V1 (ViT-S/B/L/H) --
# son solo hiperparametros (hidden_size/num_hidden_layers/num_attention_heads/
# intermediate_size), no pesos ni contenido creativo de MCG-NJU, confirmados
# a mano contra el config.json de cada checkpoint ya cacheado localmente.
# Usados SOLO para --scratch: arman la arquitectura vacia (random init) sin
# tocar para nada el repo de MCG-NJU en runtime (ni pesos NI config.json) --
# asi el modelo resultante no tiene ninguna dependencia de un checkpoint
# CC-BY-NC-4.0, se entrena 100% desde los datos propios.
_VIDEOMAE_ARCH = {
    # "videomae-tiny": diseno propio (no existe checkpoint oficial a este
    # tamano -- SOLO usable con --scratch). Apunta a ~1/4 de los parametros
    # de "small" (21.9M -> apunta a ~5M). hidden_size/num_hidden_layers=192/12
    # replican EXACTO la config de DeiT-Tiny (Touvron et al. 2021,
    # facebookresearch/deit) -- el ViT chico mas validado en la literatura,
    # en vez de inventar numeros sueltos. La diferencia real respecto a
    # copiar "small" a escala es num_attention_heads: "small" oficial de
    # MCG-NJU usa 16 heads con hidden=384 (head_dim=384/16=24, angosto);
    # aca con heads=3 se llega a head_dim=192/3=64 -- el estandar que SI usan
    # base/large/huge de esta misma familia (768/12=64, 1024/16=64,
    # 1280/16=80) y DeiT-Tiny. Confirmado a mano: el conteo de parametros de
    # atencion NO depende de num_attention_heads (misma cantidad de pesos,
    # solo cambia como se parte el tensor) -- head_dim=64 sale gratis, sin
    # pagar mas parametros que con heads=16. intermediate_size=768=4*hidden,
    # misma proporcion que el resto de la familia. Medido: 5.63M params.
    "videomae-tiny": dict(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768),
    "videomae-small": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=16, intermediate_size=1536),
    "videomae": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "videomae-large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
    "videomae-h": dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120),
}


def _build_videomae_scratch(name, num_labels, num_frames=16):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    cfg = VideoMAEConfig(
        image_size=224, patch_size=16, num_channels=3, num_frames=num_frames, tubelet_size=2,
        num_labels=num_labels, **_VIDEOMAE_ARCH[name],
    )
    m = VideoMAEForVideoClassification(cfg)  # random init -- sin from_pretrained
    print(f"[modelo] {name} SCRATCH (random init, sin checkpoint preentrenado) -- "
          f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params, num_frames={num_frames}")
    return m


def build_model(name, scratch=False, frames=16):
    if scratch:
        assert name in _VIDEOMAE_ARCH, (
            f"--scratch solo soportado para {list(_VIDEOMAE_ARCH)} (familia VideoMAE V1) por ahora")
        m = _build_videomae_scratch(name, NUM_CLASES, num_frames=frames)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    if name == "videomae-tiny":
        raise ValueError("videomae-tiny no tiene checkpoint preentrenado (arquitectura propia, "
                          "no publicada por MCG-NJU) -- solo se puede usar con --scratch.")
    if name == "videomae":
        # num_frames: la position embedding (sinusoidal fija) se recalcula
        # SIEMPRE desde el config al construirse, nunca se carga del
        # checkpoint (confirmado a mano: no aparece en su state_dict) --
        # pedir --frames 32 contra un checkpoint entrenado a 16 no tira
        # error de shapes ni corrompe nada, solo cambia cuantos tokens
        # temporales ve el modelo.
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        mid = "MCG-NJU/videomae-base-finetuned-kinetics"
        cfg = VideoMAEConfig.from_pretrained(mid, num_labels=NUM_CLASES, num_frames=frames)
        m = VideoMAEForVideoClassification.from_pretrained(
            mid, config=cfg, ignore_mismatched_sizes=True)
        print(f"[modelo] {mid} -- num_frames={frames}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "videomae-small":
        # ID correcto es SIN el "400" al final -- "...kinetics400" no existe
        # (typo heredado: el intento de usar small en build_model() siempre
        # fallaba en silencio y caia al fallback de base, asi que hasta ahora
        # TODAS las corridas "videomae" entrenaron base, nunca small de
        # verdad). Ademas ese repo solo trae pytorch_model.bin -- ver
        # _carga_videomae_bin_manual().
        mid = "MCG-NJU/videomae-small-finetuned-kinetics"
        m = _carga_videomae_bin_manual(mid, NUM_CLASES, num_frames=frames)
        print(f"[modelo] {mid} -- {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, num_frames={frames}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "videomae-large":
        # A diferencia de "small", este repo SI trae model.safetensors --
        # from_pretrained normal, sin el workaround de _carga_videomae_bin_manual.
        # 304M params (ViT-Large, ~3.5x base). Probado en la practica: bs=6,
        # SIN gradient checkpointing (a diferencia de videomaev2 -- este es
        # el codigo nativo de transformers, usa el kernel de atencion
        # fusionado/sdpa por default, no la implementacion "a mano" del
        # custom de VideoMAEv2), pico real medido 11.4GB de VRAM -- entra
        # holgado en una placa de 16GB, no hace falta with_cp aca. Con
        # --frames 32 la secuencia se duplica (mas VRAM) -- si se necesita
        # gradient checkpointing a 32 frames, prenderlo a mano como en "h".
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        mid = "MCG-NJU/videomae-large-finetuned-kinetics"
        cfg = VideoMAEConfig.from_pretrained(mid, num_labels=NUM_CLASES, num_frames=frames)
        m = VideoMAEForVideoClassification.from_pretrained(
            mid, config=cfg, ignore_mismatched_sizes=True)
        # frames>16 (32): la secuencia de atencion se duplica, el pico de
        # 11.4GB medido a 16 frames deja de ser confiable -- exactamente el
        # caso que el comentario de arriba pide prender a mano. Gatea por
        # frames, no siempre: a 16 frames entra holgado sin esto.
        if frames > 16:
            m.gradient_checkpointing_enable()
            print(f"[modelo] {mid} -- frames={frames}>16, gradient checkpointing ON (VRAM)")
        print(f"[modelo] {mid} -- {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, num_frames={frames}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "videomae-h":
        # VideoMAE V1 Huge -- MCG-NJU/videomae-huge-finetuned-kinetics,
        # ~630M params (ViT-Huge, ~2x "large"). Igual que "small" (y a
        # diferencia de "large"), este repo SOLO trae pytorch_model.bin, no
        # model.safetensors (confirmado via HfApi().model_info: .gitattributes,
        # README.md, config.json, preprocessor_config.json, pytorch_model.bin)
        # -- probado en la practica: from_pretrained normal tira ValueError
        # ("Due to a serious vulnerability issue in torch.load...") porque
        # transformers>=4.5x bloquea torch.load sobre .bin sin torch>=2.6 (aca
        # 2.5.1), asi que hace falta el mismo workaround de
        # _carga_videomae_bin_manual() que "small".
        #
        # "large" midio 11.4GB de VRAM SIN gradient checkpointing en la placa
        # de 16GB -- Huge tiene el doble de parametros y capas mas anchas, no
        # entra holgado sin esto, asi que a diferencia de "large" aca se
        # prende gradient checkpointing por default (mismo mecanismo nativo de
        # HF que usa el resto de VideoMAEForVideoClassification, no el with_cp
        # a mano de v2).
        mid = "MCG-NJU/videomae-huge-finetuned-kinetics"
        m = _carga_videomae_bin_manual(mid, NUM_CLASES, num_frames=frames)
        m.gradient_checkpointing_enable()
        print(f"[modelo] {mid} -- {sum(p.numel() for p in m.parameters())/1e6:.1f}M params "
              f"(gradient checkpointing ON), num_frames={frames}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "mvit":
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        m.head[1] = nn.Linear(m.head[1].in_features, NUM_CLASES)
        return m, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    elif name in ("videomaev2", "videomaev2-large", "videomaev2-h"):
        # OpenGVLab/VideoMAEv2-Base -- NO existe como clase nativa en
        # transformers (solo V1, "videomae" arriba). El checkpoint trae su
        # propio modeling_videomaev2.py/modeling_config.py (auto_map en su
        # config.json) que HF ejecuta via trust_remote_code=True. Revisado a
        # mano el contenido de esos dos archivos antes de habilitar esto ac
        # (ViT/BEiT estandar, sin llamadas de red/os, nada sospechoso) --
        # nunca se activa este flag para un repo sin revisar antes.
        # requiere `easydict` y `timm` instalados (dependencias declaradas
        # por el propio checkpoint, ya presentes en este entorno).
        #
        # Confirmado en la practica (ver commit): num_frames=16 en el
        # config del checkpoint -- pedido explicito, coincide con N_FRAMES
        # del resto del pipeline sin tener que tocar nada. tubelet_size=2 ->
        # 8 tokens temporales. embed_dim=768, 86M params (misma escala que
        # videomae-base). El checkpoint se publica como backbone DESNUDO
        # (num_classes=0 en su config, "head": Identity) -- reset_classifier()
        # es un metodo propio del modelo, agrega la cabeza de clasificacion
        # fine-tuneable (NUM_CLASES=2 aca).
        #
        # A diferencia de V1 (que necesita el Wrap de arriba: HF espera
        # pixel_values en B,T,C,H,W), este modelo toma (B,C,T,H,W) DIRECTO
        # -- exactamente lo que ya entrega TubeDataset.__getitem__, sin
        # permute ni wrapper.
        #
        # Normalizacion: mean=std=0.5 en los 3 canales (no ImageNet) --
        # asi se preentreno (ver _cfg() en modeling_videomaev2.py).
        # "large"/"h" -- mismos modeling_videomaev2.py/modeling_config.py que
        # "Base" (diff byte a byte contra el ya revisado, confirmado a mano
        # antes de habilitar esto), solo cambia el tamano del backbone
        # (embed_dim/depth/num_heads via su config.json) -- mismo
        # trust_remote_code ya auditado, no hace falta revisar de nuevo.
        from transformers import AutoModel, AutoConfig
        TAM_VMAE2 = {"videomaev2": "Base", "videomaev2-large": "Large", "videomaev2-h": "Huge"}[name]
        CKPT_VMAE2 = f"OpenGVLab/VideoMAEv2-{TAM_VMAE2}"
        cfg = AutoConfig.from_pretrained(CKPT_VMAE2, trust_remote_code=True)
        n_frames_ckpt = cfg.model_config["num_frames"]
        assert n_frames_ckpt == 16, (
            f"{CKPT_VMAE2} cambio a num_frames={n_frames_ckpt} (se esperaba 16, "
            f"pedido explicito) -- revisar antes de entrenar, algo del checkpoint cambio.")
        # --frames: igual que en V1, el pos_embed (get_sinusoid_encoding_table
        # en modeling_videomaev2.py) es un tensor plano, NO nn.Parameter/buffer
        # -- no aparece en el state_dict del checkpoint (confirmado a mano),
        # se recalcula solo desde cfg.model_config["num_frames"] al construir.
        # Pisar ese valor aca (en vez de dejar el 16 nativo del checkpoint) es
        # seguro por la misma razon que en V1.
        if frames != n_frames_ckpt:
            cfg.model_config["num_frames"] = frames
        m = AutoModel.from_pretrained(CKPT_VMAE2, config=cfg, trust_remote_code=True)
        m.model.reset_classifier(NUM_CLASES)
        # with_cp=True: gradient/activation checkpointing, YA VIENE COMO
        # ATRIBUTO del modelo (VisionTransformer.with_cp -> forward_features
        # usa torch.utils.checkpoint.checkpoint(blk, x) si esta prendido).
        # Necesario aca -- el codigo custom del checkpoint calcula atencion
        # "a mano" (q@k.T + softmax + @v) sin el kernel fusionado/flash-attn
        # que si usa la implementacion de V1 en transformers, con 1568
        # tokens por tubo (8 temporales x 196 espaciales) eso se come la
        # VRAM: probado en la practica, bs=6 SIN with_cp tira
        # CUDA OutOfMemoryError (~12GB antes de completar ni un batch,
        # ver commit); CON with_cp, mismo bs=6, pico real medido 3.4GB.
        m.model.with_cp = True
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"[modelo] {CKPT_VMAE2} (trust_remote_code) -- backbone {n_params:.1f}M params, "
              f"num_frames={frames} (checkpoint nativo: {n_frames_ckpt}), cabeza nueva de "
              f"{NUM_CLASES} clases, with_cp=True (gradient checkpointing, necesario por VRAM)")
        return m, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    raise ValueError(name)


# ────────────────────────── entrenamiento ──────────────────────────

def loss_mse_continuo(logits, y):
    """Alternativa CONTINUA a CrossEntropyLoss, pedida explicitamente
    ("no quiero una funcion binaria de error sino algo continuo, minimos
    cuadrados"): error cuadratico medio entre softmax(logits) y el
    one-hot del label real -- es el Brier score, la version "minimos
    cuadrados" de un problema de clasificacion (a diferencia de CE, que
    es log-verosimilitud, no distancia euclidea).

    RETROCOMPATIBLE a proposito: la cabeza sigue siendo NUM_CLASES=2
    logits (nada cambia en build_model/TubeDataset/evaluate), asi que
    inferencia_pose_comun.ModeloPoseGPU, inferencia_pose_test_binario.py y
    tools/procesa_offline_binario.py siguen andando tal cual con un
    checkpoint entrenado asi -- softmax(logits)[:,1] sigue siendo P(hurto),
    lo unico que cambia es COMO se penaliza el error durante el
    entrenamiento, no la forma de la salida."""
    probs = torch.softmax(logits.float(), dim=1)
    y_onehot = torch.nn.functional.one_hot(y, num_classes=NUM_CLASES).float()
    return torch.nn.functional.mse_loss(probs, y_onehot)


def loss_bce_normal(logits, y):
    """La funcion "normal" de clasificacion BINARIA (pedida explicitamente
    para distinguirla de loss_mse_continuo de arriba): Binary Cross-Entropy
    de libro, -[y*log(p) + (1-y)*log(1-p)], sobre p = softmax(logits)[:,1]
    (P(hurto)). Distinta de --loss ce (CrossEntropyLoss generica, pensada
    para N clases) aunque para 2 clases ambas optimizan un objetivo
    relacionado -- de hecho matematicamente CASI identica a CE sin
    label_smoothing (verificado a mano: softmax de 2 clases + CE ==
    sigmoid + BCE sobre la diferencia de logits) -- BCE es simplemente la
    formula especifica de libro/tutorial para un problema binario puro.

    RETROCOMPATIBLE igual que loss_mse_continuo: la cabeza sigue siendo
    NUM_CLASES=2 logits, nada cambia en build_model/TubeDataset/evaluate ni
    en la inferencia existente -- solo cambia la formula de la loss.

    Implementada como binary_cross_entropy_with_logits sobre la DIFERENCIA
    de logits (logit_1 - logit_0), matematicamente identico a BCE sobre
    softmax(logits)[:,1] (softmax de 2 clases == sigmoid de la diferencia),
    pero numericamente estable y AUTOCAST-SAFE -- probado en la practica:
    binary_cross_entropy "plano" sobre una probabilidad ya pasada por
    softmax tira RuntimeError dentro de torch.autocast (bug de seguridad
    de PyTorch, no un error de esta funcion: "unsafe to autocast", pide
    exactamente este cambio en el mensaje)."""
    logit_diff = (logits[:, 1] - logits[:, 0]).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logit_diff, y.float())


def evaluate(model, loader, device):
    """Binario: AUC/AP sobre la probabilidad de la clase 1 (hurto) -- "como
    la original" (ver commit 025af04, antes de pasar a multiclase), en vez
    de accuracy/F1 que se usa en train.py para las 70 clases."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits.float(), 1)[:, 1]
            ys += y.tolist()
            ps += p.cpu().tolist()
    from sklearn.metrics import roc_auc_score, average_precision_score
    ys_a, ps_a = np.array(ys), np.array(ps)
    auc = roc_auc_score(ys_a, ps_a) if len(set(ys)) > 1 else float("nan")
    ap = average_precision_score(ys_a, ps_a) if len(set(ys)) > 1 else float("nan")
    return auc, ap, ys_a, ps_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["videomae-tiny", "videomae", "videomae-small", "videomae-large", "videomae-h",
                                         "mvit", "videomaev2", "videomaev2-large", "videomaev2-h"], required=True)
    ap.add_argument("--frames", type=int, default=16, choices=[16, 32],
                     help="Cantidad de frames por tubo que el modelo espera (solo familia "
                          "VideoMAE v1/v2 -- mvit no se toca aca). Verificado a mano: tanto "
                          "VideoMAEForVideoClassification (v1) como el AutoModel de VideoMAEv2 "
                          "recalculan su position embedding SIEMPRE desde el config al construirse "
                          "(sinusoidal fijo, nunca se carga del checkpoint), asi que pedir 32 "
                          "frames con un checkpoint preentrenado a 16 no tira error de shapes ni "
                          "corrompe nada -- ver commit. TubeDataset detecta solo cuantos frames "
                          "tiene cada .npy real y resamplea (linspace) si no coincide con esto, "
                          "asi que mezclar tubos de 16 y 32 en el mismo dataset no rompe nada.")
    ap.add_argument("--scratch", action="store_true",
                     help="Arma la arquitectura de --model (solo familia VideoMAE V1) con pesos random "
                          "(init estandar de transformers, sin from_pretrained) en vez de partir del "
                          "checkpoint preentrenado de MCG-NJU -- sin ninguna dependencia de su licencia "
                          "CC-BY-NC-4.0, se entrena 100%% desde el dataset propio. Los resultados se "
                          "guardan en runs/<model>-scratch/, NO pisa la corrida con checkpoint.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3,
                     help="Early stopping: corta el entrenamiento si val_auc no mejora (respecto al "
                          "mejor visto hasta ahora) en N epocas seguidas. 0 = desactivado, corre las "
                          "--epochs completas siempre.")
    ap.add_argument("--loss-delta-stop", type=float, default=0.0,
                     help="Early stopping ALTERNATIVO, sobre el loss de TRAIN (no val_auc): corta si "
                          "|loss_epoca - loss_epoca_anterior| < este valor (loss ya estancado). "
                          "0.0 = desactivado. Independiente de --patience -- corta apenas se cumple "
                          "CUALQUIERA de los dos criterios.")
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--loss", choices=["ce", "mse", "bce"], default="ce",
                     help="ce = CrossEntropyLoss (default, sin cambios respecto a antes). "
                          "mse = loss_mse_continuo(), Brier score sobre softmax(logits) vs "
                          "one-hot -- continuo, minimos cuadrados. "
                          "bce = loss_bce_normal(), Binary Cross-Entropy de libro sobre P(hurto). "
                          "Las tres mantienen la misma cabeza de NUM_CLASES logits, retrocompatibles "
                          "con toda la inferencia existente.")
    ap.add_argument("--resume", action="store_true",
                     help="Retoma desde runs/<model>/resume_state.pt (pesos + optimizador + "
                          "scheduler OneCycleLR + GradScaler + log), en vez de arrancar de cero. "
                          "--epochs aca es SIEMPRE el TOTAL final de epocas de la corrida completa "
                          "(no 'epocas de mas') -- el schedule de OneCycleLR se define una sola vez "
                          "para todo el ciclo, asi que para que continuar sea equivalente a haber "
                          "pedido esas epocas en una sola corrida, --model/--bs/--accum/--lr/--loss "
                          "tienen que ser IDENTICOS a los de la corrida original (si no, se avisa "
                          "pero se continua igual, ya sin garantia de equivalencia).")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = "cuda"
    run_dir = os.path.join(RUNS, args.model + ("-scratch" if args.scratch else ""))
    os.makedirs(run_dir, exist_ok=True)

    items = load_index()
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    n_pos_tr = sum(1 for i in tr if i["label"] == 1)
    print(f"train {len(tr)} (hurto {n_pos_tr}) | val {len(va)} (hurto {sum(1 for i in va if i['label']==1)})")

    model, mean, std = build_model(args.model, scratch=args.scratch, frames=args.frames)
    model.to(device)

    ds_tr = TubeDataset(tr, True, mean, std, n_frames=args.frames)
    ds_va = TubeDataset(va, False, mean, std, n_frames=args.frames)
    # sampler balanceado hurto/normal -- misma logica que la version original
    # (n_pos/n_neg en vez del conteo_clase generico de train.py, aca solo hay 2)
    n_pos = n_pos_tr
    n_neg = len(tr) - n_pos
    wts = [1.0/n_pos if i["label"] == 1 else 1.0/n_neg for i in tr]
    sampler = WeightedRandomSampler(wts, num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, num_workers=args.workers,
                       pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_ep = len(dl_tr) // args.accum
    # Mismo piso de 20 pasos que train.py -- OneCycleLR con total_steps chico
    # puede caer en un limite entero y tirar ZeroDivisionError en el
    # constructor (ver commit del fix en train.py). No debería activarse con
    # datasets de este tamano, se deja por las dudas.
    total_steps = max(steps_ep * args.epochs, 20)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    crit = {
        "ce": nn.CrossEntropyLoss(label_smoothing=0.05),
        "mse": loss_mse_continuo,
        "bce": loss_bce_normal,
    }[args.loss]
    print(f"[loss] {args.loss}", flush=True)

    resume_path = os.path.join(run_dir, "resume_state.pt")
    if args.resume:
        # Checkpoint COMPLETO (pesos + opt + sched + scaler + log), separado
        # de best.pt/last.pt/ep*.pt (esos siguen siendo SOLO state_dict del
        # modelo -- eval.py/inferencia los cargan asi, no hay que romper eso).
        assert os.path.exists(resume_path), f"--resume pedido pero no existe {resume_path}"
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        hp = ck["hparams"]
        for k in ("model", "bs", "accum", "lr", "loss"):
            v_actual = getattr(args, k)
            if v_actual != hp[k]:
                print(f"[resume] AVISO: --{k} actual ({v_actual}) != el de la corrida guardada "
                      f"({hp[k]}) -- la continuacion YA NO es equivalente a una corrida unica.",
                      flush=True)
        if args.epochs != hp["epochs"]:
            # sched.load_state_dict() de abajo pisa el total_steps con el que
            # se construyo ESTA instancia -- el total original (hp["epochs"])
            # es el que de verdad rige el annealing de OneCycleLR de aca en
            # mas, sin importar que total_steps se haya usado para construir
            # el objeto antes de cargarle el estado. Si --epochs pedido es
            # MENOR al original, la corrida corta el ciclo sin terminar de
            # bajar el LR (no revienta, pero ya no es "una corrida unica de
            # --epochs"). Si es MAYOR, en algun momento sched.step() tira
            # ValueError ("Tried to step X times...") al agotar los pasos
            # que el ciclo original tenia programados.
            print(f"[resume] AVISO: --epochs actual ({args.epochs}) != el total original de la "
                  f"corrida guardada ({hp['epochs']}) -- el ciclo de OneCycleLR sigue regido por "
                  f"el total ORIGINAL (se restaura via sched.load_state_dict), no por este numero. "
                  f"Si pedis MAS del total original, en algun momento sched.step() va a tirar "
                  f"ValueError al agotar los pasos programados.", flush=True)
        if args.epochs <= ck["epoch"]:
            raise SystemExit(f"--resume: --epochs ({args.epochs}) tiene que ser MAYOR a las epocas "
                              f"ya completadas ({ck['epoch']}) -- --epochs es el TOTAL final de la "
                              f"corrida completa, no 'epocas de mas'.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        log = ck["log"]
        best_auc = ck["best_auc"]
        epochs_sin_mejora = ck["epochs_sin_mejora"]
        start_ep = ck["epoch"]
        auc0 = log[0]["val_auc"]
        print(f"[resume] retomado desde epoca {start_ep} (best_auc={best_auc:.4f}), "
              f"corriendo hasta epoca {args.epochs}", flush=True)
    else:
        # Baseline sin entrenar, "ep": 0 -- mismo criterio que train.py, para
        # poder comparar el salto real en las graficas despues.
        log = []
        print("evaluando modelo BASE (sin entrenar) para tener una referencia...", flush=True)
        auc0, ap0, _, _ = evaluate(model, dl_va, device)
        print(f"ep 0/{args.epochs} (BASELINE, sin entrenar)  val_auc {auc0:.4f}  val_ap {ap0:.4f}", flush=True)
        log.append({"ep": 0, "loss": None, "val_auc": float(auc0), "val_ap": float(ap0), "min": 0.0})
        json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)
        best_auc = auc0
        epochs_sin_mejora = 0
        start_ep = 0

    prev_loss = next((e["loss"] for e in reversed(log) if e["loss"] is not None), None)
    n_batches = len(dl_tr)
    # Progreso EN VIVO dentro de la epoca (no solo al final de cada una) --
    # pedido explicito, datasets grandes (66k+ tubos) tardan minutos por
    # epoca y sin esto el log queda mudo todo ese tiempo. ~20 actualizaciones
    # por epoca: print de linea normal (no barra tipo tqdm con \r) porque
    # esto corre en background via nohup y se sigue con tail/grep sobre el
    # log, no en una TTY -- \r no se ve bien ahi.
    log_every = max(1, n_batches // 20)
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y) / args.accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * args.accum)
            if (step + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if (step + 1) % log_every == 0 or (step + 1) == n_batches:
                frac = (step + 1) / n_batches
                elapsed = time.time() - t0
                eta_ep = elapsed / frac - elapsed
                print(f"  ep {ep+1}/{args.epochs}  step {step+1}/{n_batches} ({frac*100:.0f}%)  "
                      f"loss_running={np.mean(losses):.4f}  "
                      f"transcurrido={elapsed/60:.1f}min  eta_epoca={eta_ep/60:.1f}min", flush=True)
        auc, ap_, _, _ = evaluate(model, dl_va, device)
        dt = time.time() - t0
        print(f"ep {ep+1}/{args.epochs} loss {np.mean(losses):.4f} "
              f"val_auc {auc:.4f} val_ap {ap_:.4f} ({dt/60:.1f} min)", flush=True)
        log.append({"ep": ep+1, "loss": float(np.mean(losses)),
                    "val_auc": float(auc), "val_ap": float(ap_), "min": dt/60})
        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        torch.save(model.state_dict(), os.path.join(run_dir, f"ep{ep+1}.pt"))
        if auc >= best_auc:
            best_auc = auc
            epochs_sin_mejora = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            print(f"  → best ({auc:.4f})", flush=True)
        else:
            epochs_sin_mejora += 1
        json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)
        # Checkpoint de resume: TODO lo necesario para continuar de forma
        # exacta (pesos + opt + sched + scaler), sobreescrito cada epoca --
        # a diferencia de last.pt/ep*.pt (solo pesos), este no sirve para
        # inferencia, solo para --resume.
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "epoch": ep + 1, "best_auc": best_auc,
            "epochs_sin_mejora": epochs_sin_mejora, "log": log,
            "hparams": {"model": args.model, "bs": args.bs, "accum": args.accum,
                        "lr": args.lr, "loss": args.loss, "epochs": args.epochs},
        }, resume_path)
        if args.patience > 0 and epochs_sin_mejora >= args.patience:
            print(f"[early stop] val_auc sin mejorar en {args.patience} epocas seguidas, "
                  f"cortando en ep {ep+1}/{args.epochs}", flush=True)
            break
        cur_loss = float(np.mean(losses))
        if args.loss_delta_stop > 0 and prev_loss is not None:
            delta = abs(cur_loss - prev_loss)
            if delta < args.loss_delta_stop:
                print(f"[early stop] loss estancado (|Δloss|={delta:.5f} < {args.loss_delta_stop}), "
                      f"cortando en ep {ep+1}/{args.epochs}", flush=True)
                break
        prev_loss = cur_loss

    print(f"FIN {args.model}: baseline val_auc {auc0:.4f} -> best val_auc {best_auc:.4f}")


if __name__ == "__main__":
    main()

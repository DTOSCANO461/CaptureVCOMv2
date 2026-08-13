#!/usr/bin/env python3
"""CAPTURE v2 — Fase 1: entrenamiento del clasificador RGB de tubos.

Binario: hurto (1) vs normal (0). Modelos: videomae-s | mvit-s.

Uso:
  python train.py --model videomae --epochs 15
  python train.py --model mvit --epochs 15
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

ROOT = "/home/tomis/SBT/DATASETS"
RUNS = os.path.join(ROOT, "train", "runs")

# Multiclase: hasta 70 acciones (0-69; 69 es DEFAULT_ACCION en
# labeling_strategy.py, el placeholder de "no se pudo resolver un label
# real"). Antes esto era binario (2 = hurto/normal) -- build_model() y
# TubeDataset todavia asumen ese binario en otros puntos (WeightedRandomSampler
# balanceado por "clase" hurto/normal, evaluate() con AUC/AP de 1 sola
# columna): falta actualizar esos para multiclase de verdad, esto solo
# cambia el tamano de la cabeza.
NUM_CLASES = 70


# ────────────────────────── datos ──────────────────────────

def load_index():
    """Lee ROOT/anotaciones.csv -- UNA sola tabla (npy,label,split,...), la que
    genera training/harvest_pose_test.py / harvest_muestra.py. Reemplaza el
    join labels.csv/tubes_meta.csv del dataset viejo -- ya no aplica.

    "label" es el codigo numerico multi-clase de labeling_strategy.py
    (0-69), NO el binario hurto/normal de antes."""
    rows = list(csv.DictReader(open(f"{ROOT}/anotaciones.csv")))
    items = []
    for r in rows:
        if not r.get("npy"):
            continue
        items.append({"npy": os.path.join(ROOT, "tubes", r["npy"]), "label": int(r["label"]), "split": r["split"]})
    return items


def _ventanea_np(arr, out_size, offset=None):
    """arr: (T,H,W,C) uint8. Slice puro (sin resize) de out_size x out_size.
    Mismo criterio que pose_pipeline.ventanea (CaptureVCOMv2/pose_pipeline.py) --
    reimplementado aca en numpy puro (sin importar pose_pipeline, sin torch)
    para que este archivo se mantenga AUTONOMO: solo necesita el .npy y las
    anotaciones, nada mas del repo (ver ENTRENAMIENTO/LEEME.md).
    offset=None -> sortea uno aleatorio en [0,holgura] por eje (entrenamiento).
    offset=(oy,ox) -> ventana fija (eval: offset_centrado() mas abajo).
    """
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
    """Tubos ya vienen recortados con el marco final (STORE_SIZE=256) desde la
    cosecha (pose_pipeline.py + harvest_pose_test.py, o su version real) -- el
    zoom/marco queda CONGELADO ahi, ya no es un parametro de este dataset (a
    diferencia del dataset viejo). Lo unico que se hace aca es ventanear
    (slice puro 256->224, ver _ventanea_np) para la augmentation de posicion."""

    STORE_SIZE = 256
    OUT_SIZE = 224

    def __init__(self, items, train, mean, std, n_frames=16):
        self.items = items
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        self.n_frames = n_frames

    def __len__(self):
        return len(self.items)

    # — simetrización de artefactos de producción —
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
        arr = np.load(it["npy"])  # (16,256,256,3) uint8 RGB
        if arr.shape[0] != self.n_frames:
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

        # Ventaneo: slice puro STORE_SIZE(256) -> OUT_SIZE(224), sin resize --
        # augmentation de posicion en train (offset aleatorio), centrado en
        # eval. El zoom/marco ya viene fijo desde la cosecha, no se toca aca.
        offset = None if self.train else _offset_centrado(self.STORE_SIZE, self.OUT_SIZE)
        arr = _ventanea_np(arr, self.OUT_SIZE, offset=offset)

        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0  # T,H,W,C
        x = x.permute(3, 0, 1, 2)  # C,T,H,W
        if self.train:
            # Augmentación de APARIENCIA. Con solo ~6 actores en el dataset (uno
            # de ellos el 59%), el modelo puede agarrarse a "esta persona/ropa =
            # hurto". Variar color/brillo por canal cambia la ropa de un clip a
            # otro y rompe ese atajo de identidad.
            if random.random() < 0.85:
                x = x * (0.55 + 0.9*random.random())              # brillo, rango ensanchado
                m = x.mean()
                x = (x - m) * (0.55 + 0.9*random.random()) + m    # contraste, rango ensanchado
            if random.random() < 0.65:
                # ganancia independiente por canal → cambia el color de la ropa
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


def _carga_videomae_bin_manual(mid, num_labels):
    """Ver docstring identico en train_binario.py::build_model() -- bypass
    del bloqueo de transformers a torch.load para checkpoints que SOLO
    traen pytorch_model.bin (caso: MCG-NJU/videomae-small-finetuned-kinetics),
    exige torch>=2.6 aun con weights_only=True (tenemos 2.5.1). weights_only=True
    ya es la parte segura (evita ejecucion de codigo via pickle)."""
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    path = hf_hub_download(mid, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    cfg = VideoMAEConfig.from_pretrained(mid, num_labels=num_labels)
    m = VideoMAEForVideoClassification(cfg)
    sd.pop("classifier.weight", None)
    sd.pop("classifier.bias", None)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert set(missing) <= {"classifier.weight", "classifier.bias"}, f"missing inesperado: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return m


def build_model(name):
    if name == "videomae":
        from transformers import VideoMAEForVideoClassification
        mid = "MCG-NJU/videomae-base-finetuned-kinetics"
        m = VideoMAEForVideoClassification.from_pretrained(
            mid, num_labels=NUM_CLASES, ignore_mismatched_sizes=True)
        print(f"[modelo] {mid}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "videomae-small":
        # ID correcto es SIN el "400" final -- ver comentario identico en
        # train_binario.py::build_model() para el detalle completo (typo
        # heredado, todas las corridas "videomae" hasta ahora entrenaron
        # base, nunca small de verdad; ademas ese repo solo trae
        # pytorch_model.bin, ver _carga_videomae_bin_manual()).
        mid = "MCG-NJU/videomae-small-finetuned-kinetics"
        m = _carga_videomae_bin_manual(mid, NUM_CLASES)
        print(f"[modelo] {mid} -- {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return _WrapVideoMAE(m), mean, std
    elif name == "mvit":
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        m.head[1] = nn.Linear(m.head[1].in_features, NUM_CLASES)
        return m, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    elif name == "videomaev2":
        # OpenGVLab/VideoMAEv2-Base -- ver el docstring identico en
        # training/train_binario.py::build_model() para el detalle completo
        # (por que trust_remote_code=True, que se reviso a mano, num_frames=16
        # confirmado, input (B,C,T,H,W) directo sin Wrap, normalizacion 0.5).
        # NUM_CLASES aca es 70 (multiclase), no 2.
        from transformers import AutoModel, AutoConfig
        CKPT_VMAE2 = "OpenGVLab/VideoMAEv2-Base"
        cfg = AutoConfig.from_pretrained(CKPT_VMAE2, trust_remote_code=True)
        n_frames_ckpt = cfg.model_config["num_frames"]
        assert n_frames_ckpt == 16, (
            f"OpenGVLab/VideoMAEv2-Base cambio a num_frames={n_frames_ckpt} (se esperaba 16, "
            f"pedido explicito) -- revisar antes de entrenar, la pos_embed del checkpoint "
            f"esta atada a ese numero de frames.")
        m = AutoModel.from_pretrained(CKPT_VMAE2, trust_remote_code=True)
        m.model.reset_classifier(NUM_CLASES)
        # with_cp=True: gradient checkpointing -- ver el comentario identico
        # en train_binario.py::build_model() para el detalle completo
        # (atencion "a mano" del codigo custom, sin flash-attn; sin esto
        # tira CUDA OutOfMemoryError con bs=6, con esto pico medido 3.4GB).
        m.model.with_cp = True
        print(f"[modelo] {CKPT_VMAE2} (trust_remote_code) -- backbone 86M params, "
              f"num_frames={n_frames_ckpt}, cabeza nueva de {NUM_CLASES} clases, "
              f"with_cp=True (gradient checkpointing, necesario por VRAM)")
        return m, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    raise ValueError(name)


# ────────────────────────── entrenamiento ──────────────────────────

def evaluate(model, loader, device):
    """Multiclase (hasta NUM_CLASES): accuracy + F1 ponderado por soporte
    (weighted, no macro -- con 500 tubos repartidos en ~70 clases muchas
    tienen 1-2 muestras en val, macro les daria el mismo peso que a las
    clases grandes y el numero saldria dominado por ruido). Devuelve tambien
    ys/preds/probs crudos para poder graficar matriz de confusion despues."""
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits.float(), 1)
            ys += y.tolist()
            preds += p.argmax(1).cpu().tolist()
            probs.append(p.cpu())
    from sklearn.metrics import accuracy_score, f1_score
    ys_a, preds_a = np.array(ys), np.array(preds)
    acc = accuracy_score(ys_a, preds_a) if len(ys_a) else float("nan")
    f1 = f1_score(ys_a, preds_a, average="weighted", zero_division=0) if len(ys_a) else float("nan")
    probs_a = torch.cat(probs).numpy() if probs else np.zeros((0, NUM_CLASES))
    return acc, f1, ys_a, preds_a, probs_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["videomae", "videomae-small", "mvit", "videomaev2"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = "cuda"
    run_dir = os.path.join(RUNS, args.model)
    os.makedirs(run_dir, exist_ok=True)

    items = load_index()
    tr = [i for i in items if i["split"] == "train"]
    va = [i for i in items if i["split"] == "val"]
    n_clases_tr = len(set(i["label"] for i in tr))
    print(f"train {len(tr)} ({n_clases_tr} clases distintas) | val {len(va)}")

    model, mean, std = build_model(args.model)
    model.to(device)

    ds_tr = TubeDataset(tr, True, mean, std)
    ds_va = TubeDataset(va, False, mean, std)
    # sampler balanceado POR CLASE (antes binario hurto/normal): peso
    # inversamente proporcional al conteo de esa clase en train, asi las
    # clases chicas (algunas con 1-2 tubos en una muestra de 500) no quedan
    # invisibles frente a las grandes (ej. label 48, que junta control_negativo
    # Y doblar_sobreponer Y varios "no_obvio").
    conteo_clase = {}
    for i in tr:
        conteo_clase[i["label"]] = conteo_clase.get(i["label"], 0) + 1
    wts = [1.0 / conteo_clase[i["label"]] for i in tr]
    sampler = WeightedRandomSampler(wts, num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, num_workers=args.workers,
                       pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_ep = len(dl_tr) // args.accum
    # OneCycleLR con total_steps chico puede caer justo en un limite entero
    # (ej. total_steps=10, pct_start=0.1 -> 0.1*10=1.0 -> la fase de warmup
    # queda con end_step==start_step==0 -> ZeroDivisionError en el
    # constructor mismo). Piso de 20 pasos evita esa condicion para
    # pct_start=0.1 (hace falta pct_start*total_steps >= 2 para que la
    # primera fase tenga al menos 1 paso de ancho) -- visto en vivo con
    # exactamente 359 tubos, ver commit del bug.
    total_steps = max(steps_ep * args.epochs, 20)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Baseline: metricas del modelo SIN entrenar (cabeza de NUM_CLASES recien
    # inicializada al azar sobre el backbone preentrenado en Kinetics) -- se
    # loguea como "ep": 0, antes de que corra ninguna epoca de entrenamiento,
    # para poder comparar contra esto en las graficas despues.
    log = []
    print("evaluando modelo BASE (sin entrenar) para tener una referencia...", flush=True)
    acc0, f1_0, _, _, _ = evaluate(model, dl_va, device)
    print(f"ep 0/{args.epochs} (BASELINE, sin entrenar)  val_acc {acc0:.4f}  val_f1 {f1_0:.4f}", flush=True)
    log.append({"ep": 0, "loss": None, "val_acc": float(acc0), "val_f1": float(f1_0), "min": 0.0})
    json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)

    best_f1 = f1_0
    for ep in range(args.epochs):
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
        acc, f1, _, _, _ = evaluate(model, dl_va, device)
        dt = time.time() - t0
        print(f"ep {ep+1}/{args.epochs} loss {np.mean(losses):.4f} "
              f"val_acc {acc:.4f} val_f1 {f1:.4f} ({dt/60:.1f} min)", flush=True)
        log.append({"ep": ep+1, "loss": float(np.mean(losses)),
                    "val_acc": float(acc), "val_f1": float(f1), "min": dt/60})
        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            print(f"  → best ({f1:.4f})", flush=True)
        json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)

    print(f"FIN {args.model}: baseline val_f1 {f1_0:.4f} -> best val_f1 {best_f1:.4f}")


if __name__ == "__main__":
    main()

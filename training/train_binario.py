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

ROOT = "/home/tomis/SBT/DATASETS"
RUNS = os.path.join(ROOT, "train", "runs")

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

def build_model(name):
    if name == "videomae":
        from transformers import VideoMAEForVideoClassification
        last_err = None
        for mid in ["MCG-NJU/videomae-small-finetuned-kinetics400",
                    "MCG-NJU/videomae-base-finetuned-kinetics"]:
            try:
                m = VideoMAEForVideoClassification.from_pretrained(
                    mid, num_labels=NUM_CLASES, ignore_mismatched_sizes=True)
                print(f"[modelo] {mid}")
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

                class Wrap(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m

                    def forward(self, x):  # x: B,C,T,H,W
                        return self.m(pixel_values=x.permute(0, 2, 1, 3, 4)).logits
                return Wrap(m), mean, std
            except Exception as e:
                last_err = e
                print(f"[modelo] {mid} no disponible: {str(e)[:100]}")
        raise last_err
    elif name == "mvit":
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        m.head[1] = nn.Linear(m.head[1].in_features, NUM_CLASES)
        return m, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    elif name == "videomaev2":
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
        print(f"[modelo] {CKPT_VMAE2} (trust_remote_code) -- backbone 86M params, "
              f"num_frames={n_frames_ckpt}, cabeza nueva de {NUM_CLASES} clases")
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
    ap.add_argument("--model", choices=["videomae", "mvit", "videomaev2"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--loss", choices=["ce", "mse"], default="ce",
                     help="ce = CrossEntropyLoss (default, sin cambios respecto a antes). "
                          "mse = loss_mse_continuo(), Brier score sobre softmax(logits) vs "
                          "one-hot -- experimento pedido explicitamente, misma cabeza de "
                          "NUM_CLASES logits, retrocompatible con toda la inferencia existente.")
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
    n_pos_tr = sum(1 for i in tr if i["label"] == 1)
    print(f"train {len(tr)} (hurto {n_pos_tr}) | val {len(va)} (hurto {sum(1 for i in va if i['label']==1)})")

    model, mean, std = build_model(args.model)
    model.to(device)

    ds_tr = TubeDataset(tr, True, mean, std)
    ds_va = TubeDataset(va, False, mean, std)
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
    crit = nn.CrossEntropyLoss(label_smoothing=0.05) if args.loss == "ce" else loss_mse_continuo
    print(f"[loss] {args.loss}", flush=True)

    # Baseline sin entrenar, "ep": 0 -- mismo criterio que train.py, para
    # poder comparar el salto real en las graficas despues.
    log = []
    print("evaluando modelo BASE (sin entrenar) para tener una referencia...", flush=True)
    auc0, ap0, _, _ = evaluate(model, dl_va, device)
    print(f"ep 0/{args.epochs} (BASELINE, sin entrenar)  val_auc {auc0:.4f}  val_ap {ap0:.4f}", flush=True)
    log.append({"ep": 0, "loss": None, "val_auc": float(auc0), "val_ap": float(ap0), "min": 0.0})
    json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)

    best_auc = auc0
    n_batches = len(dl_tr)
    # Progreso EN VIVO dentro de la epoca (no solo al final de cada una) --
    # pedido explicito, datasets grandes (66k+ tubos) tardan minutos por
    # epoca y sin esto el log queda mudo todo ese tiempo. ~20 actualizaciones
    # por epoca: print de linea normal (no barra tipo tqdm con \r) porque
    # esto corre en background via nohup y se sigue con tail/grep sobre el
    # log, no en una TTY -- \r no se ve bien ahi.
    log_every = max(1, n_batches // 20)
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
        if auc >= best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            print(f"  → best ({auc:.4f})", flush=True)
        json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)

    print(f"FIN {args.model}: baseline val_auc {auc0:.4f} -> best val_auc {best_auc:.4f}")


if __name__ == "__main__":
    main()

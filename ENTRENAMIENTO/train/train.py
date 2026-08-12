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


# ────────────────────────── datos ──────────────────────────

def load_index():
    labels = {r["path"]: r for r in csv.DictReader(open(f"{ROOT}/audit/labels.csv"))}
    tubes = list(csv.DictReader(open(f"{ROOT}/audit/tubes_meta.csv")))
    items = []
    for t in tubes:
        if not t["npy"]:
            continue
        r = labels.get(t["path"])
        if not r:
            continue
        items.append({
            "npy": os.path.join(ROOT, "tubes", t["npy"]),
            "clase": r["clase"], "split": r["split"], "dataset": r["dataset"],
            "flag": t["flag"], "score_prod": r.get("score_prod", ""),
        })
    return items


class TubeDataset(Dataset):
    """zoom: fracción central del tubo que se conserva. El tubo cubre 1.9x la caja
    de la persona, así que zoom<1 recorta fondo de tienda y obliga al modelo a
    fijarse en la persona y no en el escenario (evita el atajo showroom/súper)."""

    def __init__(self, items, train, mean, std, n_frames=16, zoom=1.0):
        self.items = items
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        self.n_frames = n_frames
        self.zoom = zoom

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

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(it["npy"])  # (16,256,256,3) uint8 RGB
        if arr.shape[0] != self.n_frames:
            idx = np.linspace(0, arr.shape[0]-1, self.n_frames).round().astype(int)
            arr = arr[idx]
        arr = arr.copy()

        if self.zoom < 1.0:
            # recorte central: menos fondo de tienda, más persona
            z = self.zoom * (random.uniform(0.92, 1.08) if self.train else 1.0)
            z = float(np.clip(z, 0.35, 1.0))
            side = int(round(arr.shape[1] * z))
            o = (arr.shape[1] - side) // 2
            arr = np.ascontiguousarray(arr[:, o:o+side, o:o+side])

        if self.train:
            if random.random() < 0.5:
                arr = self._fake_circle(arr)
            if random.random() < 0.30:
                arr = self._fake_blur(arr)
            # crop espacial aleatorio (relativo al lado actual, que depende de zoom)
            L = arr.shape[1]
            s = random.randint(int(L * 0.78), L)
            y0 = random.randint(0, L - s)
            x0 = random.randint(0, L - s)
            arr = arr[:, y0:y0+s, x0:x0+s]
            if random.random() < 0.5:
                arr = arr[:, :, ::-1]  # flip H
        else:
            L = arr.shape[1]
            m = int(L * 0.0625)          # margen equivalente a 16/256
            arr = arr[:, m:L-m, m:L-m]   # center crop

        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0  # T,H,W,C
        x = x.permute(3, 0, 1, 2)  # C,T,H,W
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                                align_corners=False)
        if self.train:
            # Augmentación de APARIENCIA. Con solo ~6 actores en el dataset (uno
            # de ellos el 59%), el modelo puede agarrarse a "esta persona/ropa =
            # hurto". Variar color/brillo por canal cambia la ropa de un clip a
            # otro y rompe ese atajo de identidad.
            if random.random() < 0.8:
                x = x * (0.65 + 0.7*random.random())              # brillo
                m = x.mean()
                x = (x - m) * (0.65 + 0.7*random.random()) + m    # contraste
            if random.random() < 0.6:
                # ganancia independiente por canal → cambia el color de la ropa
                g = torch.tensor([0.75 + 0.5*random.random() for _ in range(3)]).view(3, 1, 1, 1)
                x = x * g
            if random.random() < 0.20:
                x = x.mean(0, keepdim=True).repeat(3, 1, 1, 1)    # gris
            x = x.clamp(0, 1)
        x = (x - self.mean) / self.std
        y = 1 if it["clase"] == "hurto" else 0
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
                    mid, num_labels=2, ignore_mismatched_sizes=True)
                print(f"[modelo] {mid}")
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                # HF espera (B,T,C,H,W)
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
        m.head[1] = nn.Linear(m.head[1].in_features, 2)
        return m, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
    raise ValueError(name)


# ────────────────────────── entrenamiento ──────────────────────────

def evaluate(model, loader, device):
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
    ap.add_argument("--model", choices=["videomae", "mvit"], required=True)
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
    tr = [i for i in items if i["split"] == "train" and i["clase"] in ("hurto", "normal")]
    va = [i for i in items if i["split"] == "val" and i["clase"] in ("hurto", "normal")]
    print(f"train {len(tr)} (hurto {sum(1 for i in tr if i['clase']=='hurto')}) | val {len(va)}")

    model, mean, std = build_model(args.model)
    model.to(device)

    ds_tr = TubeDataset(tr, True, mean, std)
    ds_va = TubeDataset(va, False, mean, std)
    # sampler balanceado
    n_h = sum(1 for i in tr if i["clase"] == "hurto")
    n_n = len(tr) - n_h
    wts = [1.0/n_h if i["clase"] == "hurto" else 1.0/n_n for i in tr]
    sampler = WeightedRandomSampler(wts, num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, num_workers=args.workers,
                       pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_ep = len(dl_tr) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, steps_ep*args.epochs), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_auc, log = 0.0, []
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

    print(f"FIN {args.model}: best val AUC {best_auc:.4f}")


if __name__ == "__main__":
    main()

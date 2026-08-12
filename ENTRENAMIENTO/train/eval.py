#!/usr/bin/env python3
"""Evaluación final Fase 1 sobre test: la tabla del veredicto.

- AUC/AP en test (hurto vs normal), tiendas/actores nunca vistos.
- Barrido de umbral: recall en hurtos-test vs % de reingestas de producción
  que se dispararían (las reingestas fueron TODAS disparadas por el sistema
  actual → % rechazado ≈ reducción de FPs, con el matiz de robos reales
  que pueda haber dentro).
- Recall sobre no_obvio (nunca entrenados).

Uso: python eval.py --model videomae
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TubeDataset, build_model, load_index, ROOT, RUNS


def score(model, items, mean, std, device, bs=8):
    ds = TubeDataset(items, False, mean, std)
    dl = DataLoader(ds, batch_size=bs, num_workers=5, pin_memory=True)
    ps = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, _ in dl:
            p = torch.softmax(model(x.to(device)).float(), 1)[:, 1]
            ps += p.cpu().tolist()
    return np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["videomae", "mvit"], required=True)
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    device = "cuda"

    model, mean, std = build_model(args.model)
    sd = torch.load(os.path.join(RUNS, args.model, args.ckpt), map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)

    items = load_index()
    te_h = [i for i in items if i["split"] == "test" and i["clase"] == "hurto"]
    te_n = [i for i in items if i["split"] == "test" and i["clase"] == "normal"]
    te_no = [i for i in items if i["split"] == "test" and i["clase"] == "no_obvio_eval"]
    fp = [i for i in items if i["clase"] == "fp_eval"]
    print(f"test: hurto {len(te_h)} | normal {len(te_n)} | no_obvio {len(te_no)} | reingesta {len(fp)}")

    p_h = score(model, te_h, mean, std, device)
    p_n = score(model, te_n, mean, std, device)
    p_no = score(model, te_no, mean, std, device)
    p_fp = score(model, fp, mean, std, device)

    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.r_[np.ones(len(p_h)), np.zeros(len(p_n))]
    p = np.r_[p_h, p_n]
    print(f"\nAUC test (hurto vs normal, nunca vistos): {roc_auc_score(y, p):.4f}")
    print(f"AP  test: {average_precision_score(y, p):.4f}\n")

    print(f"{'umbral':>7} {'recall hurto':>13} {'FP normales':>12} {'reingesta disparada':>20} {'recall no_obvio':>16}")
    rows = []
    for target in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        thr = float(np.quantile(p_h, 1 - target))
        rec = float((p_h >= thr).mean())
        fpr_n = float((p_n >= thr).mean())
        fire = float((p_fp >= thr).mean())
        rec_no = float((p_no >= thr).mean())
        rows.append({"target": target, "thr": thr, "recall": rec,
                     "fp_normal": fpr_n, "reingesta_fire": fire, "recall_no_obvio": rec_no})
        print(f"{thr:7.3f} {rec:13.1%} {fpr_n:12.1%} {fire:20.1%} {rec_no:16.1%}")

    # por origen del test de hurto
    print("\nRecall por origen (umbral @90%):")
    thr90 = float(np.quantile(p_h, 0.10))
    for ds_name in sorted({i["dataset"] for i in te_h}):
        mask = np.array([i["dataset"] == ds_name for i in te_h])
        print(f"  {ds_name:22} {float((p_h[mask] >= thr90).mean()):.1%} (n={mask.sum()})")
    print("\nReingesta disparada por dataset (umbral @90%):")
    for ds_name in sorted({i["dataset"] for i in fp}):
        mask = np.array([i["dataset"] == ds_name for i in fp])
        print(f"  {ds_name:22} {float((p_fp[mask] >= thr90).mean()):.1%} (n={mask.sum()})")

    # segmentado por score de producción (≥0.5 = habría disparado en tienda)
    sp = np.array([float(i["score_prod"]) if i["score_prod"] else np.nan for i in fp])
    print("\nReingesta disparada según score del sistema ACTUAL (umbral @90%):")
    for lo, hi, tag in [(0.5, 2.0, "≥0.50 (dispara en producción)"),
                        (0.4, 0.5, "0.40–0.50"), (0.0, 0.4, "<0.40 (no dispararía)")]:
        mask = (sp >= lo) & (sp < hi)
        if mask.sum():
            print(f"  {tag:32} {float((p_fp[mask] >= thr90).mean()):6.1%} (n={mask.sum()})")

    out = {"model": args.model,
           "auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
           "sweep": rows}
    json.dump(out, open(os.path.join(RUNS, args.model, "eval_test.json"), "w"), indent=1)
    np.savez(os.path.join(RUNS, args.model, "scores.npz"),
             p_h=p_h, p_n=p_n, p_no=p_no, p_fp=p_fp,
             fp_paths=np.array([i["npy"] for i in fp]),
             fp_score_prod=sp)
    print(f"\n→ guardado en runs/{args.model}/eval_test.json y scores.npz")


if __name__ == "__main__":
    main()

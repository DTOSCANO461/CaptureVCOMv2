#!/usr/bin/env python3
"""Cosecha v14: extrae tubos de entreno de las carpetas etiquetadas por el
cuestionario (video_data.json). evento=si -> hurto, evento=no -> normal.

Corre EN LA TIENDA (usa YOLO + config de /opt/capturevcomv2). Los clips viven en
/root/CAPTURE_VIDEOS/uploads/<HH-MM-SS__DD-MM-YYYY>/*.mp4.

  python3 harvest_v14.py <DD-MM-YYYY> <out_dir> [--solo si|no] [--tag FC]

Saca 1 tubo por clip (el track mas prominente en el centro temporal, donde cae
el evento). Escribe npy + manifest.csv. Los npy se traen a local para el indice.
"""
import sys, os, json, glob
import numpy as np
import cv2

sys.path.insert(0, "/opt/capturevcomv2")
import config as C
from ultralytics import YOLO

VD = "/root/CAPTURE_VIDEOS/video_data.json"
UP = "/root/CAPTURE_VIDEOS/uploads"


def make_tube(frames_bgr, boxes, w, h, n_frames=16, side_out=256):
    idx = np.linspace(0, len(frames_bgr) - 1, n_frames).round().astype(int)
    bs = np.array(boxes, dtype=float)
    cx = np.median((bs[:, 0] + bs[:, 2]) / 2); cy = np.median((bs[:, 1] + bs[:, 3]) / 2)
    side = max(float(np.median(np.maximum(bs[:, 2]-bs[:, 0], bs[:, 3]-bs[:, 1])))*1.9, 200)
    side = min(side, min(w, h))
    x1 = int(np.clip(cx-side/2, 0, w-side)); y1 = int(np.clip(cy-side/2, 0, h-side)); s = int(side)
    out = [cv2.resize(frames_bgr[i][y1:y1+s, x1:x1+s], (side_out, side_out),
           interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in idx]
    return np.ascontiguousarray(np.stack(out))


def extrae_tubo(det, clip):
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    w = int(cap.get(3)); h = int(cap.get(4))
    frames, tracks, fi = [], {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if C.SKIP_FRAMES > 1 and fi % C.SKIP_FRAMES:
            fi += 1; continue
        res = det.track(fr, classes=[0], conf=0.30, persist=True, verbose=False,
                        imgsz=960, tracker="bytetrack.yaml")[0]
        if res.boxes is not None and res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().tolist()):
                tracks.setdefault(tid, []).append((fi, tuple(b)))
        fi += 1
    cap.release()
    if not frames or not tracks:
        return None
    n = len(frames); center = n // 2
    # track mas "prominente" cerca del centro temporal: duracion x altura media
    best, best_sc = None, -1
    for tid, seq in tracks.items():
        near = [b for (f, b) in seq if abs(f - center) < n * 0.4]
        if len(near) < 2:
            continue
        sc = len(near) * float(np.mean([b[3] - b[1] for b in near]))
        if sc > best_sc:
            best_sc, best = sc, seq
    if best is None:
        return None
    win = [(f, b) for (f, b) in best if abs(f - center) <= fps * 2]  # +-2s del centro
    if len(win) < 4:
        win = best
    fr_idx = [f for (f, _) in win]
    return make_tube([frames[i] for i in fr_idx], [b for (_, b) in win], w, h)


def main():
    fecha = sys.argv[1]; out = sys.argv[2]
    solo = sys.argv[sys.argv.index("--solo") + 1] if "--solo" in sys.argv else None
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "x"
    # --excluir <substr>: salta clips cuyo nombre contenga <substr> (p.ej. cam9 en AM,
    # la cámara de cajas: genera FP por gente guardando la compra en bolsas).
    excluir = sys.argv[sys.argv.index("--excluir") + 1] if "--excluir" in sys.argv else None
    os.makedirs(out, exist_ok=True)
    d = json.load(open(VD))
    det = YOLO(C.YOLO_MODEL)
    man = open(os.path.join(out, "manifest.csv"), "w"); man.write("npy,clase,folder,clip\n")
    n_pos = n_neg = n_skip = 0
    for k, v in d.items():
        if fecha not in k:
            continue
        ev = (v.get("evento") or "").lower()
        if ev not in ("si", "no"):
            continue
        if solo and ev != solo:
            continue
        clase = "hurto" if ev == "si" else "normal"
        folder = os.path.join(UP, k)
        if not os.path.isdir(folder):
            continue
        for clip in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
            if excluir and excluir in os.path.basename(clip):
                n_skip += 1; continue
            try:
                tubo = extrae_tubo(det, clip)
            except Exception as e:
                print("ERR", clip, e); tubo = None
            if tubo is None:
                n_skip += 1; continue
            name = f"{tag}_{clase}_{k}__{os.path.basename(clip)}.npy"
            np.save(os.path.join(out, name), tubo.astype(np.uint8))
            man.write(f"{name},{clase},{k},{os.path.basename(clip)}\n")
            n_pos += clase == "hurto"; n_neg += clase == "normal"
    man.close()
    print(f"[{tag} {fecha}] tubos: {n_pos} hurto, {n_neg} normal (saltados {n_skip}) -> {out}")


if __name__ == "__main__":
    main()

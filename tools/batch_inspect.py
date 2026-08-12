import sys, os, glob
sys.path.insert(0, ".")
from prueba_historica import extrae_tubo
from ultralytics import YOLO
import cv2, numpy as np

det = YOLO("model/yolo11s.pt")
casos = [
    ("10-43-07__19-07-2026", "si", 0.0237),
    ("12-24-18__14-07-2026", "si", 0.0241),
    ("11-47-38__11-07-2026", "si", 0.0248),
    ("11-17-30__19-07-2026", "si", 0.0252),
    ("11-45-30__04-07-2026", "si", 0.0262),
    ("12-53-08__14-07-2026", "no", 0.0276),
    ("13-40-16__18-07-2026", "no", 0.0272),
    ("15-43-01__24-07-2026", "no", 0.0268),
]
filas = []
for carpeta, ev, score in casos:
    clips = sorted(glob.glob(f"/root/CAPTURE_VIDEOS/uploads/{carpeta}/*.mp4"))
    if not clips:
        print(carpeta, "SIN CLIPS"); continue
    t = extrae_tubo(clips[0], det)
    if t is None:
        print(carpeta, "SIN TUBO"); continue
    fila = np.hstack([t[0], t[6], t[11], t[15]])
    fila = cv2.cvtColor(fila, cv2.COLOR_RGB2BGR)
    etiq = f"{ev.upper()} score={score}  {carpeta}"
    cv2.putText(fila, etiq, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    filas.append(fila)

cv2.imwrite("/tmp/batch_inspect.png", np.vstack(filas))
print("ok", len(filas), "filas")

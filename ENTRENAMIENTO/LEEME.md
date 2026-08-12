# CaptureVCOMv2 — pesos + dataset + código de entrenamiento

Todo lo necesario para (a) correr el sistema completo con el modelo actual y
(b) reentrenar/reproducir ese modelo.

## Contenido

```
PARA_DAVID/
├── pesos/                  # PESOS del modelo en producción
│   ├── videomae_v6_last.pt # clasificador VideoMAE (el ft2 actual). md5 3f7f3a50…
│   ├── yolo11s.pt          # detector/tracker de personas. md5 9637097d…
│   └── MD5SUMS.txt         # para verificar integridad: `md5sum -c MD5SUMS.txt`
├── audit/                  # etiquetas del dataset
│   ├── labels.csv          # path, dataset, clase (hurto|normal), split, …
│   └── tubes_meta.csv      # path, npy (nombre de fichero del tubo), …
├── train/                  # código de entrenamiento (autónomo)
│   ├── train.py            # entrena VideoMAE o MViT
│   ├── eval_v14ft2.py      # evalúa el modelo actual
│   └── eval.py             # evaluación genérica
├── tubes_anon.tar          # 47 GB · ~15k tubos .npy (16,256,256,3) uint8 RGB
└── LEEME.md
```

## 1) Correr el sistema (repo CaptureVCOMv2, que ya tienes)

En la raíz del repo, crea `model/` y mete los pesos **con estos nombres exactos**
(es lo que apunta `config.py`: `CKPT` y `YOLO_MODEL`):

```
model/videomae_v6_last.pt
model/yolo11s.pt
```

El nombre "v6" es heredado; el contenido es el modelo actual (ft2). Copia
`config.example.py`→`config.py` y `vlm.env.example`→`vlm.env`, rellena tu API
key del VLM y las cámaras, y arranca.

## 2) Reentrenar / reproducir el modelo

⚠️ **Dataset anonimizado**: los tubos son metraje real de clientes, con la cara
(ojos/nariz/boca) **tapada** para poder compartirlos fuera de on-premise.
Entrenando con ellos sale un modelo **equivalente**, no idéntico bit a bit al de
producción (que se entrenó con caras sin tapar). La diferencia medida es mínima.

Pasos:
1. `mkdir -p tubes && tar -xf tubes_anon.tar -C tubes`  (extrae los ~15k .npy)
2. Coloca `audit/`, `train/` y `tubes/` bajo una misma raíz y ajusta la
   constante `ROOT` en `train/train.py` (línea ~22) a esa ruta. `train.py` carga
   etiquetas de `ROOT/audit/labels.csv` y tubos de `ROOT/tubes/<npy>`. La columna
   `path` de los CSV es solo clave de join entre ambos, no lee ficheros: no hace
   falta que esas rutas absolutas existan.
3. Entorno: Python 3.11, `torch` (CUDA), `transformers`, `numpy`,
   `opencv-python`, `scikit-learn`. GPU ~16 GB.
4. `python train/train.py --model videomae --epochs 15 --bs 6 --accum 5 --lr 1e-4`
   → checkpoints en `train/runs/videomae/`.

## Notas

- Modelo de producción: `videomae_v14_ft2` (fine-tune). `train.py` entrena el
  base; el ft2 exacto lleva además los añadidos del fine-tune (hurtos de Jose +
  FP de FC/AM) — se pueden pasar aparte si hacen falta.
- Zoom del tubo en producción: `ZOOM=0.6` (recorte central), ya aplicado en los
  tubos de este dataset.

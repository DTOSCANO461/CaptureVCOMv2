#!/usr/bin/env python3
"""genera_anotaciones_por_cabeza.py -- toma un anotaciones.csv CRUDO (sin
columna "label" -- el que generan ahora harvest_pose_test.py/harvest_muestra.py:
npy,split,marco,x0,y0,x1,y1,fps,dataset_tipo,clip_origen) y genera, A
POSTERIORI, un anotaciones.csv listo para entrenar (con columna "label",
mismo esquema que ya leen training/train.py y training/train_binario.py sin
tocarlos) para la cabeza que se pida -- binaria o multiclase.

Por que a posteriori: al cosechar (recortar tubos de los videos) no sabemos
todavia si el entrenamiento final va a ser binario o multiclase -- son dos
funciones de labeling DISTINTAS (labeling_strategy.py vs
labeling_strategy_binaria.py) que ni siquiera coinciden en que filas son
resolubles (ver mas abajo). Separar la resolucion del label de la cosecha
permite generar los dos esquemas a partir del MISMO conjunto de tubos ya
recortados (dataset_tipo+clip_origen alcanza para recalcular cualquiera de
los dos), sin volver a tocar video ni GPU.

--modo binario: usa labeling_strategy_binaria.aplica_reglas_etiquetado_binario()
tal cual (que a su vez usa labeling_strategy.aplica_reglas_etiquetado() para
las carpetas que SI tienen reglas multiclase). TODAS las filas son
resolubles -- incluidas manipulaciones_tiendas/reingesta_dia1_socrates/
reingesta_dia_platonv1 (siempre 0/falso ahi, ver ese modulo).

--modo multiclase: usa labeling_strategy.aplica_reglas_etiquetado() tal
cual, SIN modificarla. Las carpetas SIN reglas multiclase definidas ahi
(las 3 de arriba) NO tienen un label multiclase valido -- esas filas se
DESCARTAN de la salida (se cuentan y se avisan), no se inventa un valor.
Esto es justamente el caso que motiva este script: esas carpetas solo
sirven para binario ("sabemos que no es la clase 4, pero no sabemos cual
de las otras 69 es"), no para multiclase.

Uso:
  python3 training/genera_anotaciones_por_cabeza.py <raw.csv> <salida.csv> --modo binario
  python3 training/genera_anotaciones_por_cabeza.py <raw.csv> <salida.csv> --modo multiclase
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labeling_strategy as ls
import labeling_strategy_binaria as lsb


def _results_base(dataset_tipo, clip_origen):
    """Reconstruye el dict que esperan aplica_reglas_etiquetado()/
    aplica_reglas_etiquetado_binario() a partir de lo que ya esta en la
    tabla cruda -- total_frames no hace falta (solo lo usan un par de reglas
    para calcular frame_inds, un campo que el harvester actual no aplica al
    recorte final, ver harvest_pose_test.py)."""
    return {
        "frame_dir": clip_origen,
        "dataset_tipo": dataset_tipo,
        "label": ls.extraer_label_inicial(clip_origen),
        "total_frames": 0,
    }


def resuelve_binario(dataset_tipo, clip_origen):
    return lsb.aplica_reglas_etiquetado_binario(_results_base(dataset_tipo, clip_origen))["label"]


def resuelve_multiclase(dataset_tipo, clip_origen):
    return ls.aplica_reglas_etiquetado(_results_base(dataset_tipo, clip_origen))["label"]  # puede lanzar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_csv")
    ap.add_argument("salida_csv")
    ap.add_argument("--modo", choices=["binario", "multiclase"], required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.raw_csv)))
    if not rows:
        sys.exit(f"{args.raw_csv}: sin filas")
    fieldnames = list(rows[0].keys())
    if "label" in fieldnames:
        sys.exit(f"{args.raw_csv} ya tiene columna 'label' -- este script espera un anotaciones.csv CRUDO "
                  f"(el que generan harvest_pose_test.py/harvest_muestra.py ahora, sin label resuelto)")
    fieldnames_out = fieldnames + ["label"]

    resuelve = resuelve_binario if args.modo == "binario" else resuelve_multiclase

    n_ok = 0
    descartadas_por_carpeta = {}
    with open(args.salida_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_out)
        w.writeheader()
        for r in rows:
            try:
                label = resuelve(r["dataset_tipo"], r["clip_origen"])
            except (ls.DatasetTipoDesconocido, ls.LabelNoResuelto):
                descartadas_por_carpeta[r["dataset_tipo"]] = descartadas_por_carpeta.get(r["dataset_tipo"], 0) + 1
                continue
            r["label"] = label
            w.writerow(r)
            n_ok += 1

    n_descartadas = sum(descartadas_por_carpeta.values())
    print(f"{args.salida_csv} (modo={args.modo}): {n_ok} filas resueltas, {n_descartadas} descartadas")
    if descartadas_por_carpeta:
        for carpeta, n in sorted(descartadas_por_carpeta.items()):
            print(f"  descartadas de {carpeta!r}: {n} (sin reglas de labeling {args.modo} para esa carpeta)")


if __name__ == "__main__":
    main()

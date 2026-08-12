#!/usr/bin/env python3
"""anotaciones_a_binario.py -- convierte un anotaciones.csv YA GENERADO
(labels multiclase 0-69, columna dataset_tipo) a un anotaciones.csv BINARIO
(hurto=1 / normal=0), via labeling_strategy_binaria.label_binario_desde_multiclase.

No re-procesa videos ni recorta tubos de nuevo: los .npy ya existen y no
dependen del esquema de labels (el marco/crop quedo congelado en la
cosecha). Esto solo REESCRIBE la columna "label" en una copia NUEVA del CSV
-- nunca toca el anotaciones.csv de origen (dataset multiclase, ver
training/train.py).

Uso:
  python3 training/anotaciones_a_binario.py <anotaciones_multiclase.csv> <anotaciones_binaria.csv>
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labeling_strategy_binaria as lsb


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(src)))
    fieldnames = list(rows[0].keys())

    n_pos = 0
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            label_bin = lsb.label_binario_desde_multiclase(r["dataset_tipo"], int(r["label"]))
            r["label"] = label_bin
            n_pos += label_bin
            w.writerow(r)

    print(f"{dst}: {len(rows)} filas, {n_pos} positivos / {len(rows) - n_pos} negativos "
          f"({n_pos/len(rows)*100:.1f}% hurto)")


if __name__ == "__main__":
    main()

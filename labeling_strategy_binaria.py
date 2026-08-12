#!/usr/bin/env python3
"""labeling_strategy_binaria.py

Estrategia de etiquetado BINARIA (hurto=1 / normal=0), construida ENCIMA de
labeling_strategy.py -- no la reemplaza ni la modifica, la importa tal cual.
Reglas dadas por el usuario:

  - dataset_tipo (carpeta padre) que CONTENGA "reingesta" en el nombre ->
    FALSO (0), directo, sin pasar por labeling_strategy.aplica_reglas_etiquetado()
    (esas carpetas -- reingesta_dia1_socrates, reingesta_dia_platonv1 -- no
    tienen reglas multiclase ahi, esa funcion se detendria con
    DatasetTipoDesconocido).
  - dataset_tipo que CONTENGA "manipulacion" -> FALSO (0), mismo motivo
    (manipulaciones_tiendas).
  - Para el resto de carpetas: se resuelve el label multiclase con
    labeling_strategy.aplica_reglas_etiquetado() SIN TOCARLA, y despues:
    label numerico == 4 -> VERDADERO (1, hurto); cualquier otro valor,
    incluido 48 (negativo/no_obvio) -> FALSO (0).

De paso esto "activa" para el dataset binario las carpetas reingesta_*/
manipulaciones_tiendas que antes quedaban EXCLUIDAS del sorteo en
harvest_muestra.py (CARPETAS_EXCLUIDAS) por no tener ninguna regla de
etiquetado -- aca ya tienen una, aunque trivial (siempre falso). No se
incluyeron tubos de esas carpetas en los datasets ya cosechados (dataset_500/
dataset_5000): para usarlas hace falta correr una cosecha nueva sobre ellas
(fuera del alcance de este archivo).
"""
import labeling_strategy as ls

LABEL_POSITIVO_MULTICLASE = 4


def label_binario_desde_multiclase(dataset_tipo, label_multiclase):
    """Aplica el mapeo binario a un label QUE YA ESTA RESUELTO en la
    taxonomia multiclase (0-69) -- para remapear un anotaciones.csv
    existente sin volver a tocar los videos/tubos (ver
    training/anotaciones_a_binario.py)."""
    if "reingesta" in dataset_tipo or "manipulacion" in dataset_tipo:
        return 0
    return 1 if label_multiclase == LABEL_POSITIVO_MULTICLASE else 0


def aplica_reglas_etiquetado_binario(results):
    """Mismo contrato que ls.aplica_reglas_etiquetado: `results` es un dict
    con 'frame_dir', 'dataset_tipo', 'label' (ya inicializado via
    ls.extraer_label_inicial), 'total_frames'. Devuelve results con
    results['label'] YA REEMPLAZADO por 0/1 (no el codigo multiclase crudo).
    Para usar en una cosecha nueva (harvest_*.py) sobre video/frame_dir
    reales, incluidas las carpetas reingesta_*/manipulaciones_tiendas."""
    tipo = results["dataset_tipo"]

    if "reingesta" in tipo or "manipulacion" in tipo:
        results["label"] = 0
        return results

    resultado = ls.aplica_reglas_etiquetado(results)  # resuelve multiclase, valida, puede lanzar
    resultado["label"] = label_binario_desde_multiclase(tipo, resultado["label"])
    return resultado

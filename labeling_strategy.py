#!/usr/bin/env python3
"""labeling_strategy.py

Estrategia de etiquetado: asigna un `label` (codigo numerico de una
taxonomia multi-clase heredada de otro proyecto -- PoseC3D/pyskl, ver
ejemplo_poses.py) y, para algunas carpetas, una ventana temporal
(`frame_inds`) a cada video, segun la carpeta padre inmediata
(`dataset_tipo`, ej. "control_negativo") y el nombre del archivo
(`frame_dir`). Tambien resuelve los FPS reales por carpeta -- la metadata
del contenedor NO es confiable (ya confirmado para control_negativo /
doblar_sobreponer: dicen 11fps, son 8fps de verdad).

Las reglas por dataset_tipo estan TRANSCRITAS TAL CUAL las dio el usuario
-- no se reinterpretan ni se simplifican, incluidos los comentarios
originales (traducidos/preservados donde tenian contexto util).

IMPORTANTE -- lo que este modulo NO hace: no convierte el `label` numerico
a "hurto"/"normal" (el binario que usa training/train.py hoy). Esa
conversion todavia no esta definida -- son ~12 codigos distintos, y "48"
aparece tanto en control_negativo (la clase negativa, por nombre) como en
ainhoa/no_obvio, asi que no se puede asumir sin mas info. anotaciones.csv
guarda el label numerico crudo; falta un paso aparte, ya con el mapeo
completo, para la columna binaria.

Validacion: si `dataset_tipo` no tiene reglas conocidas, si no hay FPS
conocido para esa carpeta, o si al final de aplicar las reglas el label
sigue siendo el default sin resolver, esto ADVIERTE (imprime) y DETIENE la
ejecucion (raise) -- nunca sigue en silencio con un video mal etiquetado.
"""
import random

import numpy as np

# Si el nombre no tiene "accion___N", no hay label real codificado en el
# nombre del archivo -- placeholder explicito (dado por el usuario) para que
# las reglas por carpeta lo puedan detectar y decidir que hacer.
DEFAULT_ACCION = 69

# FPS real por carpeta (dataset_tipo) -- la metadata del contenedor
# (ffprobe / cv2.CAP_PROP_FPS) puede decir otra cosa; NO ES CONFIABLE.
FPS_POR_CARPETA = {
    "ainhoa": 25,
    "control_negativo": 8,
    "dia": 25,
    "dia_guardar": 25,
    "doblar_sobreponer": 8,
    "manipulaciones_tiendas": 9,
    "reingesta_dia1_socrates": 8,
    # El usuario listo "reingesta_dia1_socrates" dos veces en la tabla de FPS,
    # con conteos de video distintos (675 y 3505) -- asumo que la segunda
    # entrada es en realidad reingesta_dia_platonv1 (existe ese .zip en
    # videos_for_train/, y no tenia FPS propio listado). Ambas dan 8fps de
    # todos modos, asi que no cambia el valor -- pero CONFIRMAR el nombre.
    "reingesta_dia_platonv1": 8,
    "todo_showroom": 25,
}


class DatasetTipoDesconocido(RuntimeError):
    """dataset_tipo sin reglas de etiquetado y/o sin FPS conocido."""


class LabelNoResuelto(RuntimeError):
    """Se aplicaron las reglas de la carpeta pero el label quedo en el
    default sin resolver -- ningun caso contemplado aplico."""


def extraer_label_inicial(nombre_archivo):
    """Mismo criterio que get_label_from_name() del pipeline de referencia:
    si el nombre codifica la accion ("...accion___N..."), esa N es el label
    real. Si no, DEFAULT_ACCION -- son clips reingestados/sin etiquetar por
    nombre, y las reglas por carpeta de abajo deciden que hacer con eso."""
    if "accion" in nombre_archivo:
        return int(nombre_archivo.split("___")[1])
    return DEFAULT_ACCION


def fps_de_carpeta(dataset_tipo):
    if dataset_tipo not in FPS_POR_CARPETA:
        raise DatasetTipoDesconocido(
            f"No hay FPS conocido para dataset_tipo={dataset_tipo!r}. "
            f"Carpetas conocidas: {sorted(FPS_POR_CARPETA)}. "
            f"AGREGAR la entrada en FPS_POR_CARPETA antes de seguir -- "
            f"no se puede adivinar el FPS real (la metadata del contenedor no es confiable)."
        )
    return FPS_POR_CARPETA[dataset_tipo]


def aplica_reglas_etiquetado(results):
    """results: dict con al menos 'frame_dir', 'dataset_tipo', 'label' (ya
    inicializado via extraer_label_inicial(...)), 'total_frames'.

    Modifica results['label'] (y a veces results['frame_inds']) IN PLACE
    segun dataset_tipo, y tambien devuelve results. Reglas transcritas tal
    cual las dio el usuario -- ver el docstring del modulo para lo que NO
    cubren (mapeo a hurto/normal).
    """
    tipo = results["dataset_tipo"]

    if tipo == "ainhoa":
        # los 4 estan catalogados obvio y no obvio
        if "no_obvio" in results["frame_dir"]:  # para ainhoa cuyos videos no tienen no_obvio en el nombre pero tienen una llave
            results["label"] = 48
        # si es reingesta (DEFAULT_ACCION), va a bucket 48
        if results["label"] == DEFAULT_ACCION:
            results["label"] = 48
        # no tiene la accion urgar(7), topar(9 -> agarrar negativo)
        # slicing, validado ahora de forma general para max 5.5 seg

    elif tipo == "control_negativo":
        # cualquier tipo de slicing sirve
        # videos fantasma de objetos, pilas -- sirve para textil o multiobjeto
        results["label"] = 48
        # slicing, validado ahora de forma general para max 5.5 seg
        cantidad_de_frames = results["total_frames"]  # estamos reentrenando con el output del modelo
        inicio_frame = random.randint(int(0.1 * cantidad_de_frames), int(0.4 * cantidad_de_frames))
        numeros = random.sample(range(inicio_frame, inicio_frame + 32), 32)
        results["frame_inds"] = np.array(sorted(numeros)).astype(int)

    elif tipo == "dia":
        # se introducen acciones con carrito
        # existe un error, estos videos fueron etiquetados mal: agarrar es 1, no 6
        if results["label"] == 6:
            results["label"] = 1
        # la accion "urgar" proponemos catalogarla como unico antagonista a 48
        if results["label"] == 7:
            results["label"] = 48
        # la accion "topar" es practicamente agarrar-negativo (ainhoa se queda topando el objeto a veces, ojo)
        if results["label"] == 9:
            results["label"] = 11
        # 42 son imagenes de gente caminando, pero jose camina con carrito tambien, asi que es mix
        if results["label"] == 42:
            results["label"] = 47
        # relajan la posicion de acercamiento a producto pero sin acercarlo -- se entiende como agarrar-negativo
        if results["label"] == 24:
            results["label"] = 11
        # girar mientras se sostiene (a veces tambien hay carrito involucrado)
        if results["label"] == 46:
            results["label"] = 46  # no se agrupa con "sostener" para que sostener/sostener-negativo permitan entender las manos como clave
        # topar, dejar-negativo, agarrar-negativo, bajar-mano(24): es lo mismo? (pregunta abierta del original)
        # slicing, validado ahora de forma general para max 5.5 seg

    elif tipo == "doblar_sobreponer":
        # todo lo de "doblar" en realidad es manipulacion
        results["label"] = 48
        # la accion de manipulacion parece ocurrir mas al final, por eso el inicio llega hasta 0.5
        cantidad_de_frames = results["total_frames"]  # estamos reentrenando con el output del modelo
        inicio_frame = random.randint(int(0.25 * cantidad_de_frames), int(0.5 * cantidad_de_frames))
        numeros = random.sample(range(inicio_frame, inicio_frame + 32), 32)
        results["frame_inds"] = np.array(sorted(numeros)).astype(int)

    elif tipo == "dia_guardar":
        if "no_obvio" in results["frame_dir"]:  # para ainhoa cuyos videos no tienen no_obvio en el nombre pero tienen una llave
            results["label"] = 48
        # slicing, validado ahora de forma general para max 5.5 seg

    elif tipo == "todo_showroom":  # videos de robos obvios y no obvios etiquetados
        if "no_obvio" in results["frame_dir"]:
            results["label"] = 48
        else:
            results["label"] = 4

    else:
        raise DatasetTipoDesconocido(
            f"dataset_tipo={tipo!r} (clip {results.get('frame_dir')!r}) no tiene reglas de "
            f"etiquetado definidas en aplica_reglas_etiquetado(). ADVERTENCIA: se detiene el "
            f"procesamiento -- agregar la regla para esta carpeta en labeling_strategy.py antes de continuar."
        )

    if results["label"] == DEFAULT_ACCION:
        raise LabelNoResuelto(
            f"clip {results.get('frame_dir')!r} (dataset_tipo={tipo!r}): ninguna regla resolvio "
            f"un label real, quedo en el default sin resolver ({DEFAULT_ACCION}). ADVERTENCIA: se "
            f"detiene el procesamiento -- revisar el nombre del archivo o agregar la regla que falta."
        )

    return results

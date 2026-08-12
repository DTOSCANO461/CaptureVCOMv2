#!/usr/bin/env python3
"""Envío directo a la app SIN VLM.

Con el VLM apagado nada llegaba a la app. Este daemon vigila ALERTAS_DIR y, por
cada alerta del detector con score >= APP_THR, recorta el contexto y la entrega
al pc-agent (reutiliza envia_a_app del verificador). Umbral alto porque no hay
VLM que filtre: solo las alertas muy confiadas van a la app; el resto se quedan
en alertas/ para cosecha de FP. v13 tiene FP bajos, así que casi todo lo que
supere el umbral será hurto real.
"""
import glob
import os
import re
import time

import config as C
# reutiliza el mecanismo de entrega ya probado (recorte de contexto + IDCAM + pc-agent)
from verificador_vlm import envia_a_app

APP_THR = float(os.environ.get("APP_THR", "0.85"))
REGISTRO = os.path.join(C.OUT_DIR, "enviados_directo.txt")
INTERVALO_S = 5


def p_de_nombre(nombre):
    m = re.search(r"_p(\d+(?:\.\d+)?)\.mp4$", nombre)
    return float(m.group(1)) if m else None


def main():
    enviados = set()
    if os.path.exists(REGISTRO):
        enviados = set(open(REGISTRO).read().split())
    print("[envia_directo] " + f"arrancado, umbral app={APP_THR} (sin VLM, flush=True)")
    while True:
        try:
            for path in sorted(glob.glob(os.path.join(C.ALERTAS_DIR, "*.mp4"))):
                nombre = os.path.basename(path)
                if nombre in enviados:
                    continue
                enviados.add(nombre)
                with open(REGISTRO, "a") as f:
                    f.write(nombre + "\n")
                p = p_de_nombre(nombre)
                if p is not None and p >= APP_THR:
                    print("[envia_directo] " + f"{nombre} p={p:.2f} -> APP", flush=True)
                    try:
                        envia_a_app(nombre, path)
                    except Exception as e:
                        print("[envia_directo] " + f"error enviando {nombre}: {e}", flush=True)
        except Exception as e:
            print("[envia_directo] " + f"error bucle: {e}", flush=True)
        time.sleep(INTERVALO_S)


if __name__ == "__main__":
    main()

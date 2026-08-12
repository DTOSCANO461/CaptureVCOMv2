"""Configuración de CaptureVCOMv2 — PLANTILLA.

Copia este archivo a config.py y adapta cada sección a la tienda.
config.py NUNCA se sube al repo (contiene credenciales RTSP): está en .gitignore.

Las tres capas de personalización por tienda viven aquí:
  1. Detector  — cámaras, umbrales (global y por cámara), dinámica de emisión.
  2. VLM       — zonas por cámara, descripción de la tienda y del personal.
  3. Horario   — fuera del horario los hilos de cámara duermen (0 GPU).
"""
import os

# ── 1) Cámaras ──────────────────────────────────────────────────────────────
# Cualquier fuente RTSP vale (DVR/NVR, cualquier marca y resolución): el
# pipeline trabaja sobre recortes de persona normalizados, no sobre el plano.
# NO incluyas cámaras de cajas/cobro: el gesto de ensacar compra legítima es
# indistinguible de un hurto y solo genera falsos positivos (lección de AM).
CAMARAS = [
    ("cam0", "rtsp://USUARIO:PASSWORD@IP_DVR:554/RUTA_CANAL_0"),
    ("cam1", "rtsp://USUARIO:PASSWORD@IP_DVR:554/RUTA_CANAL_1"),
    # ...
]

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE = "/opt/capturevcomv2"
CKPT = f"{BASE}/model/videomae_v6_last.pt"   # checkpoint del clasificador en producción
YOLO_MODEL = f"{BASE}/model/yolo11s.pt"      # detector/tracker de personas
OUT_DIR = f"{BASE}/data"
RAW_DIR = f"{OUT_DIR}/raw"          # grabación continua (para recortar contexto)
ALERTAS_DIR = f"{OUT_DIR}/alertas"  # clips que dispararon el modelo
LOG_DIR = f"{OUT_DIR}/logs"

# ── 2) Detector: umbrales y dinámica de emisión ────────────────────────────
UMBRAL = float(os.environ.get("CVC_THR", "0.5"))  # legacy: solo informativo (banner
                                                  # y scripts offline). El gate real
                                                  # de emisión es EMIT_UMBRAL.

SEGMENTO_S = 5           # duración de cada segmento raw. Corto (5s) = el VLM
                         # puede leer contexto sin esperar; latencia total ~30-60s.
RETENCION_RAW_H = 1      # horas de raw que se conservan. ¡Vigilar disco!
                         # (10 cámaras 1080p ≈ 12 GB/h; 72h llenó un disco de 457GB.)
COOLDOWN_S = 8.0         # 8s, NO 30: con 30 los hurtos en ráfaga (varios seguidos
                         # del mismo track) se perdían — diagnóstico Jose 04-08.
MIN_TRACK_S = 2.0        # ignora tracks más cortos (gente pasando)
WINDOW_S = 4.0           # ventana que puntúa el modelo
STRIDE_S = 0.7           # puntuación densa: no se pierde el pico del gesto
SCORE_EVERY_S = 1.5      # cada track se re-puntúa cada ~1.5s
EMIT_COOLDOWN_S = 8      # mínimo entre clips del mismo track (ver COOLDOWN_S)
REARM_QUIET_S = 4.0      # calma necesaria para contar un clip como evento NUEVO
EMIT_UPGRADE = 0.25      # un salto de score rompe el cooldown (pico del hurto)
EMIT_UMBRAL = 0.45       # ← GATE REAL: solo lo que supere esto se emite/va al VLM
EMIT_VALLE = 0.15        # si el score cae bajo esto, el próximo pico es hurto nuevo

# Umbral de emisión POR CÁMARA (sobreescribe EMIT_UMBRAL). Coste computacional
# cero: el modelo puntúa todo igual, esto solo decide si se emite. Úsalo para
# cámaras problemáticas (pantallas publicitarias con vídeo, zonas de personal).
# Ejemplo real (AM): la cámara de probadores con pantalla concentraba el 95%
# de los FP de la tienda → 0.45 y se acabó.
EMIT_UMBRAL_POR_CAM = {}          # p. ej. {"cam6": 0.45}

SKIP_FRAMES = 2          # procesar 1 de cada N frames
ZOOM = 0.6               # recorte central del tubo — DEBE coincidir con el
                         # zoom usado al entrenar el checkpoint

# ── 3) Horario de tienda (día ISO 1=lunes … 7=domingo → (apertura, cierre)) ─
HORARIO = {
    1: (9, 22), 2: (9, 22), 3: (9, 22), 4: (9, 22), 5: (9, 22), 6: (9, 22),
    7: (10, 15),
}

# ── 4) VLM (segunda opinión) — contexto en texto, sin reentrenar nada ──────
# Qué ve cada cámara. El VLM razona distinto en "pasillo de licores" que en
# "entrada de probadores". Confirmar con capturas reales de cada canal.
ZONAS_CAMARAS = {
    "cam0": "descripción de la zona de la cámara 0",
    "cam1": "descripción de la zona de la cámara 1",
}

TIENDA_DESC = "descripción corta del tipo de tienda (súper, moda, bazar...)"
ARTICULO = "producto"    # "producto" para retail genérico, "prenda" para moda

# Personal de tienda: uniformes y comportamiento típico. Con esto el VLM
# descarta reponedores/empleados en vez de marcarlos como hurto probable.
# Ejemplo real (FC): "reponedores con polo azul marino liso, y reponedores con
# camisa clara de cuadros; manejan carros de reposición, transpaletas o cajas".
EMPLEADOS_DESC = ""

# Reglas del prompt del VLM (versión ganadora de la comparativa de 10 variantes,
# 31/07-01/08/2026). Solo tocar con datos: cambios de prompt se validan contra
# el benchmark antes de desplegar.
PROMPT_REGLAS = """REGLA PRINCIPAL: ver un producto de la tienda ENTRANDO en una bolsa, mochila o \
bolso propio, o bajo la ropa puesta, o el arranque de una alarma/etiqueta, es \
el indicador central de hurto y basta por si solo para 'hurto_probable', \
aunque la persona actue con total naturalidad. No necesitas ver nerviosismo \
ni vigilancia. Pero la insercion debe VERSE: llevar una bolsa propia cerca de \
productos, sin verse el producto entrar, no basta.

REGLA DE DESAPARICION: si un producto que la persona estaba sujetando o \
tocando deja de ser visible en pantalla (no esta en su mano, no esta sobre el \
mostrador/estanteria, no se ve donde fue) Y la persona lleva consigo una \
bolsa, mochila o bolso propio ABIERTO en ese mismo momento, trata esto como \
'sospechoso' aunque no se vea el gesto exacto de meterlo dentro. Si NO hay \
ninguna bolsa/mochila/bolso propio visible en la escena, esta regla NO \
aplica: el producto puede estar en su carro-cesta, en su otra mano fuera de \
plano o haber sido devuelto; no marques 'sospechoso' solo por la \
desaparicion en ese caso.

Normal: andar por la tienda, mirar/tocar productos y devolverlos, llevar \
productos EN LA MANO o el brazo a la vista (ir a caja), usar el movil, \
llevar bolsas de otras compras sin meter productos nuevos en ellas, empujar \
un carro o cesta de la tienda y colocar productos dentro de el. El carro o \
cesta de la tienda NO cuenta como "bolsa, mochila o bolso propio" a efectos \
de las reglas anteriores. Que parte del cuerpo quede oculta tras una \
estanteria o mueble NO es por si solo indicio de nada.
Empleado: reponer producto en estanterias, hacer inventario o limpieza, \
atender el mostrador.{empleados}"""

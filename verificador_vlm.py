#!/usr/bin/env python3
"""Daemon de segunda opinion por VLM para CaptureVCOMv2.

Vigila ALERTAS_DIR. Por cada clip nuevo (>=UMBRAL ya garantizado por el
detector): extrae 6 frames, los manda EN BASE64 a qwen3-vl-plus a traves del
proxy /vlm/proxy del central-server (que gestiona el pool de claves DashScope;
este PC no tiene ni necesita la API key de Alibaba) y clasifica en
hurto_probable / sospechoso / normal / empleado.

- hurto_probable,
  sospechoso      -> recorta +-30s de la grabacion cruda (contexto real, no
                     los 4s del clip) y lo deja en /tmp/CAPTURE/OUTPUTS/,
                     donde el watchdog de pc-agent lo recoge y lo sube a la
                     app con notificacion, igual que hacia CAPTURE.
- todos           -> hardlink del clip en data/reentreno/<veredicto>/ y
                     veredicto en data/vlm/veredictos.jsonl (dataset v6:
                     prevlm = todo alertas/, postvlm = subcarpetas).

La exclusion de personal es por tienda via config: si config.EMPLEADOS_DESC
describe el uniforme/aspecto del personal, el prompt lo usa y esos casos
salen como 'empleado' (no suben a la app). Sin tocar el modelo.

NOTA piloto: sin pixelado de caras (fase de pruebas, decision explicita).
"""
import base64
import glob
import json
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# El PC ya no llama a DashScope/Alibaba directamente: manda los mismos
# `messages` (multi-imagen) al proxy VLM del central-server (vlm_proxy/,
# pool de claves DashScope cifradas en BD con cola y reintento entre claves;
# ver central-server/vlm_proxy/README.txt). Alibaba solo necesita whitelistear
# la IP de ese servidor, y este PC no guarda ninguna API key de Alibaba.
# Se autentica con el mismo PC_API_KEY que ya usa pc-agent para /heartbeat.
PC_API_KEY = os.environ.get("PC_API_KEY", "").strip()
CENTRAL_SERVER_URL = os.environ.get("CENTRAL_SERVER_URL", "https://api.sb-tec.com").rstrip("/")
API_URL = f"{CENTRAL_SERVER_URL}/vlm/analyze_frames"
MODELO = "qwen3-vl-plus"

VLM_DIR = os.path.join(C.OUT_DIR, "vlm")
REENTRENO = os.path.join(C.OUT_DIR, "reentreno")
ESTADO = os.path.join(VLM_DIR, "estado.json")
VEREDICTOS = os.path.join(VLM_DIR, "veredictos.jsonl")
OUTPUTS_PCAGENT = "/tmp/CAPTURE/OUTPUTS"
N_FRAMES = 6
CONTEXTO_S = 7.5         # segundos de raw antes/despues para el video de la app (15s totales)
CONTEXTO_VLM_S = 15      # segundos antes/despues para los frames del VLM
N_CTX = 5                # fotos de contexto por lado (a intervalos regulares hasta CONTEXTO_VLM_S)
MAX_ESPERA_S = 45        # max espera al contexto posterior; luego se procesa igual
INTERVALO_S = 5          # cada cuanto se revisa alertas/
VLM_WORKERS = int(os.environ.get("VLM_WORKERS", "6"))  # clips en paralelo (aprovecha la pool de keys del central)
_lock = threading.Lock()  # protege estado + escritura de veredictos entre hilos
ASENTADO_S = 3           # no tocar clips modificados hace menos de esto

# recorta_contexto() lee segmentos RAW (SEGMENTO_S, tipicamente 60s), no las
# fotos rapidas de contexto_disponible(). Si la alerta cae justo despues de
# que arranque un segmento nuevo, ese segmento puede tardar hasta SEGMENTO_S
# en cerrarse. Antes esto se intentaba una sola vez y, si fallaba, el clip
# quedaba marcado como "hecho" para siempre con el video corto de fallback
# (visto en produccion: "sin raw disponible, enviado el clip corto" con el
# segmento cerrandose pocos segundos despues). Solo afecta a hurto_probable/
# sospechoso (baja frecuencia), asi que un reintento acotado aqui es barato.
REINTENTOS_CONTEXTO_APP = 14
ESPERA_REINTENTO_CONTEXTO_APP_S = 5.0
# ventana completa = 2*CONTEXTO_S; con menos que esto se considera clip
# incompleto y se reintenta. Tolerancia 3.0s: los bordes de segmento comen
# hasta ~2.6s reales — con 1.5s el objetivo era inalcanzable y CADA clip
# agotaba los 14 reintentos (70s dormido en un worker → cola de minutos
# en rafagas, visto en produccion 07-08-2026)
MIN_CONTEXTO_APP_S = 2 * CONTEXTO_S - 3.0

ZONAS = getattr(C, "ZONAS_CAMARAS", {})
EMPLEADOS_DESC = getattr(C, "EMPLEADOS_DESC", "")

# Contexto de tienda para el prompt, adaptable por config sin tocar el codigo:
# TIENDA_DESC  - "una tienda de ropa en Espana" (defecto)
# ARTICULO     - palabra para el genero: "prenda" (defecto), "producto"...
# PROMPT_REGLAS- bloque completo REGLA PRINCIPAL/Normal/Empleado/Sospechoso;
#                debe incluir el placeholder {empleados}. Si no se define se
#                usa el bloque de tienda de ropa validado en Alvaro Moreno.
TIENDA_DESC = getattr(C, "TIENDA_DESC", "una tienda de ropa en Espana")
ARTICULO = getattr(C, "ARTICULO", "prenda")

REGLAS_DEFECTO = """REGLA PRINCIPAL: ver una prenda de la tienda ENTRANDO en una bolsa, mochila o \
bolso propio, o bajo la ropa puesta, o el arranque de una alarma/etiqueta, es \
el indicador central de hurto y basta por si solo para 'hurto_probable', \
aunque la persona actue con total naturalidad. No necesitas ver nerviosismo \
ni vigilancia. Pero la insercion debe VERSE: llevar una bolsa propia cerca de \
prendas, sin verse la prenda entrar, no basta.

Normal: andar por la tienda, mirar/tocar prendas y devolverlas, llevar \
prendas EN LA MANO o el brazo a la vista (ir al probador o a caja), usar el \
movil, empujar carrito de bebe, llevar bolsas de otras compras sin meter \
prendas en ellas. Que parte del cuerpo quede oculta tras un perchero o mueble \
NO es por si solo indicio de nada.
Empleado: reponer o doblar ropa, llevar pilas de prendas dobladas a la vista, \
atender el mostrador, colocar alarmas.{empleados}
Sospechoso: manipulacion anomala sin verse la insercion con claridad (p.ej. \
una prenda desaparece de plano junto a una bolsa abierta, o agacharse con \
genero tras un mueble y reaparecer sin el)."""

PROMPT_REGLAS = getattr(C, "PROMPT_REGLAS", REGLAS_DEFECTO)


def log(msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [vlm] {msg}"
    print(linea, flush=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    with open(os.path.join(C.LOG_DIR, "vlm.log"), "a") as f:
        f.write(linea + "\n")


def prompt_sistema(con_contexto):
    empleados = ""
    if EMPLEADOS_DESC:
        empleados = (f"\nPersonal de tienda: {EMPLEADOS_DESC}. La vestimenta "
                     "sola no basta (un cliente puede vestir igual): responde "
                     "'empleado' solo si ademas la conducta es de trabajo de "
                     "tienda.")
    if con_contexto:
        secuencia = ("Veras 12 fotogramas cronologicos: los 3 primeros son "
                     "CONTEXTO PREVIO (~15s antes), los 6 centrales son la "
                     "ACCION detectada (~4s) y los 3 ultimos son CONTEXTO "
                     f"POSTERIOR (~15s despues). Usa el contexto: si la persona "
                     f"devuelve la {ARTICULO} a su sitio o va hacia caja, no es hurto.")
    else:
        secuencia = (f"Veras {N_FRAMES} fotogramas cronologicos (unos 4 "
                     "segundos de video) centrados en la persona marcada.")
    reglas = PROMPT_REGLAS
    if "{empleados}" in reglas:
        reglas = reglas.replace("{empleados}", empleados)
    elif empleados:
        reglas = reglas + empleados
    return f"""Eres un analista experto en prevencion de perdidas (loss prevention) \
revisando camaras de seguridad de {TIENDA_DESC}. Un detector \
automatico ha marcado a una persona como posible hurto y tu das la segunda opinion.

{secuencia}

{reglas}

Responde SOLO con un JSON valido, sin markdown, con este formato exacto:
{{"veredicto": "hurto_probable"|"sospechoso"|"normal"|"empleado", "confianza": "alta"|"media"|"baja", "que_se_ve": "...", "razon": "..."}}"""


def frames_de_clip(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 2:
        cap.release()
        return []
    idxs = set(np.linspace(0, n - 1, N_FRAMES).round().astype(int).tolist())
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idxs:
            out.append(fr)
        i += 1
    cap.release()
    return out


def segmentos_cerrados(cam):
    """Segmentos raw ya legibles: todos menos el que aun se esta escribiendo
    (el mas reciente, salvo que lleve un rato sin tocarse)."""
    ficheros = sorted(glob.glob(os.path.join(C.RAW_DIR, cam, f"{cam}_*.mp4")))
    if not ficheros:
        return []
    ahora = time.time()
    out = []
    for i, p in enumerate(ficheros):
        es_ultimo = i == len(ficheros) - 1
        if es_ultimo and ahora - os.path.getmtime(p) < 5:
            continue
        m = re.search(r"_(\d{8})_(\d{6})\.mp4$", p)
        if m:
            out.append((datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), p))
    return out


def frame_en(segs, t):
    """Extrae el frame mas cercano al instante t de los segmentos cerrados."""
    for ini, p in segs:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fin = ini + timedelta(seconds=n / fps)
        if not (ini <= t <= fin):
            cap.release()
            continue
        idx = int((t - ini).total_seconds() * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, n - 1)))
        ok, fr = cap.read()
        cap.release()
        if ok:
            return fr
    return None


def ctx_jpgs(cam):
    """[(epoch, path)] de las fotos de contexto que escribe camara_worker."""
    d = os.path.join(C.OUT_DIR, "ctx", cam)
    out = []
    for f in glob.glob(os.path.join(d, "*.jpg")):
        try:
            out.append((int(os.path.basename(f)[:-4]), f))
        except ValueError:
            pass
    return sorted(out)


def contexto_disponible(cam, t):
    """True si ya hay un segmento raw CERRADO que cubre t+CONTEXTO_VLM_S.
    (Los ctx jpgs estaban deprecados -> ctx_jpgs() siempre vacio -> antes cada
    clip esperaba el timeout MAX_ESPERA_S entero. Ahora nos basamos en los
    segmentos raw cerrados, de donde el VLM y recorta_contexto leen los frames:
    se suelta el clip justo cuando el contexto existe, sin timeout ciego.)"""
    objetivo = t + timedelta(seconds=CONTEXTO_VLM_S)
    for ini, _ in segmentos_cerrados(cam):
        if ini <= objetivo <= ini + timedelta(seconds=C.SEGMENTO_S):
            return True
    return False


def _jpg_mas_cercano(jpgs, epoch, tol=4):
    mejor = min(jpgs, key=lambda x: abs(x[0] - epoch), default=None)
    if mejor and abs(mejor[0] - epoch) <= tol:
        return cv2.imread(mejor[1])
    return None


def frames_de_contexto(cam, t):
    """3 frames antes y 3 despues de la alerta. Primero de las fotos de
    contexto (rapidas); si faltan, de los segmentos raw cerrados (backlog)."""
    jpgs = ctx_jpgs(cam)
    antes, despues = [], []
    pasos = [CONTEXTO_VLM_S * k / N_CTX for k in range(1, N_CTX + 1)]  # 3,6,9,12,15
    offsets_a = tuple(-p for p in reversed(pasos))
    offsets_d = tuple(pasos)
    for dt in offsets_a:
        fr = _jpg_mas_cercano(jpgs, t.timestamp() + dt)
        if fr is not None:
            antes.append(fr)
    for dt in offsets_d:
        fr = _jpg_mas_cercano(jpgs, t.timestamp() + dt)
        if fr is not None:
            despues.append(fr)
    if len(antes) >= 2 and len(despues) >= 2:
        return antes, despues
    # fallback para clips antiguos (backlog): segmentos raw cerrados
    segs = segmentos_cerrados(cam)
    antes, despues = [], []
    for dt in offsets_a:
        fr = frame_en(segs, t + timedelta(seconds=dt))
        if fr is not None:
            antes.append(fr)
    for dt in offsets_d:
        fr = frame_en(segs, t + timedelta(seconds=dt))
        if fr is not None:
            despues.append(fr)
    return antes, despues


def clasifica(path):
    nombre = os.path.basename(path)
    cam = nombre.split("_")[0]
    zona = ZONAS.get(cam, "tienda")
    frames = frames_de_clip(path)
    if len(frames) < 3:
        return {"veredicto": "error", "razon": "clip ilegible"}
    antes, despues = [], []
    try:
        antes, despues = frames_de_contexto(cam, t_de_nombre(nombre))
    except Exception:
        pass
    con_contexto = len(antes) >= 2 and len(despues) >= 2
    if con_contexto:
        frames = antes + frames + despues
    contenido = [{"type": "text",
                  "text": f"Camara: {zona}. Fotogramas en orden temporal:"}]
    for fr in frames:
        ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.standard_b64encode(jpg.tobytes()).decode()
        contenido.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    contenido.append({"type": "text",
                      "text": "Evalua si la persona marcada (la mas centrada en "
                              "los recortes) esta cometiendo un hurto."})
    r = requests.post(
        API_URL,
        headers={"X-PC-Api-Key": PC_API_KEY,
                 "Content-Type": "application/json"},
        json={"model_alias": MODELO,
              "idempotency_key": nombre,
              "params": {"max_tokens": 500},
              "messages": [{"role": "system", "content": prompt_sistema(con_contexto)},
                           {"role": "user", "content": contenido}]},
        timeout=130,
    )
    r.raise_for_status()
    texto = r.json()["content"].strip()
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    out = json.loads(m.group(0))
    out["con_contexto"] = con_contexto
    return out


def t_de_nombre(nombre):
    m = re.search(r"_(\d{8})_(\d{6})_", nombre)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _faststart(origen, destino):
    """Reordena el atomo moov al principio del MP4 (qtfaststart, python puro,
    sin ffmpeg). OpenCV escribe el moov al final; sin esto la app no puede
    generar miniaturas descargando solo el primer MB ('moov atom not found')."""
    try:
        from qtfaststart import processor
        processor.process(origen, destino)
        return True
    except Exception:
        log("aviso: no se pudo aplicar faststart:\n" + traceback.format_exc())
        return False


def recorta_contexto(cam, t_alerta, destino):
    """Concatena de los segmentos raw los frames en [t-30s, t+30s]."""
    t0, t1 = t_alerta - timedelta(seconds=CONTEXTO_S), t_alerta + timedelta(seconds=CONTEXTO_S)
    segs = []
    for p in sorted(glob.glob(os.path.join(C.RAW_DIR, cam, f"{cam}_*.mp4"))):
        m = re.search(r"_(\d{8})_(\d{6})\.mp4$", p)
        if not m:
            continue
        ini = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if ini <= t1 and ini + timedelta(seconds=C.SEGMENTO_S + 60) >= t0:
            segs.append((ini, p))
    if not segs:
        return 0.0
    destino_raw = destino + ".raw.mp4"
    writer, escritos, fps_w = None, 0, 12.0
    for ini, p in segs:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = ini + timedelta(seconds=i / fps)
            i += 1
            if t < t0:
                continue
            if t > t1:
                break
            if writer is None:
                h, w = fr.shape[:2]
                fps_w = max(fps, 1)
                writer = cv2.VideoWriter(destino_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_w, (w, h))
            writer.write(fr)
            escritos += 1
        cap.release()
    if writer is not None:
        writer.release()
    if escritos == 0:
        return 0.0
    if not _faststart(destino_raw, destino):
        os.replace(destino_raw, destino)  # fallback: sin faststart, mejor que nada
    else:
        os.remove(destino_raw)
    return escritos / fps_w


VEREDICTOS_VALIDOS = ("hurto_probable", "sospechoso", "normal", "empleado")

# Bypass por confianza del detector: con p >= BYPASS_P la alerta va a la app
# DIRECTAMENTE, sin esperar (ni obedecer) al VLM. El VLM se consulta despues
# igualmente, pero solo como etiqueta informativa para reentreno/ — no puede
# vetar un hurto de alta confianza (visto en produccion: hurtos reales con
# detector fuerte que el VLM clasificaba "normal" por salir la persona lejos).
BYPASS_P = float(os.environ.get("BYPASS_P", "0.90"))
# Diseño de dos niveles (04-08 noche, v13): score MUY alto (>=BYPASS_P) va DIRECTO
# a la app sin esperar al VLM (protege el recall de los hurtos confiados); el resto
# (0.30-0.90) pasa por el VLM paralelizado, que deja pasar hurto_probable/sospechoso.
# v13 tiene FP bajos, así que el bypass a 0.90 casi solo deja pasar hurtos reales.


def p_de_nombre(nombre):
    m = re.search(r"_p(\d+(?:\.\d+)?)\.mp4$", nombre)
    return float(m.group(1)) if m else None


def _normaliza_txt(s):
    """Quita acentos, pasa a minusculas y colapsa letras consecutivas
    repetidas (para tolerar erratas del VLM como 'vereddicto' o 'normall')."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


_VEREDICTOS_NORM = {_normaliza_txt(v): v for v in VEREDICTOS_VALIDOS}


def extrae_veredicto(r):
    """Devuelve el veredicto canonico tolerando erratas del VLM tanto en la
    clave del JSON ('vereddicto', 'veredícto', ...) como en el valor. Si no
    reconoce nada, 'error' (no se archiva ni sube a la app)."""
    clave = "veredicto" if "veredicto" in r else next(
        (k for k in r if _normaliza_txt(k) == "veredicto"), None)
    if clave is None:
        return "error"
    return _VEREDICTOS_NORM.get(_normaliza_txt(r[clave]), "error")


def archiva(path, veredicto):
    d = os.path.join(REENTRENO, veredicto)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(path))
    if not os.path.exists(dst):
        try:
            os.link(path, dst)
        except OSError:
            import shutil
            shutil.copy2(path, dst)


# Deduplicado de entregas a la app: el clip de contexto son 15s del plano
# COMPLETO de la camara, asi que dos envios de la misma camara a <15s
# repiten los mismos segundos de video (medido 07-08: 64% de clips
# consecutivos solapados, 6s repetidos de media). Con 15s de separacion los
# clips teselan el tiempo: cobertura continua sin segundos duplicados. Solo
# afecta a la entrega a la app — el VLM analiza y archiva todos los clips.
MIN_SEP_APP_S = 15.0
_enviados_t = {}      # cam -> [datetimes de alertas ya enviadas]
_enviados_lock = threading.Lock()


def envia_a_app(nombre, path):
    cam = nombre.split("_")[0]
    try:
        t = t_de_nombre(nombre)
        with _enviados_lock:
            ts = _enviados_t.setdefault(cam, [])
            if any(abs((t - x).total_seconds()) < MIN_SEP_APP_S for x in ts):
                log(f"no enviado ({cam} ya tiene clip a <{MIN_SEP_APP_S:.0f}s "
                    f"en la app, mismos segundos de video): {nombre}")
                return
            ts.append(t)
            del ts[:-30]
        tmp = os.path.join(VLM_DIR, f"ctx_{nombre}")
        # La app extrae el numero de camara con el patron IDCAM(\d+): el
        # digito tiene que ir PEGADO a "IDCAM" (sin "_" de por medio) o
        # no reconoce la camara y no muestra ningun nombre.
        digito_cam = re.search(r"\d+", cam)
        digito_cam = digito_cam.group(0) if digito_cam else ""
        destino_final = os.path.join(OUTPUTS_PCAGENT, f"IDCAM{digito_cam}_{nombre}")
        dur = recorta_contexto(cam, t, tmp)
        intentos_extra = 0
        while dur < MIN_CONTEXTO_APP_S and intentos_extra < REINTENTOS_CONTEXTO_APP:
            # el segmento raw que cubre t+CONTEXTO_S puede seguir abierto
            # (se cierra cada SEGMENTO_S) y un clip parcial cuenta como exito
            # para recorta_contexto: reintentar hasta cubrir la ventana entera,
            # no solo cuando no se escribio nada.
            time.sleep(ESPERA_REINTENTO_CONTEXTO_APP_S)
            intentos_extra += 1
            dur = recorta_contexto(cam, t, tmp)
        if dur > 0:
            os.replace(tmp, destino_final)
            log(f"contexto {dur:.1f}s (obj +-{CONTEXTO_S}s) enviado a pc-agent: "
                f"{os.path.basename(destino_final)}"
                + (f" (tras {intentos_extra} reintentos)" if intentos_extra else ""))
        else:
            if not _faststart(path, destino_final):
                import shutil
                shutil.copy2(path, destino_final)
            log(f"sin raw disponible tras {intentos_extra} reintentos, "
                f"enviado el clip corto: {nombre}")
    except Exception:
        log("error recortando contexto:\n" + traceback.format_exc())


def procesa(path, estado):
    nombre = os.path.basename(path)
    p_det = p_de_nombre(nombre)
    bypass = p_det is not None and p_det >= BYPASS_P
    if bypass:
        log(f"{nombre} -> BYPASS a la app (detector p={p_det:.2f} >= {BYPASS_P})")
        envia_a_app(nombre, path)
    try:
        r = clasifica(path)
    except Exception as e:
        if bypass:
            # la alerta ya esta en la app; el veredicto VLM era solo informativo
            log(f"aviso: VLM fallo tras bypass en {nombre}: {e}")
            with _lock, open(VEREDICTOS, "a") as f:
                f.write(json.dumps({"veredicto": "error", "bypass": True,
                                    "p_detector": p_det, "clip": nombre,
                                    "ts": datetime.now().isoformat()},
                                   ensure_ascii=False) + "\n")
            archiva(path, "hurto_probable")
            estado[nombre] = "bypass_sin_vlm"
            return True
        log(f"error clasificando {nombre}: {e}")
        return False  # se reintenta en el siguiente ciclo
    v = extrae_veredicto(r)
    for k in [k for k in r if k != "veredicto" and _normaliza_txt(k) == "veredicto"]:
        del r[k]  # limpia claves con errata, se deja solo la canonica
    r["veredicto"] = v
    r["bypass"] = bypass
    if p_det is not None:
        r["p_detector"] = p_det
    log(f"{nombre} -> {v} ({r.get('confianza','-')})" + (" [bypass ya enviado]" if bypass else ""))
    r["clip"], r["ts"] = nombre, datetime.now().isoformat()
    with _lock, open(VEREDICTOS, "a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if v in ("hurto_probable", "sospechoso", "normal", "empleado"):
        archiva(path, v)
    if not bypass and v in ("hurto_probable", "sospechoso"):
        envia_a_app(nombre, path)
    estado[nombre] = v if not bypass else f"{v}|bypass"
    return True


def main():
    if not PC_API_KEY:
        sys.exit("Falta PC_API_KEY en el entorno")
    os.makedirs(VLM_DIR, exist_ok=True)
    os.makedirs(REENTRENO, exist_ok=True)
    estado = {}
    if os.path.exists(ESTADO):
        estado = json.load(open(ESTADO))
    log(f"daemon arrancado, {len(estado)} clips ya procesados | {VLM_WORKERS} hilos")
    en_curso = set()  # clips que ya se enviaron a un hilo (evita re-enviar el mismo)

    def trabaja(p):
        # cada hilo procesa un clip; al terminar, guarda el estado bajo lock.
        try:
            if procesa(p, estado):
                with _lock:
                    json.dump(estado, open(ESTADO, "w"))
        except Exception:
            log("excepcion procesando " + os.path.basename(p) + ":\n" + traceback.format_exc())
        finally:
            with _lock:
                en_curso.discard(os.path.basename(p))

    with ThreadPoolExecutor(max_workers=VLM_WORKERS) as pool:
        while True:
            try:
                ahora = time.time()
                for p in sorted(glob.glob(os.path.join(C.ALERTAS_DIR, "*.mp4"))):
                    nombre = os.path.basename(p)
                    if nombre in estado or nombre in en_curso:
                        continue
                    edad = ahora - os.path.getmtime(p)
                    if edad < ASENTADO_S:
                        continue
                    # esperar (max MAX_ESPERA_S) a que el raw cubra t+contexto
                    if edad < MAX_ESPERA_S:
                        try:
                            if not contexto_disponible(nombre.split("_")[0], t_de_nombre(nombre)):
                                continue
                        except Exception:
                            pass
                    en_curso.add(nombre)
                    pool.submit(trabaja, p)   # varios clips en vuelo a la vez -> la pool de keys del central los reparte
            except Exception:
                log("excepcion en el bucle:\n" + traceback.format_exc())
            time.sleep(INTERVALO_S)


if __name__ == "__main__":
    main()

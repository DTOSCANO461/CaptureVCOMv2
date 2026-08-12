"""
ejemplo_captura.py
===================================================================
COMO CAPTURAMOS VIDEO EN ESTE PROYECTO (cámara EN VIVO y .ts OFFLINE)
SIN cv2 y SIN copiar frames CPU -> GPU, usando decodificacion por
HARDWARE de NVIDIA (NVDEC) a traves de la libreria "PyNvVideoCodec".

Este archivo es 100% autocontenido: no importa nada de otros .py del
repo (nada de custom_utils / config_tienda / etc). Todo lo que en el
proyecto real sale de configuracion aca esta HARDCODEADO como ejemplo,
marcado con "# <-- CONFIGURAR".

-------------------------------------------------------------------
POR QUE NO cv2.VideoCapture
-------------------------------------------------------------------
cv2.VideoCapture (con backend FFmpeg o GStreamer) decodifica el video
y entrega el frame como un numpy array EN RAM (CPU). Si despues ese
frame se necesita en la GPU (para un modelo torch, como en este
proyecto), hay que hacer una copia explicita CPU->GPU por PCIe en
CADA frame (`torch.from_numpy(frame).cuda()`). Con varias camaras en
paralelo eso es un cuello de botella real de ancho de banda y de CPU.

PyNvVideoCodec usa el motor de decodificacion por hardware NVDEC de
la placa NVIDIA:
  - El video comprimido (H264/H265) entra tal cual a la GPU.
  - NVDEC decodifica ADENTRO de la GPU, en memoria de dispositivo
    (usedevicememory=True).
  - El frame resultante se expone via DLPack, y `torch.from_dlpack()`
    lo envuelve como un tensor CUDA SIN copiar nada: es la misma
    memoria de video ya decodificada. Cero trafico CPU<->GPU por frame.

En consecuencia, cv2 en este archivo NO se usa para decodificar nada.
Si en algun punto queres ver un frame en pantalla o guardarlo a disco
como jpg (debug), eso es una operacion aparte, opcional, y ahi si
tocaria bajar el tensor a CPU (`tensor.cpu().numpy()`) — pero eso NO
es el pipeline de captura, es solo una inspeccion puntual.

-------------------------------------------------------------------
DOS FORMAS DE CAPTURAR "EN VIVO" DESDE UNA CAMARA (RTSP)
-------------------------------------------------------------------
A) RECOMENDADA / LA QUE USAMOS EN PRODUCCION:
   Un proceso FFmpeg graba el RTSP a disco en segmentos .ts rotativos
   usando "-c:v copy" (remux puro, SIN decodificar, FFmpeg solo copia
   los paquetes comprimidos tal cual llegan de la camara -> costo de
   CPU minimo). Un segundo componente (el mismo lector que usamos para
   offline) va leyendo esos .ts a medida que aparecen y los decodifica
   en GPU con NVDEC. Ver: `grabar_rtsp_a_ts()` + `LectorTS`.

   Ventajas: si el decoder GPU se cae o hay que reiniciar el proceso
   de inferencia, los .ts en disco no se pierden (se puede retomar /
   reprocesar); y el codigo que lee "camara en vivo" y el que lee
   ".ts offline" es EXACTAMENTE EL MISMO (`LectorTS`), solo cambia si
   la carpeta sigue recibiendo archivos nuevos o no.

B) DECODIFICACION DIRECTA EN RAM, SIN TOCAR DISCO:
   FFmpeg lee el RTSP y por stdout (pipe) entrega el elementary stream
   H264 crudo (de nuevo, "-c:v copy", sin decodificar). Esos bytes
   comprimidos se pasan a PyNvVideoCodec via un demuxer "por callback"
   (`nvc.CreateDemuxer(feeder_callback)`) en vez de pasarle una ruta
   de archivo. NVDEC decodifica igual, 100% en GPU. Util para no
   escribir nada a disco, pero si el proceso muere se pierde lo que
   no se proceso (no hay .ts para reprocesar). Ver: `decodificar_rtsp_en_vivo()`.

-------------------------------------------------------------------
CAPTURA OFFLINE DESDE .ts YA EXISTENTES
-------------------------------------------------------------------
Es el mismo `LectorTS` de la opcion (A): se le apunta a una carpeta
con archivos .ts ya grabados (por ejemplo, generados por
`grabar_rtsp_a_ts()` en una corrida anterior, o extraidos de un video
mp4 con ffmpeg -f segment) y decodifica frame a frame en GPU en el
mismo orden en que fueron grabados. Ver: `ejemplo_offline()`.

-------------------------------------------------------------------
REQUISITOS
-------------------------------------------------------------------
- GPU NVIDIA con NVDEC (la gran mayoria de las GeForce/RTX/Tesla/A/L/H
  lo traen). Ojo: las GPU "de consumo" (GeForce) limitan la cantidad
  de sesiones de decodificacion simultaneas por driver (suele ser 3-5
  sesiones concurrentes salvo que se aplique el patch no-oficial
  "nvidia-patch"); las Quadro/Tesla/Data-center no tienen ese limite.
- Driver NVIDIA + CUDA instalados.
- `pip install PyNvVideoCodec` (requiere que el driver soporte el
  Video Codec SDK que usa la version instalada de la libreria).
- Binario `ffmpeg` disponible en PATH (se usa SOLO para leer el RTSP
  de la red / remuxear "copy", nunca para decodificar frames).
- `pip install torch` (para envolver el frame decodificado como
  tensor CUDA via `torch.from_dlpack`).
===================================================================
"""

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import PyNvVideoCodec as nvc


# ===================================================================
# CONFIG DE EJEMPLO (hardcodeado a proposito: nada de imports locales)
# ===================================================================
RTSP_URL_EJEMPLO = "rtsp://usuario:password@192.168.1.108:554/unicast/c1/s0/live"  # <-- CONFIGURAR
CARPETA_TS_EJEMPLO = "./rstp_ejemplo/cam0"  # <-- CONFIGURAR (donde se graban/leen los .ts)
GPU_ID = 0  # <-- CONFIGURAR (indice de GPU si hay mas de una)
SEGMENT_TIME_SEG = 2     # duracion de cada .ts grabado
MAX_SEGMENTOS = 200      # cuantos .ts se conservan antes de rotar/borrar viejos


# ===================================================================
# A.1) GRABAR RTSP -> .ts ROTATIVOS, SIN DECODIFICAR (FFmpeg "-c:v copy")
# ===================================================================
def grabar_rtsp_a_ts(
    rtsp_url: str,
    carpeta_salida: str,
    segment_time: int = SEGMENT_TIME_SEG,
    max_segmentos: int = MAX_SEGMENTOS,
    stop_event: Optional[threading.Event] = None,
) -> subprocess.Popen:
    """
    Lanza un proceso FFmpeg que se conecta al RTSP y va escribiendo
    segmentos .ts rotativos a disco (output_0000001.ts, 0000002.ts, ...).

    "-c:v copy" / "-c:a copy" = FFmpeg NO decodifica ni recodifica, solo
    reempaqueta (remux) los paquetes comprimidos que ya vienen de la
    camara. Es la parte de "captura de red", corre en CPU pero es
    barata (no hay decodificacion de pixeles involucrada).
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    output_pattern = os.path.join(carpeta_salida, "output_%07d.ts")

    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-segment_wrap", str(max_segmentos),
        "-segment_format", "mpegts",
        output_pattern,
    ]

    print(f"[grabar_rtsp_a_ts] Iniciando FFmpeg -> {output_pattern}")

    proceso = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if stop_event is not None:
        def _vigilar_stop():
            stop_event.wait()
            proceso.terminate()

        threading.Thread(target=_vigilar_stop, daemon=True).start()

    return proceso


# ===================================================================
# A.2 / B) LECTOR-DECODER GPU (NVDEC) DE ARCHIVOS .ts
#   Sirve tanto para OFFLINE (carpeta ya completa, no cambia mas) como
#   para "camara grabando en vivo" (carpeta a la que FFmpeg le sigue
#   agregando .ts nuevos mientras este lector los va consumiendo).
# ===================================================================
class LectorTS:
    """
    Decodifica en GPU (NVDEC), frame a frame, la secuencia de archivos
    .ts de una carpeta, en orden de aparicion. Cuando termina un .ts,
    pasa automaticamente al siguiente (esperando si es "en vivo" y
    todavia no llego el proximo).
    """

    def __init__(self, carpeta: str, gpu_id: int = GPU_ID, en_vivo: bool = False):
        """
        Args:
            carpeta: carpeta con los archivos .ts (mismo path que
                `carpeta_salida` de grabar_rtsp_a_ts, o cualquier
                carpeta con .ts ya generados para modo offline puro).
            gpu_id: GPU NVIDIA a usar para NVDEC.
            en_vivo: si True, cuando no hay .ts nuevo disponible espera
                (poll) en vez de terminar. En offline puro se puede
                dejar en False: si no hay mas archivos, se corta.
        """
        self.carpeta = Path(carpeta)
        self.gpu_id = gpu_id
        self.en_vivo = en_vivo

        self._procesados: set = set()
        self._demuxer = None
        self._decoder = None
        self._packet_iter = None
        self._frames_pendientes = []
        self._video_actual: Optional[Path] = None

    # -------------------------------------------------------------
    def _listar_ts_pendientes(self):
        archivos = sorted(
            f for f in self.carpeta.glob("*.ts")
            if f.is_file() and f.name not in self._procesados
        )
        return archivos

    def _esperar_siguiente_video(self) -> Path:
        while True:
            pendientes = self._listar_ts_pendientes()

            if pendientes:
                # Igual que en produccion: no abrimos el .ts mas nuevo
                # de inmediato porque FFmpeg todavia puede estar
                # escribiendolo. Si hay margen, tomamos el mas viejo
                # disponible (asegura que ya este cerrado).
                return pendientes[0]

            if not self.en_vivo:
                raise StopIteration("No hay mas archivos .ts en la carpeta (modo offline).")

            time.sleep(0.2)

    def _esperar_archivo_estable(self, path: Path):
        """Evita abrir un .ts que FFmpeg todavia esta escribiendo."""
        size_prev = path.stat().st_size
        time.sleep(0.2)
        size_now = path.stat().st_size

        while size_prev != size_now:
            size_prev = size_now
            time.sleep(0.2)
            size_now = path.stat().st_size

    def _abrir_video(self, path: Path):
        self._esperar_archivo_estable(path)

        # Demuxer/decoder de PyNvVideoCodec directo sobre un archivo local:
        # el .ts ya esta en disco, no hace falta callback de red aca.
        self._demuxer = nvc.CreateDemuxer(str(path))

        self._decoder = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,              # decodifica DENTRO de la GPU
            outputColorType=nvc.OutputColorType.RGB,
        )

        self._packet_iter = iter(self._demuxer)
        self._video_actual = path
        self._procesados.add(path.name)

    # -------------------------------------------------------------
    def get_frame(self) -> torch.Tensor:
        """
        Devuelve el siguiente frame decodificado como tensor CUDA
        (H, W, 3) uint8 RGB, ya viviendo en la GPU (sin copia CPU).
        """
        if self._demuxer is None:
            self._abrir_video(self._esperar_siguiente_video())

        while True:
            if not self._frames_pendientes:
                try:
                    paquete = next(self._packet_iter)
                except StopIteration:
                    # Se termino este .ts -> pasar al siguiente
                    self._demuxer = None
                    self._abrir_video(self._esperar_siguiente_video())
                    continue

                self._frames_pendientes = list(self._decoder.Decode(paquete))

            if self._frames_pendientes:
                frame_gpu = self._frames_pendientes.pop(0)

                # DLPack: envuelve el buffer YA DECODIFICADO EN GPU como
                # tensor de torch. No hay memcpy CPU<->GPU en esta linea.
                tensor = torch.from_dlpack(frame_gpu)

                # El buffer interno del decoder se puede reciclar en la
                # proxima llamada a Decode(): si vas a guardar el frame
                # para usarlo mas adelante (o mandarlo a una cola de
                # otro proceso), cloná el tensor. Si lo consumis
                # inmediatamente (ej. modelo forward ya mismo), el clone
                # es innecesario y te ahorra esa copia.
                return tensor.clone()


# ===================================================================
# B) DECODIFICACION DIRECTA DE RTSP EN RAM, SIN TOCAR DISCO
#    (demuxer "por callback" de PyNvVideoCodec + FFmpeg como fuente
#    de bytes comprimidos vía pipe, otra vez con "-c:v copy": FFmpeg
#    NO decodifica nada, solo entrega el elementary stream H264 crudo)
# ===================================================================
def decodificar_rtsp_en_vivo(rtsp_url: str, gpu_id: int = GPU_ID, n_frames: int = 200):
    """
    Ejemplo de captura 100% en memoria: no escribe ningun .ts a disco.
    Utilizar solo si NO se necesita reprocesar / persistir el video
    (en este proyecto usamos la opcion A porque queremos poder volver
    a mirar el video de un evento despues).
    """
    buffer_compartido = bytearray()
    lock = threading.Lock()
    hay_datos = threading.Event()

    def feeder_callback(demuxer_buffer: bytearray) -> int:
        """PyNvVideoCodec llama esto cada vez que necesita mas bytes comprimidos."""
        hay_datos.wait()
        with lock:
            largo = min(len(buffer_compartido), len(demuxer_buffer))
            if largo == 0:
                return 0
            demuxer_buffer[:largo] = buffer_compartido[:largo]
            del buffer_compartido[:largo]
            if len(buffer_compartido) == 0:
                hay_datos.clear()
            return largo

    def lector_ffmpeg():
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-c:v", "copy",   # sin decodificar, solo remux a elementary stream
            "-f", "h264",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            with lock:
                buffer_compartido.extend(chunk)
                hay_datos.set()

    threading.Thread(target=lector_ffmpeg, daemon=True).start()

    demuxer = nvc.CreateDemuxer(feeder_callback)
    decoder = nvc.CreateDecoder(
        gpuid=gpu_id,
        codec=demuxer.GetNvCodecId(),
        usedevicememory=True,
        outputColorType=nvc.OutputColorType.RGB,
    )

    print("[decodificar_rtsp_en_vivo] Decodificando en GPU (NVDEC), sin escribir a disco...")

    contador = 0
    for paquete in demuxer:
        for frame_gpu in decoder.Decode(paquete):
            tensor = torch.from_dlpack(frame_gpu).clone()
            contador += 1
            print(f"frame {contador}: shape={tuple(tensor.shape)} device={tensor.device} dtype={tensor.dtype}")

            if contador >= n_frames:
                return


# ===================================================================
# A) EJEMPLO OFFLINE: leer una carpeta de .ts ya grabados
# ===================================================================
def ejemplo_offline(carpeta_ts: str, n_frames: int = 200):
    lector = LectorTS(carpeta_ts, gpu_id=GPU_ID, en_vivo=False)

    for i in range(n_frames):
        tensor = lector.get_frame()
        print(f"[offline] frame {i}: shape={tuple(tensor.shape)} device={tensor.device} dtype={tensor.dtype}")


# ===================================================================
# A) EJEMPLO "EN VIVO GRABANDO": FFmpeg graba .ts + LectorTS decodifica
#    a medida que van apareciendo (esto es lo que usamos en produccion)
# ===================================================================
def ejemplo_vivo_grabando(rtsp_url: str, carpeta_salida: str, n_frames: int = 200):
    stop_event = threading.Event()

    proceso_ffmpeg = grabar_rtsp_a_ts(rtsp_url, carpeta_salida, stop_event=stop_event)

    try:
        lector = LectorTS(carpeta_salida, gpu_id=GPU_ID, en_vivo=True)

        for i in range(n_frames):
            tensor = lector.get_frame()
            print(f"[vivo] frame {i}: shape={tuple(tensor.shape)} device={tensor.device} dtype={tensor.dtype}")
    finally:
        stop_event.set()
        proceso_ffmpeg.terminate()


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ejemplo de captura con NVDEC (PyNvVideoCodec), sin cv2, sin copias CPU<->GPU."
    )
    parser.add_argument(
        "--modo",
        choices=["vivo_directo", "vivo_grabando", "offline"],
        default="vivo_grabando",
        help=(
            "vivo_directo: RTSP -> decodifica en RAM, sin tocar disco (opcion B). "
            "vivo_grabando: RTSP -> graba .ts -> decodifica (opcion A, la de produccion). "
            "offline: lee una carpeta de .ts ya existentes."
        ),
    )
    parser.add_argument("--rtsp_url", type=str, default=RTSP_URL_EJEMPLO)
    parser.add_argument("--carpeta_ts", type=str, default=CARPETA_TS_EJEMPLO)
    parser.add_argument("--n_frames", type=int, default=200)
    args = parser.parse_args()

    if args.modo == "vivo_directo":
        decodificar_rtsp_en_vivo(args.rtsp_url, gpu_id=GPU_ID, n_frames=args.n_frames)
    elif args.modo == "vivo_grabando":
        ejemplo_vivo_grabando(args.rtsp_url, args.carpeta_ts, n_frames=args.n_frames)
    else:
        ejemplo_offline(args.carpeta_ts, n_frames=args.n_frames)


if __name__ == "__main__":
    main()

# kafka_test_producer.py
"""
Multi-camera Kafka Producer for testing SmartBuildings with the Kafka container
(confluentinc/cp-kafka:latest on localhost:9092, topic 'meu-topico-de-video').

Publishes JPEG-encoded frames keyed by camera ID (e.g. b'cam1', b'cam2') to match
the 'compact,delete' cleanup policy of the Kafka container.

Usage:
    ./venv/bin/python kafka_test_producer.py            # Auto-detects RTSP or falls back to local MP4 videos
    ./venv/bin/python kafka_test_producer.py --local    # Streams local MP4 videos as cam1 & cam2 at 30 FPS
    ./venv/bin/python kafka_test_producer.py --fps 60   # Streams at 60 FPS per camera
"""
import argparse
import logging
import os
import signal
import threading
import time
import cv2
from confluent_kafka import Producer

from Config import Config

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;3000000",
)


def _stream_camera_to_kafka(
    cam_id: str,
    source: str,
    producer: Producer,
    topic: str,
    width: int,
    height: int,
    fps: float,
    jpeg_quality: int,
    stop_event: threading.Event,
) -> None:
    is_network = str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    key_bytes = cam_id.encode("utf-8")
    interval = 1.0 / fps if fps > 0 else 0.0

    cap = None
    sent_count = 0
    t_log = time.perf_counter()

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logging.warning(f"[Producer-{cam_id}] Não foi possível abrir '{source}'. Tentando em 2s...")
                time.sleep(2.0)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logging.info(f"[Producer-{cam_id}] Conectado a '{source}' -> publicando no tópico '{topic}'...")

        t_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            if not is_network:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        ok_enc, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok_enc:
            try:
                producer.produce(topic, key=key_bytes, value=buf.tobytes())
                producer.poll(0)
                sent_count += 1
            except BufferError:
                producer.poll(0.05)

        now = time.perf_counter()
        if (now - t_log) >= 5.0:
            actual_fps = sent_count / (now - t_log)
            logging.info(f"[Producer-{cam_id}] Publicando no Kafka a {actual_fps:.1f} FPS")
            sent_count = 0
            t_log = now

        if interval > 0:
            elapsed = time.perf_counter() - t_start
            sleep_s = interval - elapsed
            if sleep_s > 0.0005:
                time.sleep(sleep_s)

    if cap is not None:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Camera Kafka Video Producer for SmartBuildings")
    parser.add_argument("--bootstrap", default=Config.KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=Config.KAFKA_TOPIC, help="Kafka topic name")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS per camera")
    parser.add_argument("--quality", type=int, default=75, help="JPEG compression quality (1-100)")
    parser.add_argument("--local", action="store_true", help="Stream local MP4 video files (default)")
    parser.add_argument("--rtsp", action="store_true", help="Stream live LAN RTSP cameras to Kafka")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.rtsp:
        sources = {
            "cam1": "rtsp://admin:Aluno%4000@10.145.80.52:554",
            "cam2": "rtsp://admin:Aluno%4000@10.145.80.41:554/Streaming/Channels/101",
        }
    else:
        sources = {
            "cam1": "models/pessoas.mp4",
            "cam2": "examples/pessoas_rua_60fps.mp4",
        }

    producer_conf = {
        "bootstrap.servers": args.bootstrap,
        "linger.ms": 2,
        "batch.num.messages": 16,
        "queue.buffering.max.kbytes": 32768,
        "message.max.bytes": 5242880,
        "acks": 1,
    }
    producer = Producer(producer_conf)

    stop_event = threading.Event()

    def _sig_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    threads = []
    for cid, src in sources.items():
        if not os.path.exists(src) and not src.startswith("rtsp://"):
            logging.warning(f"Arquivo '{src}' não encontrado para {cid}, ignorando.")
            continue
        t = threading.Thread(
            target=_stream_camera_to_kafka,
            args=(
                cid,
                src,
                producer,
                args.topic,
                Config.PROCESSING_WIDTH,
                Config.PROCESSING_HEIGHT,
                args.fps,
                args.quality,
                stop_event,
            ),
            daemon=True,
            name=f"Producer-{cid}",
        )
        t.start()
        threads.append(t)

    logging.info(
        f"Produtor Kafka iniciado em '{args.bootstrap}' (tópico='{args.topic}', câmeras={list(sources.keys())}). "
        f"Pressione Ctrl+C para encerrar."
    )

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        producer.flush(timeout=2.0)
        logging.info("Produtor Kafka encerrado.")


if __name__ == "__main__":
    main()

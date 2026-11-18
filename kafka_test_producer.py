# kafka_test_producer.py
import argparse
import logging
import os
import signal
import threading
import time
import cv2
from confluent_kafka import Producer

from Config import Config

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;3000000")


def _open_capture(source: str, cam_id: str):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.warning(f"[Producer-{cam_id}] Não foi possível abrir '{source}'. Tentando em 2s...")
        time.sleep(2.0)
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logging.info(f"[Producer-{cam_id}] Conectado a '{source}'...")
    return cap


def _read_and_encode_frame(cap: cv2.VideoCapture, is_network: bool, width: int, height: int, quality: int):
    ok, frame = cap.read()
    if (not ok or frame is None) and not is_network:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok_enc else None


def _publish_payload(producer: Producer, topic: str, key_bytes: bytes, payload: bytes) -> bool:
    try:
        producer.produce(topic, key=key_bytes, value=payload)
        producer.poll(0)
        return True
    except BufferError:
        producer.poll(0.05)
        return False


def _stream_camera_to_kafka(cam_id: str, source: str, producer: Producer, topic: str, width: int, height: int, fps: float, jpeg_quality: int, stop_event: threading.Event) -> None:
    is_net = str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
    key_b, interval, cap = cam_id.encode("utf-8"), (1.0 / fps if fps > 0 else 0.0), None
    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            cap = _open_capture(source, cam_id)
            continue
        t_start = time.perf_counter()
        payload = _read_and_encode_frame(cap, is_net, width, height, jpeg_quality)
        if payload is None:
            cap.release()
            cap = None
            continue
        _publish_payload(producer, topic, key_b, payload)
        sleep_s = interval - (time.perf_counter() - t_start)
        if sleep_s > 0.0005:
            time.sleep(sleep_s)
    if cap is not None:
        cap.release()


def _parse_args():
    parser = argparse.ArgumentParser(description="Multi-Camera Kafka Video Producer")
    parser.add_argument("--bootstrap", default=Config.KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=Config.KAFKA_TOPIC)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--rtsp", action="store_true")
    return parser.parse_args()


def _create_producer(bootstrap_servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap_servers, "linger.ms": 2, "batch.num.messages": 16,
        "queue.buffering.max.kbytes": 32768, "message.max.bytes": 5242880, "acks": 1,
    })


def _spawn_producer_threads(sources: dict, producer: Producer, args, stop_event: threading.Event) -> list[threading.Thread]:
    threads = []
    for cid, src in sources.items():
        if not os.path.exists(src) and not src.startswith("rtsp://"):
            continue
        t = threading.Thread(
            target=_stream_camera_to_kafka,
            args=(cid, src, producer, args.topic, Config.PROCESSING_WIDTH, Config.PROCESSING_HEIGHT, args.fps, args.quality, stop_event),
            daemon=True, name=f"Producer-{cid}",
        )
        t.start()
        threads.append(t)
    return threads


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    sources = {"cam1": "rtsp://admin:Aluno%4000@10.145.80.52:554", "cam2": "rtsp://admin:Aluno%4000@10.145.80.41:554/Streaming/Channels/101"} if args.rtsp else {"cam1": "testes/pessoas.mp4", "cam2": "examples/pessoas_rua_60fps.mp4"}
    producer, stop_event = _create_producer(args.bootstrap), threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    threads = _spawn_producer_threads(sources, producer, args, stop_event)
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        producer.flush(timeout=2.0)


if __name__ == "__main__":
    main()

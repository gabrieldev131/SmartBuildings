# main.py
import logging
import multiprocessing as mp

from Config import Config
from extraction.FrameReaderManagement import FrameReaderManagement
from identity.runtime_setup import apply as apply_runtime_setup


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    apply_runtime_setup()
    config = Config()
    app = FrameReaderManagement(config)
    app.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

# main.py
import config
import logging
from multiprocessing import freeze_support
from view.console_view import ConsoleView
from controller.AppController import AppController


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    view = ConsoleView()

    # Sem NetworkScanner nem FrameReader: a fonte de dados agora e o Kafka.
    # O AppController instancia o KafkaFrameReader internamente.
    model = {}

    controller = AppController(model, view, config)
    controller.run()


if __name__ == "__main__":
    freeze_support()
    main()
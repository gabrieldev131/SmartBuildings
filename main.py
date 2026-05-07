# main.py
import logging
from Config import Config
from extraction.FrameReaderManagement import FrameReaderManagement

def main():
    # Configuração de logs global
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    config = Config()
    
    app = FrameReaderManagement(config)
    app.run()

if __name__ == "__main__":
    main()
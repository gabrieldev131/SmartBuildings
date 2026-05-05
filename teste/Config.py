# config.py
class Config:
    # Modelos
    YOLO_MODEL_PATH = "models/yolo26n.pt"
    
    # Limiares de Movimento / Paragem
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30
    
    # Limiares de Identidade
    SIMILARITY_THRESHOLD = 0.25      
    MAX_SPATIAL_DISTANCE = 150    
    MAX_TIME_LOST = 5.0
    
    # Interface
    COLOR_STOPPED = (0, 0, 255)  # Vermelho
    COLOR_MOVING = (0, 255, 0)   # Verde
    FONT_SCALE = 0.5
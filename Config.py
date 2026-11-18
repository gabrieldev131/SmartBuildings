# Config.py
class Config:
    """Centralized configuration for SmartBuildings Multi-Camera Vision & Re-ID System."""

    # === EXECUTION & PERFORMANCE PIPELINE ===
    USE_GPU = True                   # True = CUDA GPU (FP16), False = CPU (FP32)
    EXECUTION_MODE = "batched"       # "batched" (CUDA Graph + Double-Buffered Pipeline) or "multiprocess"
    USE_CUDA_GRAPHS = True
    MAX_CAMERAS_BATCH = 8
    MAX_DETECTIONS_PER_CAMERA = 18

    VIDEO_FILE_REALTIME_PACING = False
    LOOP_LOCAL_VIDEOS = True

    # === YOLO DETECTION PARAMETERS (ultralytics + torchvision) ===
    YOLO_MODEL_PATH = "models/yolo26n.pt"
    YOLO_CONF_THRESHOLD = 0.40       # Confidence threshold for person tracking
    YOLO_INIT_CONF = 0.48            # Minimum confidence required to confirm a brand-new person track
    YOLO_CONF_LOW = 0.34             # Minimum confidence kept after GPU post-processing
    YOLO_IOU_THRESHOLD = 0.50        # Torchvision NMS IoU threshold

    # Geometric human bounding-box validation (rejects chairs, tables, and wide non-human objects)
    MIN_BOX_WIDTH = 16
    MIN_BOX_HEIGHT = 34
    MIN_ASPECT_RATIO = 0.18          # min (width / height)
    MAX_ASPECT_RATIO = 0.95          # max (width / height) - humans are taller than wide

    # === DEEPSORT TRACKING PARAMETERS (deep-sort-realtime) ===
    DEEPSORT_MAX_AGE = 90            # Max missed frames before a local track is deleted
    DEEPSORT_N_INIT = 2              # Consecutive hits required to confirm a tentative track
    DEEPSORT_NMS_MAX_OVERLAP = 0.75  # NMS overlap threshold inside DeepSort
    DEEPSORT_MAX_COSINE_DISTANCE = 0.40
    DEEPSORT_MAX_IOU_DISTANCE = 0.72
    DEEPSORT_NN_BUDGET = 35          # Bounded appearance gallery per local track
    DEEPSORT_MAX_COAST_FRAMES = 3    # Max frames to render damped prediction during brief detector blink
    DEEPSORT_BOX_SMOOTH_ALPHA = 0.65

    # === DECOUPLED BEHAVIOR PLUGINS (Registry Pattern in behaviors/) ===
    # Add any registered plugin key here (e.g., "stopped", "restricted_zone") - zero if/else required!
    ACTIVE_BEHAVIORS = ("stopped",)
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30
    RESTRICTED_ZONE_NORM = (0.0, 0.0, 0.25, 0.25)

    # === GLOBAL RE-ID & CLUSTERING THRESHOLDS ===
    SIMILARITY_THRESHOLD = 0.35      # Intra-camera appearance cosine distance threshold
    INTER_CAMERA_THRESHOLD = 0.37    # Cross-camera appearance cosine distance threshold
    SIMULTANEOUS_INTER_CAMERA_THRESHOLD = 0.30
    MAX_SPATIAL_DISTANCE = 180
    MAX_TIME_LOST = 120.0
    CAMERA_SWITCH_COOLDOWN = 0.8
    AUTO_CLUSTER_TIME_THRESHOLD = 15.0
    FEATURE_UPDATE_INTERVAL = 3

    # === DECOUPLED DATA EXPORTERS (Registry Pattern in exporters/) ===
    # Add any registered exporter key here (e.g., ("csv",), ("csv", "kafka")) - zero if/else required!
    ACTIVE_EXPORTERS = ("csv",)
    EXPORT_CSV_FILENAME = "tracking_data_final.csv"
    KAFKA_EVENTS_TOPIC = "smartbuildings-eventos"

    # === UI & PRESENTATION ===
    DISPLAY_ENABLED = True
    DISPLAY_GRID_VIEW = False
    DISPLAY_MAX_FPS = 60
    COLOR_STOPPED = (0, 0, 255)
    COLOR_MOVING = (0, 255, 0)
    FONT_SCALE = 0.5

    # === KAFKA VIDEO STREAMING ===
    KAFKA_BOOTSTRAP_SERVERS = "10.145.80.61:9092"
    KAFKA_TOPIC = "meu-topico-de-video"
    KAFKA_GROUP_ID = "smart-builds-consumer"
    KAFKA_TARGET_CAMERA = None
    KAFKA_EXPECTED_CAMERAS = 4

    # === FRAME DIMENSIONS ===
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480

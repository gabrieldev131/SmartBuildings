# Config.py
class Config:
    """
    Centralized configuration for SmartBuildings Multi-Camera Vision & Re-ID System.
    """

    # === EXECUTION & PERFORMANCE PIPELINE ===
    # "batched": Centralized GPU Batched Inference + CUDA Graphs + Double-Buffered Pipeline (60+ FPS across 8 cameras)
    # "multiprocess": Legacy 1-OS-process-per-camera with BaseManager socket IPC
    EXECUTION_MODE = "batched"
    USE_CUDA_GRAPHS = True
    MAX_CAMERAS_BATCH = 8
    MAX_DETECTIONS_PER_CAMERA = 16

    # Video file playback behavior:
    # False = process frames as fast as hardware allows (ideal for throughput testing)
    # True  = pace local video files (.mp4) at their native framerate (e.g., 60 FPS)
    VIDEO_FILE_REALTIME_PACING = False
    LOOP_LOCAL_VIDEOS = True

    # === YOLO DETECTION PARAMETERS ===
    YOLO_MODEL_PATH = "models/yolo26n.pt"
    YOLO_CONF_HIGH = 0.34          # High-confidence threshold for track creation & appearance updates
    YOLO_CONF_LOW = 0.22           # Low-confidence threshold for 2nd-stage occlusion recovery (ByteTrack-style)
    YOLO_IOU_THRESHOLD = 0.50      # Intra-class NMS IoU threshold to prevent duplicate boxes on one person

    # Geometric bounding-box validation (filters false positives such as furniture/reflections)
    MIN_BOX_WIDTH = 12
    MIN_BOX_HEIGHT = 24
    MIN_ASPECT_RATIO = 0.12        # min (width / height)
    MAX_ASPECT_RATIO = 3.50        # max (width / height) - supports close-up/seated/upper-body views

    # === DEEPSORT TRACKING PARAMETERS ===
    DEEPSORT_MAX_AGE = 60          # Max frames a track survives occlusion (1.0s at 60 FPS)
    DEEPSORT_N_INIT = 3            # Consecutive hits required to confirm a tentative track
    DEEPSORT_MAX_IOU_DISTANCE = 0.70
    DEEPSORT_MAX_COSINE_DISTANCE = 0.35
    DEEPSORT_NN_BUDGET = 25        # Bounded appearance gallery per local track
    DEEPSORT_MAX_COAST_FRAMES = 2  # Keep rendering Kalman prediction during 1-2 frame detector flicker

    # === STOPPED / LOITERING BEHAVIOR THRESHOLDS ===
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30

    # === BIFURCATED GLOBAL RE-ID THRESHOLDS ===
    SIMILARITY_THRESHOLD = 0.36           # Strict: Intra-camera appearance re-identification
    INTER_CAMERA_THRESHOLD = 0.44         # Permissive: Cross-camera re-identification under lighting shifts

    MAX_SPATIAL_DISTANCE = 150
    MAX_TIME_LOST = 12.0                  # Seconds an identity stays eligible for Re-ID after disappearing
    CAMERA_SWITCH_COOLDOWN = 2.0          # Prevents ping-pong switching across overlapping camera views
    AUTO_CLUSTER_TIME_THRESHOLD = 10.0    # Max transition time (s) to merge cameras into the same spatial cluster

    # Periodic global gallery EMA update interval (in frames) for already-mapped local tracks
    FEATURE_UPDATE_INTERVAL = 3

    # === UI & PRESENTATION ===
    DISPLAY_ENABLED = True
    DISPLAY_GRID_VIEW = False             # False = dedicated window/tab per camera; True = single unified grid
    DISPLAY_MAX_FPS = 60                  # Throttles GUI redraws to 60 Hz so X11 never stalls GPU inference
    COLOR_STOPPED = (0, 0, 255)
    COLOR_MOVING = (0, 255, 0)
    FONT_SCALE = 0.5

    # === KAFKA STREAMING ===
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    KAFKA_TOPIC = "meu-topico-de-video"
    KAFKA_GROUP_ID = "smart-builds-consumer"
    KAFKA_TARGET_CAMERA = None
    KAFKA_EXPECTED_CAMERAS = 4

    # === FRAME DIMENSIONS (divisible by 32 for zero-padding YOLO inference) ===
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480

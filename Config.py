# Config.py
class Config:
    """
    Centralized configuration for SmartBuildings Multi-Camera Vision & Re-ID System.
    """

    # === EXECUTION & PERFORMANCE PIPELINE ===
    # Boolean toggle to choose whether to use GPU (CUDA) or force CPU execution
    USE_GPU = True

    # "batched": Centralized Batched Inference + CUDA Graphs + Double-Buffered Pipeline (60+ FPS across 8 cameras)
    # "multiprocess": Legacy 1-OS-process-per-camera with BaseManager socket IPC
    EXECUTION_MODE = "batched"
    USE_CUDA_GRAPHS = True
    MAX_CAMERAS_BATCH = 8
    MAX_DETECTIONS_PER_CAMERA = 28

    # Video file playback behavior:
    # False = process frames as fast as hardware allows (ideal for throughput testing)
    # True  = pace local video files (.mp4) at their native framerate (e.g., 60 FPS)
    VIDEO_FILE_REALTIME_PACING = False
    LOOP_LOCAL_VIDEOS = True

    # === YOLO DETECTION PARAMETERS ===
    YOLO_MODEL_PATH = "models/yolo26n.pt"
    YOLO_CONF_HIGH = 0.28          # High-confidence threshold for primary tracking & appearance updates
    YOLO_CONF_LOW = 0.18           # Low-confidence threshold for 2nd-stage occlusion/crowd recovery (ByteTrack-style)
    YOLO_INIT_CONF = 0.30          # Minimum confidence required to initiate a brand-new local track
    YOLO_IOU_THRESHOLD = 0.55      # Crowd-aware NMS IoU threshold to preserve close-proximity people

    # Geometric bounding-box validation (filters false positives such as furniture/reflections)
    MIN_BOX_WIDTH = 14
    MIN_BOX_HEIGHT = 28
    MIN_ASPECT_RATIO = 0.14        # min (width / height)
    MAX_ASPECT_RATIO = 2.85        # max (width / height) - supports close-up/seated/upper-body views

    # === DEEPSORT TRACKING PARAMETERS ===
    DEEPSORT_MAX_AGE = 90          # Max frames an occluded local track is kept before handing off to Global Re-ID
    DEEPSORT_N_INIT = 2            # Consecutive hits required to confirm a tentative track
    DEEPSORT_MAX_IOU_DISTANCE = 0.62
    DEEPSORT_MAX_COSINE_DISTANCE = 0.30
    DEEPSORT_NN_BUDGET = 35        # Bounded appearance gallery per local track
    DEEPSORT_MAX_COAST_FRAMES = 2  # Max frames to render damped prediction during brief detector/occlusion blink
    DEEPSORT_BOX_SMOOTH_ALPHA = 0.65  # EMA smoothing factor for bounding box stability (1.0 = no smoothing)

    # === STOPPED / LOITERING BEHAVIOR THRESHOLDS ===
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30

    # === BIFURCATED GLOBAL RE-ID THRESHOLDS ===
    SIMILARITY_THRESHOLD = 0.32           # Intra-camera appearance re-identification
    INTER_CAMERA_THRESHOLD = 0.36         # Cross-camera re-identification after leaving previous camera
    SIMULTANEOUS_INTER_CAMERA_THRESHOLD = 0.30  # Cross-camera threshold when identity is recently seen in another camera

    MAX_SPATIAL_DISTANCE = 160
    MAX_TIME_LOST = 120.0                 # Seconds an identity stays eligible for Re-ID after disappearing
    CAMERA_SWITCH_COOLDOWN = 0.8          # Anti-ping-pong cooldown for overlapping camera views
    AUTO_CLUSTER_TIME_THRESHOLD = 15.0    # Max transition time (s) to merge cameras into the same spatial cluster

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
    KAFKA_BOOTSTRAP_SERVERS = "10.145.80.61:9092"
    KAFKA_TOPIC = "meu-topico-de-video"
    KAFKA_GROUP_ID = "smart-builds-consumer"
    KAFKA_TARGET_CAMERA = None
    KAFKA_EXPECTED_CAMERAS = 4

    # === FRAME DIMENSIONS (divisible by 32 for zero-padding YOLO inference) ===
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480

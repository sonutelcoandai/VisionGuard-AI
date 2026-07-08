import os

MODEL_PATH = "yolov8n.pt"

IP_CAMERA_URL = 0

LOG_FILE = "detections_log.csv"

SAMPLE_FILE = "popular_times_sample.csv"

KNOWN_DIR = "known_faces"

KNOWN_IMAGES_DIR = os.path.join(
    KNOWN_DIR,
    "images"
)

FACES_CSV = os.path.join(
    KNOWN_DIR,
    "faces.csv"
)

AGE_MODEL_PROTO = "models/deploy_age.prototxt"

AGE_MODEL_BIN = "models/age_net.caffemodel"

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

GENDER_MODEL_PROTO = "models/deploy_gender.prototxt"

GENDER_MODEL_BIN = "models/gender_net.caffemodel"

GENDER_LIST = [
    "Male",
    "Female"
]

FACE_MATCH_THRESHOLD = 0.45

FACE_DETECTOR_MODEL = "hog"

FACE_UPSAMPLE = 1

ENCODER_JITTERS = 1

ENCODER_MODEL = "small"

GENDER_CONF_THRESH = 0.60

MIN_FACE_SIDE_PX = 80

PAD_RATIO = 0.12

QUEUE_THRESH = 5

CROWD_THRESH = 10

QUEUE_HOLD_SEC = 5

CROWD_HOLD_SEC = 5

ZONES = {
    "entrance": [(20,680),(620,680),(620,720),(20,720)],
    "exit": [(660,680),(1260,680),(1260,720),(660,720)],
    "billing": [(900,300),(1250,300),(1250,650),(900,650)],
    "crowd1": [(50,50),(600,50),(600,400),(50,400)],
    "cloth": [(0,0),(1280,0),(1280,720),(0,720)]
}

COUNT_LINES = {
    "entry": (320,660,320,720),
    "exit": (960,660,960,720)
}
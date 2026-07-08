import os

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from app.core.config import (
    MODEL_PATH,
    AGE_MODEL_PROTO,
    AGE_MODEL_BIN,
    GENDER_MODEL_PROTO,
    GENDER_MODEL_BIN
)


class ModelManager:

    def __init__(self):

        self.yolo = None
        self.age_net = None
        self.gender_net = None

    def load_models(self):

        print("Loading AI Models...")

        if YOLO is not None:
            try:
                self.yolo = YOLO(MODEL_PATH)
            except Exception as e:
                print("YOLO model load failed:", e)

        if cv2 is not None:

            try:
                if (
                    os.path.exists(AGE_MODEL_PROTO)
                    and
                    os.path.exists(AGE_MODEL_BIN)
                ):
                    self.age_net = cv2.dnn.readNetFromCaffe(
                        AGE_MODEL_PROTO,
                        AGE_MODEL_BIN
                    )
            except Exception:
                pass

            try:
                if (
                    os.path.exists(GENDER_MODEL_PROTO)
                    and
                    os.path.exists(GENDER_MODEL_BIN)
                ):
                    self.gender_net = cv2.dnn.readNetFromCaffe(
                        GENDER_MODEL_PROTO,
                        GENDER_MODEL_BIN
                    )
            except Exception:
                pass


model_manager = ModelManager()
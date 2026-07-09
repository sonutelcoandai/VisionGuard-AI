try:
    import cv2
except Exception:
    cv2 = None

from app.core.config import (
    GENDER_LIST,
    GENDER_CONF_THRESH
)


class GenderAnalyzer:

    def __init__(self, gender_net):

        self.gender_net = gender_net

    def predict(self, face_roi):

        if (
            self.gender_net is None
            or face_roi is None
            or cv2 is None
        ):
            return None, None

        try:

            blob = cv2.dnn.blobFromImage(
                face_roi,
                1.0,
                (227, 227),
                (78, 87, 114),
                swapRB=False
            )

            self.gender_net.setInput(blob)

            scores = self.gender_net.forward()[0]

            idx = int(scores.argmax())

            conf = float(scores[idx])

            gender = GENDER_LIST[idx]

            if conf < GENDER_CONF_THRESH:
                gender = "Unknown"

            return gender, conf

        except Exception:

            return None, None
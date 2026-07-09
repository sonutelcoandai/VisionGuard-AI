try:
    import cv2
except Exception:
    cv2 = None

from app.core.config import (
    AGE_LIST
)


class AgeAnalyzer:

    def __init__(self, age_net):

        self.age_net = age_net

    def predict(self, face_roi):

        if (
            self.age_net is None
            or face_roi is None
            or cv2 is None
        ):
            return None

        try:

            blob = cv2.dnn.blobFromImage(
                face_roi,
                1.0,
                (227, 227),
                (78, 87, 114),
                swapRB=False
            )

            self.age_net.setInput(blob)

            scores = self.age_net.forward()[0]

            idx = int(scores.argmax())

            return AGE_LIST[idx]

        except Exception:

            return None
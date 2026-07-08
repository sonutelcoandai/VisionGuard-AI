import os

import pandas as pd

try:
    import face_recognition
except Exception:
    face_recognition = None

from app.core.config import (
    FACES_CSV,
    KNOWN_IMAGES_DIR,
    ENCODER_JITTERS,
    ENCODER_MODEL
)


class FaceManager:

    def ensure_faces_csv(self):

        os.makedirs(
            KNOWN_IMAGES_DIR,
            exist_ok=True
        )

        if (
            not os.path.exists(FACES_CSV)
            or os.path.getsize(FACES_CSV) == 0
        ):

            pd.DataFrame(
                columns=[
                    "filename",
                    "name",
                    "categories"
                ]
            ).to_csv(
                FACES_CSV,
                index=False
            )

    def load_known_faces(self):

        self.ensure_faces_csv()

        encodings = []
        meta = []

        try:

            df = pd.read_csv(
                FACES_CSV
            )

        except Exception:

            df = pd.DataFrame(
                columns=[
                    "filename",
                    "name",
                    "categories"
                ]
            )

        for _, row in df.iterrows():

            filename = str(
                row.get("filename") or ""
            ).strip()

            if not filename:
                continue

            image_path = os.path.join(
                KNOWN_IMAGES_DIR,
                filename
            )

            if not os.path.exists(
                image_path
            ):
                continue

            try:

                if face_recognition is None:
                    continue

                image = face_recognition.load_image_file(
                    image_path
                )

                face_encodings = (
                    face_recognition.face_encodings(
                        image,
                        num_jitters=ENCODER_JITTERS,
                        model=ENCODER_MODEL
                    )
                )

                if not face_encodings:
                    continue

                encodings.append(
                    face_encodings[0]
                )

                categories = str(
                    row.get("categories") or ""
                )

                meta.append(
                    {
                        "name": str(
                            row.get("name")
                            or "Unknown"
                        ),
                        "categories": (
                            categories.split(";")
                            if categories
                            else []
                        )
                    }
                )

            except Exception:
                continue

        return encodings, meta


face_manager = FaceManager()
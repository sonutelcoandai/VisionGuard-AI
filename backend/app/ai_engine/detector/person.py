class PersonDetector:

    def __init__(self, model):

        self.model = model

    def detect(self, frame):

        if self.model is None:
            return None, []

        try:

            results = self.model(
                frame,
                verbose=False
            )

        except Exception as e:

            print(
                "YOLO inference error:",
                e
            )

            return None, []

        boxes = None

        try:

            if results and len(results) > 0:
                boxes = results[0].boxes

        except Exception:
            boxes = None

        return boxes, results

    def extract_persons(
        self,
        frame,
        boxes
    ):

        persons = []

        if not boxes:
            return persons

        for box in boxes:

            try:

                cls = int(box.cls[0])

                name_cls = (
                    self.model.names[cls]
                    if (
                        hasattr(
                            self.model,
                            "names"
                        )
                        and cls in self.model.names
                    )
                    else str(cls)
                )

                if (
                    str(name_cls).lower()
                    != "person"
                ):
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                crop_bgr = frame[
                    y1:y2,
                    x1:x2
                ]

                if crop_bgr.size == 0:
                    continue

                persons.append(
                    {
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "crop": crop_bgr
                    }
                )

            except Exception:
                continue

        return persons
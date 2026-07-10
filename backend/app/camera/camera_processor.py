import gc
import time

from datetime import datetime

from app.core.config import *

from app.camera.stream import (
    open_capture,
    blank_jpeg
)

from app.services.alert_service import (
    alert_service
)

from app.core.logger import (
    write_log_row
)

from app.bootstrap.ai_bootstrap import (
    ai_container
)

from app.state.app_state import (
    app_state
)

from app.ai_engine.analytics.behavior import (
    bbox_center,
    crosses_line,
    zone_of_point,
    point_in_poly
)

try:
    import cv2
except Exception:
    cv2 = None


class CameraProcessor:

    def generate_frames(self):

        cap = open_capture(
            IP_CAMERA_URL
        )

        if cap is None or not cap.isOpened():

            while True:

                yield blank_jpeg(
                    "Camera not available"
                )

                time.sleep(1.0)

        miss = 0

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:

                miss += 1

                if miss >= 10:

                    try:
                        cap.release()
                    except Exception:
                        pass

                    time.sleep(0.5)

                    cap = open_capture(
                        IP_CAMERA_URL
                    )

                    miss = 0

                    if cap is None or not cap.isOpened():

                        yield blank_jpeg(
                            "Reconnecting..."
                        )

                        time.sleep(1.0)

                        continue

                else:

                    yield blank_jpeg(
                        "Reconnecting..."
                    )

                    time.sleep(0.2)

                    continue

            else:

                miss = 0

            detections = []

            det_list = []

            boxes, results = (
                ai_container.person_detector.detect(
                    frame
                )
            )

            if boxes:

                persons = (
                    ai_container.person_detector.extract_persons(
                        frame,
                        boxes
                    )
                )

                for person in persons:

                    try:

                        x1, y1, x2, y2 = (
                            person["bbox"]
                        )

                        crop_bgr = (
                            person["crop"]
                        )

                        (
                            person_name,
                            cats,
                            face_roi
                        ) = (
                            ai_container.identity_recognizer.recognize(
                                crop_bgr
                            )
                        )

                        age_bucket = None

                        gender = None

                        gender_conf = None

                        if (
                            face_roi is not None
                            and
                            face_roi.size > 0
                            and
                            min(face_roi.shape[:2]) >= MIN_FACE_SIDE_PX
                        ):

                            try:

                                age_bucket = (
                                    ai_container.age_analyzer.predict(
                                        face_roi
                                    )
                                )

                            except Exception:

                                age_bucket = None

                            try:

                                (
                                    gender,
                                    gender_conf
                                ) = (
                                    ai_container.gender_analyzer.predict(
                                        face_roi
                                    )
                                )

                            except Exception:

                                gender = None
                                gender_conf = None

                        parts = [
                            person_name
                        ] if person_name else [
                            "Unknown"
                        ]

                        if gender:
                            parts.append(
                                gender
                            )

                        if age_bucket:
                            parts.append(
                                age_bucket
                            )

                        label = " ".join(
                            parts
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(
                                    y1 - 6,
                                    10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                        z = zone_of_point(
                            bbox_center(
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                )
                            )
                        )

                        det = {
                            "name": person_name,
                            "categories": cats,
                            "zone": z,
                            "img": crop_bgr.copy(),
                            "face": face_roi,
                            "age": age_bucket,
                            "gender": gender,
                            "bbox": (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        }

                        detections.append(
                            det
                        )

                        write_log_row({
                            "timestamp":
                                datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            "name":
                                person_name,
                            "categories":
                                ";".join(cats)
                                if cats else "",
                            "zone":
                                z,
                            "alert":
                                "",
                            "gender":
                                gender or "",
                            "gender_conf":
                                f"{gender_conf:.2f}"
                                if gender_conf else "",
                            "age_bucket":
                                age_bucket or "",
                            "detection_type":
                                "person"
                        })

                        cats_norm = [
                            str(c)
                            .strip()
                            .lower()
                            for c in cats
                        ] if cats else []

                        if "vip" in cats_norm:

                            info = (
                                f"VIP detected: "
                                f"{person_name} "
                                f"in {z}."
                            )

                            alert_service.add_alert(
                                "vip",
                                info,
                                zone=z,
                                name=person_name
                            )

                        if "highbuyer" in cats_norm:

                            info = (
                                f"High-buyer candidate: "
                                f"{person_name} "
                                f"in {z}."
                            )

                            alert_service.add_alert(
                                "highbuyer",
                                info,
                                zone=z,
                                name=person_name
                            )

                        cx, cy = bbox_center(
                            (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        )

                        det_list.append({
                            "center":
                                (
                                    cx,
                                    cy
                                ),
                            "bbox":
                                (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),
                            "gender":
                                gender,
                            "age":
                                age_bucket
                        })

                    except Exception:

                        continue

            tracks = (
                ai_container.crowd_tracker.update_tracks(
                    det_list
                )
            )

            ai_container.crowd_analytics.process(
                tracks=tracks,
                entered_hourly=app_state.entered_hourly,
                exited_hourly=app_state.exited_hourly,
                count_lines=COUNT_LINES,
                zones=ZONES,
                queue_thresh=QUEUE_THRESH,
                crowd_thresh=CROWD_THRESH,
                point_in_poly=point_in_poly,
                crosses_line=crosses_line,
                alert_service=alert_service
            )

            app_state.last_detections = (
                detections[:16]
            )

            if cv2 is not None:

                _, buffer = cv2.imencode(
                    ".jpg",
                    frame
                )

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

            else:

                yield blank_jpeg(
                    "No OpenCV"
                )

            gc.collect()


camera_processor = CameraProcessor()
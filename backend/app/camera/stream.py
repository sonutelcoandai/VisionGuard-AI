import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def open_capture(src):

    if cv2 is None:
        return None

    try:

        if (
            isinstance(src, int)
            or (
                isinstance(src, str)
                and str(src).isdigit()
            )
        ):

            cap = cv2.VideoCapture(
                int(src),
                apiPreference=cv2.CAP_V4L2
            )

            if not cap.isOpened():

                cap.release()

                cap = cv2.VideoCapture(
                    int(src),
                    apiPreference=cv2.CAP_ANY
                )

            if cap.isOpened():

                cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    1280
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    720
                )

                return cap

        else:

            os.environ[
                "OPENCV_FFMPEG_CAPTURE_OPTIONS"
            ] = (
                "rtsp_transport;tcp|"
                "stimeout;20000000|"
                "buffer_size;102400"
            )

            cap = cv2.VideoCapture(
                src,
                apiPreference=cv2.CAP_FFMPEG
            )

            return cap

    except Exception as e:

        print(
            "open_capture error:",
            e
        )

    return None


def blank_jpeg(
    text="Camera not available"
):

    if cv2 is None:
        return b""

    blank = np.zeros(
        (360, 640, 3),
        dtype=np.uint8
    )

    cv2.putText(
        blank,
        text,
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    ok, buf = cv2.imencode(
        ".jpg",
        blank
    )

    if not ok:
        return b""

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + buf.tobytes()
        + b"\r\n"
    )

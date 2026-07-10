import hashlib

from flask import (
    make_response
)

from app.state.app_state import (
    app_state
)

try:
    import cv2
except Exception:
    cv2 = None
from flask import (
    Blueprint,
    Response
)

cameras_bp = Blueprint(
    "cameras",
    __name__
)

generate_frames_func = None


@cameras_bp.route("/video1")
def video1():

    return Response(
        generate_frames_func(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@cameras_bp.route("/current_img/<int:idx>")
def current_img(idx):

    if idx < 0 or idx >= len(
        app_state.last_detections
    ):
        return make_response("", 404)

    img = (
        app_state
        .last_detections[idx]
        .get("img")
    )

    if img is None:
        return make_response("", 404)

    ok, buf = cv2.imencode(
        ".jpg",
        img
    )

    if not ok:
        return make_response("", 500)

    return make_response(
        buf.tobytes(),
        200,
        {
            "Content-Type":
            "image/jpeg"
        }
    )


@cameras_bp.route(
    "/current_img_id/<string:img_id>"
)
def current_img_id(img_id):

    for d in app_state.last_detections:

        try:

            if (
                hashlib.md5(
                    str(id(d)).encode()
                ).hexdigest()[:8]
                ==
                img_id
            ):

                img = d.get("img")

                if img is None:
                    return make_response(
                        "",
                        404
                    )

                ok, buf = cv2.imencode(
                    ".jpg",
                    img
                )

                if not ok:
                    return make_response(
                        "",
                        500
                    )

                return make_response(
                    buf.tobytes(),
                    200,
                    {
                        "Content-Type":
                        "image/jpeg"
                    }
                )

        except Exception:

            continue

    return make_response("", 404)
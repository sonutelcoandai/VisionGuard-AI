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
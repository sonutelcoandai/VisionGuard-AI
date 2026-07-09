from flask import Blueprint

users_bp = Blueprint(
    "users",
    __name__
)

upload_image_func = None
upload_csv_func = None
download_faces_csv_func = None


@users_bp.route(
    "/upload_image",
    methods=["POST"]
)
def upload_image():

    return upload_image_func()


@users_bp.route(
    "/upload_csv",
    methods=["POST"]
)
def upload_csv():

    return upload_csv_func()


@users_bp.route(
    "/download_faces_csv"
)
def download_faces_csv():

    return download_faces_csv_func()
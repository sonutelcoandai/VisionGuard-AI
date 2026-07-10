import os
import uuid
import pandas as pd

from flask import (
    request,
    jsonify,
    redirect,
    url_for,
    send_file
)

from werkzeug.utils import secure_filename

from app.core.config import (
    KNOWN_IMAGES_DIR,
    FACES_CSV
)

from app.ai_engine.recognition.face import (
    face_manager
)


class FaceService:

    def upload_image(self):

        try:

            file = request.files.get("image")

            name = (request.form.get("name") or "").strip()

            categories = (
                request.form.get("categories") or ""
            ).strip()

            if not file or not name:

                return jsonify({
                    "ok": False,
                    "msg": "invalid"
                }), 400

            fn = secure_filename(
                file.filename
            )

            out_fn = (
                f"{uuid.uuid4().hex}_{fn}"
            )

            out_path = os.path.join(
                KNOWN_IMAGES_DIR,
                out_fn
            )

            file.save(out_path)

            if os.path.exists(FACES_CSV):

                df = pd.read_csv(FACES_CSV)

            else:

                df = pd.DataFrame(
                    columns=[
                        "filename",
                        "name",
                        "categories"
                    ]
                )

            df.loc[len(df)] = {
                "filename": out_fn,
                "name": name,
                "categories": categories
            }

            df.to_csv(
                FACES_CSV,
                index=False
            )

            face_manager.load_known_faces()

            return redirect(
                url_for("index")
            )

        except Exception as e:

            return jsonify({
                "ok": False,
                "msg": str(e)
            }), 500

    def upload_csv(self):

        try:

            file = request.files.get("file")

            if not file:

                return jsonify({
                    "ok": False,
                    "msg": "no file"
                }), 400

            df = pd.read_csv(file)

            for col in [
                "filename",
                "name",
                "categories"
            ]:

                if col not in df.columns:

                    return jsonify({
                        "ok": False,
                        "msg": "missing columns"
                    }), 400

            df.to_csv(
                FACES_CSV,
                index=False
            )

            face_manager.load_known_faces()

            return redirect(
                url_for("index")
            )

        except Exception as e:

            return jsonify({
                "ok": False,
                "msg": str(e)
            }), 500

    def download_faces_csv(self):

        face_manager.ensure_faces_csv()

        return send_file(
            FACES_CSV,
            as_attachment=True,
            download_name="faces.csv"
        )


face_service = FaceService()
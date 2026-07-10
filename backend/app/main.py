#!/usr/bin/env python3
# ipcam_yolo4_final.py
# Single-file app: live YOLO detections -> detections_log.csv
# Analytics: Google-style "Popular Times" (09:00-21:00), current hour highlighted (vertical bars)
# Alerts persisted to CSV and shown on Home

import os, time, gc, uuid, math, hashlib
from collections import deque, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, Response, request, redirect, url_for, jsonify, send_file, make_response, render_template
from werkzeug.utils import secure_filename

# added from top------main script from Here working fine
from app.core.logger import (
    ensure_log_headers
)

from app.ai_engine.recognition.face import (
    face_manager
)

from app.services.alert_service import (
    alert_service
)

from app.api.cameras import (
    cameras_bp
)
import app.api.cameras as cameras_module

from app.api.events import (
    events_bp
)
import app.api.events as events_module

from app.api.users import (
    users_bp
)
import app.api.users as users_module

from app.state.app_state import (
    app_state
)

from app.services.face_service import (
    face_service
)

from app.bootstrap.ai_bootstrap import (
    ai_container
)
from app.services.event_service import (
    event_service
)
from app.services.analytics_service import (
    analytics_service
)
from app.camera.camera_processor import (
    camera_processor
)

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import face_recognition
except Exception:
    face_recognition = None
# ---------------- CONFIG ----------------

from app.core.config import *
os.makedirs(
    KNOWN_IMAGES_DIR,
    exist_ok=True
)

os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(SAMPLE_FILE),
    exist_ok=True
)

from app.ai_engine.recognition.face import (
    face_manager
)
os.makedirs(KNOWN_IMAGES_DIR, exist_ok=True)

# ---------- faces csv ----------

face_manager.ensure_faces_csv()
# ---------- load models ----------
from app.ai_engine.model_manager import (
    model_manager
)
# below line for test
alert_service.add_alert(
    "test",
    "Test alert generated during startup"
)
#above lines for test
# ---------- known faces ----------

# ---------- logging ----------

ensure_log_headers()


# ---------- helpers ----------
#added this code to behaviour.py
# ---------- state ----------
#moved these lines to app_state.py

# ---------- capture & detection ----------

cameras_module.generate_frames_func = (
    camera_processor.generate_frames
)


# ---------- HTML (single-file SPA) ----------

# this HTML code moved to /template/index.html

# ---------- Flask endpoints ----------
app = Flask(__name__, static_folder="static")

app.register_blueprint(
    cameras_bp
)

app.register_blueprint(
    events_bp
)

app.register_blueprint(
    users_bp
)



# serve SPA on common routes to avoid 404 on direct navigation
@app.route("/")
@app.route("/analytics")
@app.route("/inventory")
@app.route("/offers")
@app.route("/dashboard")
@app.route("/wifi")
def index():
    return render_template(
        "index.html"
    )

@app.route("/current_img/<int:idx>")
def current_img(idx):
    # keep old endpoint (index-based) for backward compatibility
    if idx < 0 or idx >= len(app_state.last_detections):
        return make_response("", 404)
    img = app_state.last_detections[idx].get("img")
    if img is None:
        return make_response("", 404)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return make_response("", 500)
    return make_response(buf.tobytes(), 200, {"Content-Type":"image/jpeg"})

@app.route("/current_img_id/<string:img_id>")
def current_img_id(img_id):
    # new: return image by stable id generated in /current
    for d in app_state.last_detections:
        try:
            if hashlib.md5(str(id(d)).encode()).hexdigest()[:8] == img_id:
                img = d.get("img")
                if img is None:
                    return make_response("", 404)
                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    return make_response("", 500)
                return make_response(buf.tobytes(), 200, {"Content-Type":"image/jpeg"})
        except Exception:
            continue
    return make_response("", 404)

#created helper in 6B
#end helper



events_module.last_detections_ref = (
    app_state.last_detections
)

events_module.alert_service_ref = (
    event_service.get_alerts
)

events_module.analytics_func = (
    analytics_service.build
)

users_module.upload_image_func = (
    face_service.upload_image
)

users_module.upload_csv_func = (
    face_service.upload_csv
)

users_module.download_faces_csv_func = (
    face_service.download_faces_csv
)

# ---------- start ----------
if __name__ == "__main__":
    print("Starting ipcam_yolo4_final at http://0.0.0.0:5000")
    # placeholder image for faces
    placeholder = os.path.join("static","faces","placeholder.png")
    os.makedirs(os.path.dirname(placeholder), exist_ok=True)
    if not os.path.exists(placeholder) and cv2 is not None:
        try:
            p = 255*np.ones((120,120,3), dtype=np.uint8)
            cv2.putText(p, "No Image", (6,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.imwrite(placeholder, p)
        except Exception:
            pass
    app.run(host="0.0.0.0", port=5000, threaded=True)

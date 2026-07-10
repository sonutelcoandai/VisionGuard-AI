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
    ensure_log_headers,
    write_log_row
)

from app.ai_engine.recognition.face import (
    face_manager
)

from app.services.alert_service import (
    alert_service
)
from app.camera.stream import (
    open_capture,
    blank_jpeg
)
from app.ai_engine.analytics.behavior import (
    point_in_poly,
    bbox_center,
    crosses_line,
    zone_of_point
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

from app.core.logger import (
    ensure_log_headers,
    write_log_row
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

def generate_frames():
    global tracks, next_tid
    cap = open_capture(IP_CAMERA_URL)
    if cap is None or not cap.isOpened():
        while True:
            yield blank_jpeg("Camera not available")
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
                cap = open_capture(IP_CAMERA_URL)
                miss = 0
                if cap is None or not cap.isOpened():
                    yield blank_jpeg("Reconnecting...")
                    time.sleep(1.0)
                    continue
            else:
                yield blank_jpeg("Reconnecting...")
                time.sleep(0.2)
                continue
        else:
            miss = 0

        detections = []
        det_list = []
        boxes, results = ai_container.person_detector.detect(frame)
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
                    age_bucket, gender, gender_conf = None, None, None
                    if face_roi is not None and face_roi.size>0 and min(face_roi.shape[:2]) >= MIN_FACE_SIDE_PX:
                        try:
                            
                              age_bucket = (
    ai_container
    .age_analyzer
    .predict(face_roi)
)
                        except Exception:
                            age_bucket = None
                        try:
                            gender, gender_conf = (ai_container.gender_analyzer.predict(face_roi))
                        except Exception:
                            gender, gender_conf = None, None

                    parts = [person_name] if person_name else ["Unknown"]
                    if gender: parts.append(gender)
                    if age_bucket: parts.append(age_bucket)
                    label = " ".join(parts)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, max(y1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                    z = zone_of_point(bbox_center((x1,y1,x2,y2)))
                    det = {"name": person_name, "categories": cats, "zone": z, "img": crop_bgr.copy(), "face": face_roi, "age": age_bucket, "gender": gender, "bbox": (x1,y1,x2,y2)}
                    detections.append(det)
                    write_log_row({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "name": person_name,
                        "categories": ";".join(cats) if cats else "",
                        "zone": z,
                        "alert": "",
                        "gender": gender or "",
                        "gender_conf": f"{gender_conf:.2f}" if gender_conf else "",
                        "age_bucket": age_bucket or "",
                        "detection_type": "person"
                    })
                    cats_norm = [str(c).strip().lower() for c in cats] if cats else []
                    if "vip" in cats_norm:
                        info = f"VIP detected: {person_name} in {z}. Suggest priority assistance."
                        alert_service.add_alert("vip", info, zone=z, name=person_name)
                    if "highbuyer" in cats_norm:
                        info = f"High-buyer candidate: {person_name} in {z}."
                        alert_service.add_alert("highbuyer", info, zone=z, name=person_name)
                    cx,cy = bbox_center((x1,y1,x2,y2))
                    det_list.append({"center": (cx,cy), "bbox": (x1,y1,x2,y2), "gender": gender, "age": age_bucket})
                except Exception:
                    continue

        # tracking update
        tracks = (
            ai_container.crowd_tracker.update_tracks(
                det_list
            )
        )

        # line crossing counts
        

        # queue & crowd detection
        crowd_stats = (
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
)
        app_state.last_detections = detections[:16]
        if cv2 is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            yield blank_jpeg("No OpenCV")
        gc.collect()
cameras_module.generate_frames_func = (
    generate_frames
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

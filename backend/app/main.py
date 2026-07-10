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
                            gender, gender_conf = (gender_analyzer.predict(face_roi))
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



def _build_analytics_from_sample(sample_df):
    """ sample_df expected to contain columns: date (YYYY-MM-DD), hour (HH:00), visitors (int), wait_time (optional numeric), zone (optional), gender (optional), age_bucket (optional)
    We'll aggregate per-day and per-hour.
    """
    sample_df = sample_df.copy()
    if 'date' not in sample_df.columns:
        sample_df['date'] = datetime.now().date().isoformat()
    if 'hour' not in sample_df.columns and 'timestamp' in sample_df.columns:
        sample_df['hour'] = pd.to_datetime(sample_df['timestamp'], errors='coerce').dt.strftime('%H:00')
    if 'visitors' not in sample_df.columns:
        # try to infer from single detections: count rows per hour
        sample_df['visitors'] = 1
    # ensure hour is in HH:00 format
    sample_df['hour'] = sample_df['hour'].astype(str).str.slice(0,5)
    days = sorted(sample_df['date'].dropna().unique())
    hours = [f"{str(h).zfill(2)}:00" for h in range(9,22)]
    visitors_per_day = {}
    avg_wait_per_day = {}
    zones_per_day = {}
    gender_counts_per_day = {}
    age_buckets_per_day = {}
    age_gender_map = {}
    for d in days:
        day_df = sample_df[sample_df['date']==d]
        visitors_per_day[d] = [int(day_df[day_df['hour']==h]['visitors'].sum()) for h in hours]
        # avg wait per hour if provided (analytics endpoint may return avg_wait_per_hour)
        if 'wait_time' in day_df.columns:
            avg_wait_per_day[d] = [float(day_df[day_df['hour']==h]['wait_time'].mean() or 0) for h in hours]
        else:
            avg_wait_per_day[d] = [0 for _ in hours]
        # zones
        if 'zone' in day_df.columns:
            zcounts = day_df['zone'].fillna('').replace('', 'Cloth Section')
            zones_per_day[d] = zcounts.value_counts().to_dict()
        else:
            zones_per_day[d] = {}
        # gender
        if 'gender' in day_df.columns:
            gender_counts_per_day[d] = day_df['gender'].fillna('Unknown').replace('', 'Unknown').value_counts().to_dict()
        else:
            gender_counts_per_day[d] = {}
        # age buckets and age x gender map
        if 'age_bucket' in day_df.columns and 'gender' in day_df.columns:
            ag = day_df.groupby(['gender','age_bucket']).size().unstack(fill_value=0)
            age_buckets_per_day[d] = ag.sum(axis=0).to_dict()
            age_gender_map[d] = {'age_labels': list(ag.columns)}
            for g in ag.index:
                age_gender_map[d][g] = ag.loc[g].to_dict()
        else:
            age_buckets_per_day[d] = {}
            age_gender_map[d] = None
    return {
        "days": days,
        "hours": hours,
        "visitors_per_day": visitors_per_day,
        "avg_wait_per_hour": avg_wait_per_day,
        "zones_per_day": zones_per_day,
        "gender_counts_per_day": gender_counts_per_day,
        "age_buckets_per_day": age_buckets_per_day,
        "age_gender_map": age_gender_map
    }
def build_analytics_data():
    """ Return analytics structure used by the frontend. Primary source: LOG_FILE (detections_log.csv) where each detection writes a timestamp and zone/gender/age_bucket. Fallback (manual): SAMPLE_FILE (popular_times_sample.csv) if present and LOG_FILE doesn't have enough data. """
    try:
        # if LOG_FILE exists and contains timestamped detections, prefer it
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE)>0:
            df = pd.read_csv(LOG_FILE)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["date"] = df["timestamp"].dt.date.astype(str)
                df["hour"] = df["timestamp"].dt.strftime("%H:00")
            else:
                df["date"] = ""; df["hour"] = ""
            hours = [f"{str(h).zfill(2)}:00" for h in range(9,22)]
            days = sorted(df["date"].dropna().unique())
            visitors_per_day = {}; gender_counts_per_day = {}; zones_per_day = {}; age_buckets_per_day = {}; age_gender_map = {}; avg_wait_per_hour = {}
            for d in days:
                day_df = df[df["date"]==d]
                # visitors per hour is simply count per hour
                visitors_per_day[d] = [int(day_df[day_df["hour"]==h].shape[0]) for h in hours]
                # zones
                zcounts = day_df["zone"].fillna("").replace("", "Cloth Section").apply(lambda x: (x if x!='unknown' else 'Cloth Section'))
                zones_per_day[d] = zcounts.value_counts().to_dict()
                # gender counts
                gender_counts_per_day[d] = day_df["gender"].fillna("Unknown").replace("", "Unknown").value_counts().to_dict()
                # age buckets distribution
                if "age_bucket" in day_df.columns:
                    age_buckets_per_day[d] = day_df["age_bucket"].fillna("Unknown").replace("", "Unknown").value_counts().to_dict()
                else:
                    age_buckets_per_day[d] = {}
                # age x gender map if both available
                if "age_bucket" in day_df.columns and "gender" in day_df.columns:
                    ag = day_df.groupby(['gender','age_bucket']).size().unstack(fill_value=0)
                    age_gender_map[d] = {'age_labels': list(ag.columns)}
                    for g in ag.index:
                        age_gender_map[d][g] = ag.loc[g].to_dict()
                else:
                    age_gender_map[d] = None
                # avg wait per hour: if log contains 'wait_time' column (rare), use it; else zeros
                if "wait_time" in day_df.columns:
                    avg_wait_per_hour[d] = [float(day_df[day_df["hour"]==h]["wait_time"].mean() or 0) for h in hours]
                else:
                    avg_wait_per_hour[d] = [0 for _ in hours]
            return jsonify({
                "days": list(days),
                "hours": hours,
                "visitors_per_day": visitors_per_day,
                "gender_counts_per_day": gender_counts_per_day,
                "zones_per_day": zones_per_day,
                "age_buckets_per_day": age_buckets_per_day,
                "age_gender_map": age_gender_map,
                "avg_wait_per_hour": avg_wait_per_hour
            })
        else:
            # fallback: if SAMPLE_FILE exists (manual), load it to synthesize analytics
            if os.path.exists(SAMPLE_FILE):
                sample_df = pd.read_csv(SAMPLE_FILE)
                out = _build_analytics_from_sample(sample_df)
                return jsonify(out)
        # no data available
        return jsonify({"days":[], "hours":[], "visitors_per_day":{}, "gender_counts_per_day":{}, "zones_per_day":{}, "age_buckets_per_day":{}, "age_gender_map":{}, "avg_wait_per_hour":{}})
    except Exception as e:
        print("analytics_data error:", e)
        return jsonify({"days":[], "hours":[], "visitors_per_day":{}, "gender_counts_per_day":{}, "zones_per_day":{}, "age_buckets_per_day":{}, "age_gender_map":{}, "avg_wait_per_hour":{}})
events_module.last_detections_ref = (
    app_state.last_detections
)

events_module.alert_service_ref = (
    event_service.get_alerts
)

events_module.analytics_func = (
    build_analytics_data
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

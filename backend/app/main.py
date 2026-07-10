#!/usr/bin/env python3

"""
VisionGuard-AI
==============

Application Entry Point

Responsibilities:
-----------------
- Initialize application resources
- Register Flask blueprints
- Wire services and APIs together
- Serve dashboard UI
- Start Flask server

Business logic, AI processing, analytics, tracking,
camera streaming and alert generation are intentionally
kept outside this file.

Those responsibilities are implemented in:

- bootstrap/ai_bootstrap.py
- camera/camera_processor.py
- services/*
- ai_engine/*
- api/*
"""

import os
import numpy as np

from flask import (
    Flask,
    render_template
)

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
# Core Services
# ============================================================

from app.core.logger import (
    ensure_log_headers
)

from app.services.alert_service import (
    alert_service
)

from app.services.face_service import (
    face_service
)

from app.services.event_service import (
    event_service
)

from app.services.analytics_service import (
    analytics_service
)


# ============================================================
# Runtime State
# ============================================================

from app.state.app_state import (
    app_state
)


# ============================================================
# Camera Processing
# ============================================================

from app.camera.camera_processor import (
    camera_processor
)


# ============================================================
# Face Management
# ============================================================

from app.ai_engine.recognition.face import (
    face_manager
)


# ============================================================
# API Blueprints
# ============================================================

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


# ============================================================
# Application Configuration
# ============================================================

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


# ============================================================
# Initial Startup Tasks
# ============================================================

# Ensure face database exists.
face_manager.ensure_faces_csv()

# Ensure detection log contains required headers.
ensure_log_headers()

# Optional startup test alert.
# Remove this in production deployments.
alert_service.add_alert(
    "test",
    "VisionGuard-AI startup validation"
)


# ============================================================
# API Wiring
# ============================================================

# Camera streaming endpoint uses the extracted
# CameraProcessor pipeline.
cameras_module.generate_frames_func = (
    camera_processor.generate_frames
)

# Event APIs consume runtime state and services.
events_module.last_detections_ref = (
    app_state.last_detections
)

events_module.alert_service_ref = (
    event_service.get_alerts
)

events_module.analytics_func = (
    analytics_service.build
)

# User APIs delegate to FaceService.
users_module.upload_image_func = (
    face_service.upload_image
)

users_module.upload_csv_func = (
    face_service.upload_csv
)

users_module.download_faces_csv_func = (
    face_service.download_faces_csv
)


# ============================================================
# Flask Application
# ============================================================

app = Flask(
    __name__,
    static_folder="static"
)

# Register application modules.
app.register_blueprint(
    cameras_bp
)

app.register_blueprint(
    events_bp
)

app.register_blueprint(
    users_bp
)


# ============================================================
# Dashboard Routes
# ============================================================

@app.route("/")
@app.route("/analytics")
@app.route("/inventory")
@app.route("/offers")
@app.route("/dashboard")
@app.route("/wifi")
def index():
    """
    Serve the VisionGuard dashboard.

    These routes all render the same SPA entry page
    so browser refreshes on nested routes don't
    generate 404 errors.
    """

    return render_template(
        "index.html"
    )


# ============================================================
# Application Startup
# ============================================================

if __name__ == "__main__":

    print(
        "Starting VisionGuard-AI "
        "at http://0.0.0.0:5000"
    )

    # Create dashboard placeholder image used
    # when a face image is unavailable.
    placeholder = os.path.join(
        "static",
        "faces",
        "placeholder.png"
    )

    os.makedirs(
        os.path.dirname(placeholder),
        exist_ok=True
    )

    if (
        not os.path.exists(placeholder)
        and cv2 is not None
    ):

        try:

            img = 255 * np.ones(
                (120, 120, 3),
                dtype=np.uint8
            )

            cv2.putText(
                img,
                "No Image",
                (6, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

            cv2.imwrite(
                placeholder,
                img
            )

        except Exception:
            pass

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
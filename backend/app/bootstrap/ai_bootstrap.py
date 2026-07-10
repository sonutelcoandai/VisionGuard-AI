from app.ai_engine.model_manager import (
    model_manager
)

from app.ai_engine.analytics.age import (
    AgeAnalyzer
)

from app.ai_engine.analytics.gender import (
    GenderAnalyzer
)

from app.ai_engine.detector.person import (
    PersonDetector
)

from app.ai_engine.recognition.face import (
    face_manager
)

from app.ai_engine.recognition.identity import (
    IdentityRecognizer
)

from app.ai_engine.analytics.crowd import (
    CrowdTracker,
    CrowdAnalytics
)

from app.core.config import (
    FACE_MATCH_THRESHOLD,
    FACE_UPSAMPLE,
    FACE_DETECTOR_MODEL,
    ENCODER_JITTERS,
    ENCODER_MODEL,
    PAD_RATIO
)

try:
    import face_recognition
except Exception:
    face_recognition = None


class AIContainer:

    def __init__(self):

        print("Loading AI models...")

        # -------------------------
        # Load all ML models
        # -------------------------

        model_manager.load_models()

        # -------------------------
        # YOLO Detector
        # -------------------------

        self.person_detector = (
            PersonDetector(
                model_manager.yolo
            )
        )

        # -------------------------
        # Age Analyzer
        # -------------------------

        self.age_analyzer = (
            AgeAnalyzer(
                model_manager.age_net
            )
        )

        # -------------------------
        # Gender Analyzer
        # -------------------------

        self.gender_analyzer = (
            GenderAnalyzer(
                model_manager.gender_net
            )
        )

        # -------------------------
        # Known Faces
        # -------------------------

        self.known_encodings, self.known_meta = (
            face_manager.load_known_faces()
        )

        # -------------------------
        # Face Recognition Engine
        # -------------------------

        self.identity_recognizer = (
            IdentityRecognizer(
                face_recognition,
                self.known_encodings,
                self.known_meta,
                FACE_MATCH_THRESHOLD,
                FACE_UPSAMPLE,
                FACE_DETECTOR_MODEL,
                ENCODER_JITTERS,
                ENCODER_MODEL,
                PAD_RATIO
            )
        )

        # -------------------------
        # Crowd Tracking
        # -------------------------

        self.crowd_tracker = (
            CrowdTracker()
        )

        # -------------------------
        # Crowd Analytics
        # -------------------------

        self.crowd_analytics = (
            CrowdAnalytics()
        )

        print("AI initialization complete.")


ai_container = AIContainer()
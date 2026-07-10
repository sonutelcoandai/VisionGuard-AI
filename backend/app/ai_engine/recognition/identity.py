import cv2


class IdentityRecognizer:

    def __init__(
        self,
        face_recognition_module,
        known_encodings,
        known_meta,
        threshold,
        upsample,
        detector_model,
        encoder_jitters,
        encoder_model,
        pad_ratio
    ):

        self.face_recognition = (
            face_recognition_module
        )

        self.known_encodings = (
            known_encodings
        )

        self.known_meta = (
            known_meta
        )

        self.threshold = threshold

        self.upsample = upsample

        self.detector_model = (
            detector_model
        )

        self.encoder_jitters = (
            encoder_jitters
        )

        self.encoder_model = (
            encoder_model
        )

        self.pad_ratio = pad_ratio

    def recognize(
        self,
        crop_bgr
    ):

        person_name = "Unknown"

        cats = []

        face_roi = None

        if self.face_recognition is None:

            return (
                person_name,
                cats,
                face_roi
            )

        try:

            crop_rgb = cv2.cvtColor(
                crop_bgr,
                cv2.COLOR_BGR2RGB
            )

            locs = (
                self.face_recognition
                .face_locations(
                    crop_rgb,
                    number_of_times_to_upsample=self.upsample,
                    model=self.detector_model
                )
            )

            if not locs:

                return (
                    person_name,
                    cats,
                    face_roi
                )

            encs = (
                self.face_recognition
                .face_encodings(
                    crop_rgb,
                    known_face_locations=locs,
                    num_jitters=self.encoder_jitters,
                    model=self.encoder_model
                )
            )

            if (
                encs
                and self.known_encodings
            ):

                dists = (
                    self.face_recognition
                    .face_distance(
                        self.known_encodings,
                        encs[0]
                    )
                )

                if (
                    len(dists) > 0
                    and min(dists)
                    < self.threshold
                ):

                    idx = int(
                        dists.argmin()
                    )

                    person_name = (
                        self.known_meta[idx]["name"]
                    )

                    cats = (
                        self.known_meta[idx]
                        ["categories"]
                    )

            top, right, bottom, left = (
                locs[0]
            )

            h, w = crop_bgr.shape[:2]

            pad_y = int(
                self.pad_ratio
                * (bottom - top)
            )

            pad_x = int(
                self.pad_ratio
                * (right - left)
            )

            t = max(
                0,
                top - pad_y
            )

            b = min(
                h,
                bottom + pad_y
            )

            l = max(
                0,
                left - pad_x
            )

            r = min(
                w,
                right + pad_x
            )

            if b > t and r > l:

                face_roi = (
                    crop_bgr[
                        t:b,
                        l:r
                    ]
                    .copy()
                )

        except Exception:
            pass

        return (
            person_name,
            cats,
            face_roi
        )
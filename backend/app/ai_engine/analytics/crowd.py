import math
import time


class CrowdTracker:

    def __init__(self):

        self.tracks = {}

        self.next_tid = 1

    def update_tracks(
        self,
        det_list
    ):

        t_now = time.time()

        for tid in list(
            self.tracks.keys()
        ):
            if (
                t_now
                - self.tracks[tid]["last_seen"]
                > 3.0
            ):
                del self.tracks[tid]

        used = set()

        for d in det_list:

            cx, cy = d["center"]

            best_tid = None

            best_dist = 9999

            for tid, tr in (
                self.tracks.items()
            ):

                px, py = tr["now"]

                dist = math.hypot(
                    cx - px,
                    cy - py
                )

                if (
                    dist < 120
                    and dist < best_dist
                    and tid not in used
                ):
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:

                tid = self.next_tid

                self.next_tid += 1

                self.tracks[tid] = {
                    "last": (cx, cy),
                    "now": (cx, cy),
                    "last_seen": t_now,
                    "gender": d.get(
                        "gender"
                    ),
                    "age": d.get("age")
                }

                used.add(tid)

            else:

                tr = self.tracks[
                    best_tid
                ]

                tr["last"] = tr["now"]

                tr["now"] = (
                    cx,
                    cy
                )

                tr["last_seen"] = t_now

                used.add(best_tid)

        return self.tracks
class CrowdAnalytics:

    def process(
        self,
        tracks,
        entered_hourly,
        exited_hourly,
        count_lines,
        zones,
        queue_thresh,
        crowd_thresh,
        point_in_poly,
        crosses_line,
        alert_service
    ):

        from datetime import datetime

        for tid, tr in tracks.items():

            if (
                "last" in tr
                and "now" in tr
            ):

                if crosses_line(
                    tr["last"],
                    tr["now"],
                    count_lines["entry"]
                ):

                    hk = (
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:00"
                        )
                    )

                    entered_hourly.setdefault(
                        hk,
                        {
                            "total": 0,
                            "Male": 0,
                            "Female": 0,
                            "Unknown": 0
                        }
                    )

                    entered_hourly[hk][
                        "total"
                    ] += 1

                if crosses_line(
                    tr["last"],
                    tr["now"],
                    count_lines["exit"]
                ):

                    hk = (
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:00"
                        )
                    )

                    exited_hourly.setdefault(
                        hk,
                        {
                            "total": 0,
                            "Male": 0,
                            "Female": 0,
                            "Unknown": 0
                        }
                    )

                    exited_hourly[hk][
                        "total"
                    ] += 1

        queue_count = sum(
            1
            for tr in tracks.values()
            if (
                tr.get("now")
                and point_in_poly(
                    tr["now"],
                    zones["billing"]
                )
            )
        )

        crowd_count = sum(
            1
            for tr in tracks.values()
            if (
                tr.get("now")
                and point_in_poly(
                    tr["now"],
                    zones["crowd1"]
                )
            )
        )

        if queue_count > queue_thresh:

            alert_service.add_alert(
                "queue_overflow",
                (
                    f"Queue overflow: "
                    f"{queue_count} persons "
                    f"at billing."
                ),
                zone="billing"
            )

        if crowd_count > crowd_thresh:

            alert_service.add_alert(
                "crowd",
                (
                    f"Crowd: "
                    f"{crowd_count} persons."
                ),
                zone="crowd1"
            )

        return {
            "queue_count": queue_count,
            "crowd_count": crowd_count
        }
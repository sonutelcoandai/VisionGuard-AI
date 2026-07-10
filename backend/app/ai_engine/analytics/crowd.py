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
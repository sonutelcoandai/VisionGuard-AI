import os
import hashlib
import pandas as pd

from collections import deque

from app.core.config import LOG_FILE

CSV_COLUMNS = [
    "timestamp",
    "name",
    "categories",
    "zone",
    "alert",
    "gender",
    "gender_conf",
    "age_bucket",
    "detection_type"
]


def ensure_log_headers():

    if (
        not os.path.exists(LOG_FILE)
        or os.path.getsize(LOG_FILE) == 0
    ):

        pd.DataFrame(
            columns=CSV_COLUMNS
        ).to_csv(
            LOG_FILE,
            index=False
        )


def write_log_row(row):

    base = {
        k: ""
        for k in CSV_COLUMNS
    }

    base.update(row)

    sig = hashlib.sha1(
        "|".join(
            str(base.get(k, ""))
            for k in CSV_COLUMNS
        ).encode("utf-8")
    ).hexdigest()

    if not hasattr(
        write_log_row,
        "_recent"
    ):
        write_log_row._recent = deque(
            maxlen=400
        )

    if sig in write_log_row._recent:
        return

    write_log_row._recent.append(sig)

    try:

        pd.DataFrame(
            [base]
        ).to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )

    except Exception as e:

        print(
            "write_log_row error:",
            e
        )
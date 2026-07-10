import os
import pandas as pd

from app.core.config import (
    LOG_FILE
)

from app.services.alert_service import (
    alert_service
)


class EventService:

    def get_alerts(self):

        combined = []

        try:

            if os.path.exists(LOG_FILE):

                df = pd.read_csv(LOG_FILE)

                if "alert" in df.columns:

                    df2 = df[
                        df["alert"].notna()
                        &
                        (
                            df["alert"]
                            .astype(str)
                            != ""
                        )
                    ]

                    df2 = (
                        df2.sort_values(
                            "timestamp",
                            ascending=False
                        )
                        .head(200)
                    )

                    for _, r in df2.iterrows():

                        combined.append({
                            "timestamp":
                                str(
                                    r.get(
                                        "timestamp"
                                    ) or ""
                                ),
                            "alert_type":
                                str(
                                    r.get(
                                        "alert"
                                    ) or ""
                                ),
                            "info":
                                f"name={r.get('name','')} zone={r.get('zone','')}"
                        })

        except Exception as e:

            print(
                "event_service error:",
                e
            )

        for a in alert_service.alerts[:10]:

            combined.insert(
                0,
                a
            )

        seen = set()

        out = []

        for c in combined:

            key = (
                c.get("timestamp", ""),
                c.get("alert_type", ""),
                c.get("info", "")
            )

            if key in seen:

                continue

            seen.add(key)

            out.append(c)

        return out[:10]


event_service = EventService()
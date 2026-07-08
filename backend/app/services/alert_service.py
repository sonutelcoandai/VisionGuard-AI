from datetime import datetime

from app.core.logger import write_log_row


class AlertService:

    def __init__(self):

        self.alerts = []

    def add_alert(
        self,
        alert_type,
        info,
        zone="",
        name=""
    ):
        print("DEBUG ALERT:", alert_type, info)
        rec = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "alert_type": alert_type,
            "info": info,
            "zone": zone,
            "name": name
        }

        self.alerts.insert(
            0,
            rec
        )

        write_log_row({
            "timestamp": rec["timestamp"],
            "name": name or alert_type,
            "zone": zone,
            "alert": alert_type,
            "detection_type": alert_type
        })
alert_service = AlertService()
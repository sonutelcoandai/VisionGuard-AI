from flask import (
    Blueprint,
    jsonify
)

events_bp = Blueprint(
    "events",
    __name__
)

last_detections_ref = None
alert_service_ref = None
analytics_func = None


@events_bp.route("/current")
def current():

    out = []

    for d in last_detections_ref[:16]:

        out.append({
            "id": d.get("id", ""),
            "name": d.get("name", ""),
            "categories": ";".join(
                d.get("categories") or []
            ),
            "zone": d.get("zone", ""),
            "gender": d.get("gender", ""),
            "age": d.get("age", ""),
            "has_img": True if d.get("img") is not None else False,
            "has_face": True if d.get("face") is not None else False
        })

    return jsonify(out)


@events_bp.route("/alerts")
def alerts():

    return jsonify(
        alert_service_ref()
    )


@events_bp.route("/analytics_data")
def analytics_data():

    return analytics_func()
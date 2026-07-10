from flask import (
    Flask,
    render_template
)

from app.api.cameras import (
    cameras_bp
)

from app.api.events import (
    events_bp
)

from app.api.users import (
    users_bp
)


def create_app():

    app = Flask(
        __name__,
        static_folder="static"
    )

    app.register_blueprint(
        cameras_bp
    )

    app.register_blueprint(
        events_bp
    )

    app.register_blueprint(
        users_bp
    )

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

    return app
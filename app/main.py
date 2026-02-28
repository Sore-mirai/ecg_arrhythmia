"""
Flask Application Entry Point
===============================
"""

from flask import Flask, render_template
from flask_cors import CORS

from app.routes.api import api
from app.model.inference import engine


def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    CORS(app)

    # Register API blueprint
    app.register_blueprint(api)

    # Load ML model on startup
    with app.app_context():
        engine.load()

    # ── Main dashboard route ──
    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    return app

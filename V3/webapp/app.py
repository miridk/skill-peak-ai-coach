"""
Flask app factory for the SkillPeak V3 web interface.
Usage:
    from webapp.app import create_app
    app = create_app(outputs_root="V3/outputs")
    app.run(port=5173)
"""

import os
from flask import Flask


def create_app(outputs_root: str) -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir   = os.path.join(os.path.dirname(__file__), "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )
    app.config["OUTPUTS_ROOT"] = os.path.abspath(outputs_root)
    app.config["SECRET_KEY"]   = os.urandom(24)

    from webapp.routes import bp
    app.register_blueprint(bp)

    return app

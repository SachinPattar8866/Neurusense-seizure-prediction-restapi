from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

# Import blueprints
from routes.predict_routes import predict_bp
from routes.rehab_routes import rehab_bp
from routes.health_routes import health_bp
from routes.auth_routes import auth_bp
from routes.history_routes import history_bp
from routes.metrics_routes import metrics_bp

# Import global error handler
from utils.errors import register_error_handlers

# Load environment variables
load_dotenv()


def create_app():
    """
    Factory to create Flask app instance.
    Registers blueprints, configures CORS, and global error handlers.
    """
    app = Flask(__name__)
    # load settings from config module
    app.config.from_object('config.settings')
    # map JWT settings
    # flask-jwt-extended expects JWT_SECRET_KEY
    if app.config.get('JWT_SECRET'):
        app.config['JWT_SECRET_KEY'] = app.config.get('JWT_SECRET')
    # set token expiry if present
    try:
        hours = int(app.config.get('JWT_EXPIRATION_HOURS', 24))
        from datetime import timedelta

        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=hours)
    except Exception:
        pass

    CORS(app)

    # JWT
    jwt = JWTManager(app)

    # Global auth guard for /api/* endpoints: allow only signup & login as public
    from flask import request, jsonify
    from flask_jwt_extended import verify_jwt_in_request, exceptions as jwt_exceptions

    # make model-metrics public so unauthenticated UI can render training stats
    PUBLIC_ENDPOINTS = ['/api/signup', '/api/login', '/api/model-metrics']

    @app.before_request
    def require_auth_for_api():
        path = request.path
        # Only enforce for /api/*
        if not path.startswith('/api'):
            return None
        # Allow public endpoints
        if path in PUBLIC_ENDPOINTS:
            return None
        # For other API endpoints, verify JWT
        try:
            verify_jwt_in_request()
        except Exception as e:
            # any JWT verification error -> 401
            return jsonify({'msg': 'Missing or invalid Authorization token'}), 401

    # Register blueprints with common prefix
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(rehab_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")
    from routes.reports_routes import reports_bp
    app.register_blueprint(reports_bp, url_prefix="/api")

    # Register global error handlers
    register_error_handlers(app)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    try:
        # Disable the Werkzeug reloader to avoid issues on Windows where
        # child/reloader threads can attempt to use closed sockets during shutdown
        # which raises OSError: [WinError 10038]. For development you can set
        # debug=True but keep use_reloader=False to still show debug output.
        app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        print("Shutting down server")

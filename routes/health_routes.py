from flask import Blueprint, jsonify, request
from config.settings import API_VERSION
from utils.logging_utils import log_request

# Create blueprint
health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health():
    """
    GET /health
    Simple health check endpoint to confirm API is alive.
    """
    log_request(request)
    return jsonify({
        "status": "UP",
        "message": "NeuroSense REST API is running."
    }), 200


@health_bp.route("/version", methods=["GET"])
def version():
    """
    GET /version
    Returns current API version.
    """
    log_request(request)
    return jsonify({
        "api_version": API_VERSION
    }), 200

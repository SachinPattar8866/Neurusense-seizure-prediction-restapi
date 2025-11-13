from flask import jsonify

class APIError(Exception):
    """Custom API exception class."""
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.message = message
        if status_code:
            self.status_code = status_code
        self.payload = payload or {}

    def to_dict(self):
        rv = dict(self.payload)
        rv["success"] = False
        rv["error"] = self.message
        return rv


def register_error_handlers(app):
    """Register global error handlers for Flask app."""
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"success": False, "error": "Resource not found"}), 404

    @app.errorhandler(413)
    def file_too_large(_):
        return jsonify({"success": False, "error": "Uploaded file too large"}), 413

    @app.errorhandler(500)
    def internal_error(_):
        return jsonify({"success": False, "error": "Internal server error"}), 500

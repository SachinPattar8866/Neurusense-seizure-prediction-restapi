from flask import Blueprint, request, jsonify
from services.rehab_service import evaluate_rehab
from utils.errors import APIError
from utils.logging_utils import log_request
from validators.schema_validation import validate_rehab_payload

rehab_bp = Blueprint("rehab", __name__)

@rehab_bp.route("/rehab-status", methods=["POST"])
def rehab_status():
    """
    POST /api/rehab-status
    Expects JSON payload: {"probability": 0.78, "patient_id": "12345" (optional)}
    Returns clinical decision based on ALERT_THRESHOLD.
    """
    try:
        log_request(request)

        data = request.get_json(silent=True)
        if not data:
            raise APIError("Invalid or empty JSON payload", status_code=400)

        # Validate payload schema
        validate_rehab_payload(data)
        probability = float(data["probability"])

        decision = evaluate_rehab(probability)

        return jsonify({
            "success": True,
            "decision": decision,
            "message": "Rehabilitation status evaluated successfully."
        }), 200

    except APIError as e:
        return jsonify({"success": False, "error": e.message}), e.status_code
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500

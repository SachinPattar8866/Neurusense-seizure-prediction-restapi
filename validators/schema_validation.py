from utils.errors import APIError

def validate_rehab_payload(payload: dict):
    """
    Validate JSON payload for /rehab-status endpoint.

    Expected Example:
    {
        "probability": 0.78,
        "patient_id": "12345"  # optional
    }
    """
    if not isinstance(payload, dict):
        raise APIError("Payload must be a valid JSON object", status_code=400)

    # Required field: probability
    if "probability" not in payload:
        raise APIError("Missing required field: 'probability'", status_code=400)

    prob = payload["probability"]
    if not isinstance(prob, (int, float)) or not (0.0 <= prob <= 1.0):
        raise APIError("Field 'probability' must be a float between 0 and 1", status_code=400)

    # Optional field: patient_id (must be string if provided)
    if "patient_id" in payload and not isinstance(payload["patient_id"], str):
        raise APIError("Field 'patient_id' must be a string", status_code=400)

    return True

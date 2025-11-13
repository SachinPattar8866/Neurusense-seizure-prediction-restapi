import math
from config.settings import ALERT_THRESHOLD
from utils.errors import APIError


def evaluate_rehab(probability: float) -> dict:
    """
    Evaluate seizure risk and determine rehabilitation/hospital alert.
    - probability: seizure probability (float between 0–1)
    """

    # --- Validate probability ---
    if probability is None or isinstance(probability, str) or math.isnan(probability):
        raise APIError("Invalid probability value: must be a number between 0 and 1", status_code=400)

    if not (0.0 <= probability <= 1.0):
        raise APIError("Invalid probability value: must be between 0.0 and 1.0", status_code=400)

    # --- Decision logic ---
    alert = probability >= ALERT_THRESHOLD
    status = "ALERT" if alert else "STABLE"

    return {
        "probability": round(probability, 4),
        "status": status,
        "threshold": ALERT_THRESHOLD,
        "alert_triggered": alert,
        "action": "Notify clinical staff immediately" if alert else "Continue monitoring",
        "message": (
            "⚠️ High seizure risk detected — clinical attention required."
            if alert else "No critical seizure activity detected."
        ),
    }

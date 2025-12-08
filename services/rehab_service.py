import math
from config.settings import ALERT_THRESHOLD
from utils.errors import APIError


# Seizure-specific rehabilitation plans
SEIZURE_PLANS = {
    "seizure": {
        "risk_level": "medium",
        "recommended_exercises": [
            "Deep breathing exercises (5-10 minutes daily)",
            "Gentle yoga or stretching (20 minutes, 3x weekly)",
            "Progressive muscle relaxation",
            "Meditation (10 minutes daily)"
        ],
        "medications": ["Standard AED regimen as prescribed"],
        "safety_precautions": [
            "Wear medical alert bracelet",
            "Avoid high-risk activities (swimming, driving)",
            "Keep emergency medication accessible",
            "Have seizure action plan with family/caregivers"
        ],
        "lifestyle_modifications": [
            "Maintain regular sleep schedule (7-9 hours)",
            "Avoid alcohol and recreational drugs",
            "Manage stress through relaxation techniques",
            "Regular meals to maintain stable blood sugar"
        ],
        "follow_up_schedule": "Every 4-6 weeks",
        "therapy_duration": "Ongoing with periodic reviews"
    },
    "preictal": {
        "risk_level": "high",
        "recommended_exercises": [
            "Immediate grounding techniques (5-Sense method)",
            "Gentle breathing exercises during episodes",
            "Light walking in safe environment",
            "Mindfulness and meditation"
        ],
        "medications": ["Acute intervention medication may be needed"],
        "safety_precautions": [
            "Constant supervision during preictal phases",
            "Move to safe location immediately",
            "Alert medical team of symptoms",
            "Keep rescue medication within reach"
        ],
        "lifestyle_modifications": [
            "Identify and avoid triggers",
            "Maintain seizure diary to track patterns",
            "Strict sleep and stress management",
            "Immediate medical notification of symptoms"
        ],
        "follow_up_schedule": "Daily during active phase, then weekly",
        "therapy_duration": "Until stable, typically 1-3 months"
    },
    "non-seizure": {
        "risk_level": "low",
        "recommended_exercises": [
            "Regular aerobic exercise (30 minutes, 5x weekly)",
            "Strength training (2-3x weekly)",
            "Flexibility and balance training",
            "Sports and recreational activities as tolerated"
        ],
        "medications": ["Continue maintenance AED as prescribed"],
        "safety_precautions": [
            "Continue wearing medical alert identification",
            "Maintain medication compliance",
            "Regular check-ups (every 6-12 months)",
            "Report any new symptoms immediately"
        ],
        "lifestyle_modifications": [
            "Maintain healthy sleep schedule",
            "Regular balanced diet",
            "Moderate stress management",
            "Continue normal daily activities"
        ],
        "follow_up_schedule": "Every 3-6 months",
        "therapy_duration": "Ongoing maintenance"
    }
}


def get_seizure_plan(seizure_type: str) -> dict:
    """Get rehabilitation plan for a specific seizure type."""
    key = seizure_type.lower().replace(" seizures", "").strip()
    type_mapping = {
        "seizure": "seizure",
        "preictal": "preictal",
        "pre-ictal": "preictal",
        "non-seizure": "non-seizure",
        "no_seizure": "non-seizure",
    }
    mapped_key = type_mapping.get(key, "seizure")
    return SEIZURE_PLANS.get(mapped_key, SEIZURE_PLANS["seizure"])


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

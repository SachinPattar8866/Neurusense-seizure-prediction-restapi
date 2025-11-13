from flask import Blueprint, jsonify
import os
import csv
from config.settings import MODEL_METADATA_PATH

metrics_bp = Blueprint('metrics', __name__)


@metrics_bp.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """Return model metrics parsed from training/results/metrics.csv if available.

    Response shape:
    {
       "success": true,
       "models": {
           "cnn": { "accuracy": x, "recall": y, "f1Score": z },
           "hybrid": { ... }
       }
    }
    """
    try:
        # MODEL_METADATA_PATH lives in the restapi package (e.g. .../Neurosense-seizure-prediction-restapi/models/...)
        # We expect the training folder to live at the repo root sibling to the restapi folder.
        proj_root = os.path.dirname(os.path.dirname(MODEL_METADATA_PATH))
        repo_root = os.path.dirname(proj_root)
        # Candidate locations to look for metrics.csv (prefer repository-level training/...) 
        candidates = [
            os.path.join(repo_root, 'training', 'results', 'metrics.csv'),
            os.path.join(proj_root, 'training', 'results', 'metrics.csv'),
            os.path.join(repo_root, 'training', 'results', 'metrics.csv'),
        ]
        metrics_csv = None
        for c in candidates:
            if os.path.exists(c):
                metrics_csv = c
                break
        models_out = {}
        if metrics_csv and os.path.exists(metrics_csv):
            with open(metrics_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_key = row.get('model') or row.get('model_name')
                    if not model_key:
                        continue
                    # normalize model_key to short name
                    mk = model_key.strip()
                    if 'hybrid' in mk.lower():
                        short = 'hybrid'
                    elif 'cnn' in mk.lower():
                        short = 'cnn'
                    else:
                        short = mk

                    try:
                        accuracy = float(row.get('accuracy', 0) or 0)
                    except Exception:
                        accuracy = 0.0
                    try:
                        recall = float(row.get('recall', 0) or 0)
                    except Exception:
                        recall = 0.0
                    try:
                        f1 = float(row.get('f1_score', row.get('f1', 0)) or 0)
                    except Exception:
                        f1 = 0.0

                    models_out.setdefault(short, {})
                    models_out[short].update({
                        'accuracy': round(accuracy * 100, 2) if 0 <= accuracy <= 1 else round(accuracy, 2),
                        'recall': round(recall * 100, 2) if 0 <= recall <= 1 else round(recall, 2),
                        'f1Score': round(f1 * 100, 2) if 0 <= f1 <= 1 else round(f1, 2),
                    })

        return jsonify({'success': True, 'models': models_out}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

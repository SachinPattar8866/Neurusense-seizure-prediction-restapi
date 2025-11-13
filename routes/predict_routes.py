from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.model_service import model_service
from services.image_preprocess import preprocess_image, waveform_image_to_spectrogram_bytes
import io
from services.auth_service import auth_service
from config.settings import MODEL_METADATA_PATH, ALERT_THRESHOLD
from validators.file_validation import validate_image_file
from utils.errors import APIError
from utils.logging_utils import log_request
import os
import json
import csv
import datetime
import numpy as np
from PIL import Image
import base64
try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except Exception:
    ZoneInfo = None
    ZONEINFO_AVAILABLE = False

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """Handle uploaded EEG image, preprocess, run inference, optionally save history and return prediction.
    Returns JSON with model probabilities, ensemble and optional 'history' meta when saved to DB.
    """
    try:
        log_request(request)

        if "file" not in request.files:
            raise APIError("No file uploaded", status_code=400)

        file = request.files["file"]
        validate_image_file(file)

        # optional form fields that clients may send to help with timezone/display
        patient_name = request.form.get('patientName') or request.form.get('patient_name')
        patient_id = request.form.get('patientId') or request.form.get('patient_id')
        client_now_iso = request.form.get('client_now') or request.form.get('clientNow') or None
        client_tz = request.form.get('client_timezone') or request.form.get('clientTimezone') or None
        client_tz_offset = None
        try:
            cto = request.form.get('client_tz_offset_minutes') or request.form.get('clientTzOffsetMinutes')
            client_tz_offset = int(cto) if cto is not None else None
        except Exception:
            client_tz_offset = None

        orig_file_bytes = file.read()
        file_bytes = orig_file_bytes

        used_precomputed_spectrogram = None
        used_waveform_conversion = False

        # Try to find a precomputed spectrogram in training images by base filename
        try:
            root_dir = os.path.dirname(os.path.dirname(MODEL_METADATA_PATH))
            training_images_root = os.path.join(os.path.dirname(root_dir), 'training', 'data', 'images')
            base_name = os.path.splitext(file.filename)[0]
            candidates = [base_name]
            if base_name.lower().endswith('_raw'):
                candidates.append(base_name[:-4])
            if base_name.lower().endswith('-raw'):
                candidates.append(base_name[:-4])
            if base_name.lower().endswith('_raw.png'):
                candidates.append(base_name[:-8])

            matched_path = None
            if os.path.exists(training_images_root):
                import glob
                for cand in candidates:
                    for ext in ('png', 'jpg', 'jpeg'):
                        pattern = os.path.join(training_images_root, '**', f'{cand}*.{ext}')
                        matches = glob.glob(pattern, recursive=True)
                        if matches:
                            matched_path = matches[0]
                            break
                    if matched_path:
                        break

            if matched_path:
                try:
                    with open(matched_path, 'rb') as sf:
                        spec_bytes = sf.read()
                        file_bytes = spec_bytes
                    proj_root = os.path.dirname(os.path.dirname(MODEL_METADATA_PATH))
                    used_precomputed_spectrogram = os.path.relpath(matched_path, start=proj_root)
                    print(f"[DEBUG] Replaced uploaded bytes with precomputed spectrogram: {matched_path}")
                except Exception as e:
                    print(f"[DEBUG] Failed to read matched spectrogram {matched_path}: {e}")
        except Exception:
            # non-fatal
            pass

        # Heuristic to detect waveform screenshots and convert to spectrogram if needed
        try:
            img_probe = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            probe_small = img_probe.resize((128, 128))
            hsv = np.array(probe_small.convert('HSV'))
            sat = hsv[:, :, 1].astype('float32') / 255.0
            mean_sat = float(np.mean(sat))
            gray = np.array(probe_small.convert('L')).astype('float32') / 255.0
            std_gray = float(np.std(gray))
            colors = np.unique(np.reshape(np.array(probe_small), (-1, 3)), axis=0)
            n_colors = int(len(colors))
            print(f"[DEBUG] spectrogram_check mean_sat={mean_sat:.6f} std_gray={std_gray:.6f} n_colors={n_colors}")

            if mean_sat < 0.04 and std_gray < 0.06 and n_colors < 120:
                if not used_precomputed_spectrogram:
                    try:
                        converted = waveform_image_to_spectrogram_bytes(orig_file_bytes, model_name="hybrid_best.h5")
                        if converted:
                            file_bytes = converted
                            used_waveform_conversion = True
                            print(f"[DEBUG] Converted uploaded waveform image to spectrogram (in-memory)")
                    except APIError as ce:
                        raise APIError(
                            "Uploaded image looks like a raw EEG waveform and automatic conversion failed.\n"
                            "Please upload a spectrogram image or provide raw EEG data (EDF) for server-side conversion.",
                            status_code=422,
                        )
        except APIError:
            raise
        except Exception:
            pass

        buf_hybrid = io.BytesIO(file_bytes)
        buf_cnn = io.BytesIO(file_bytes)

        # Prepare frontend previews (original and spectrogram used)
        try:
            uploaded_b64 = base64.b64encode(orig_file_bytes).decode('ascii') if orig_file_bytes else None
            uploaded_data_uri = f"data:image/png;base64,{uploaded_b64}" if uploaded_b64 else None
        except Exception:
            uploaded_data_uri = None

        try:
            spec_b64 = base64.b64encode(file_bytes).decode('ascii') if file_bytes else None
            spectrogram_data_uri = f"data:image/png;base64,{spec_b64}" if spec_b64 else None
        except Exception:
            spectrogram_data_uri = None

        # Preprocess
        image_array_hybrid = preprocess_image(buf_hybrid, model_name="hybrid_best.h5")
        image_array_cnn = preprocess_image(buf_cnn, model_name="cnn_best.h5")

        # Save debug previews and stats (non-fatal)
        try:
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(MODEL_METADATA_PATH)), 'logs', 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            def _log_and_save(arr, tag):
                a = np.array(arr)
                if a.ndim >= 4 and a.shape[0] == 1:
                    sample = a[0]
                else:
                    sample = a
                try:
                    sv = sample
                    if sv.ndim == 3 and sv.shape[2] == 1:
                        sv = sv[:, :, 0]
                    sv_scaled = np.clip(sv * 255.0, 0, 255).astype('uint8')
                    img = Image.fromarray(sv_scaled)
                    preview_path = os.path.join(debug_dir, f"preproc_{tag}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.png")
                    img.save(preview_path)
                except Exception:
                    pass

            _log_and_save(image_array_cnn, 'cnn')
            _log_and_save(image_array_hybrid, 'hybrid')
        except Exception:
            pass

        # Predict
        hybrid_probs = model_service.predict_with_hybrid_full(image_array_hybrid)
        cnn_probs = model_service.predict_with_cnn_full(image_array_cnn)

        labels = ["non_seizure", "preictal", "seizure"]
        hybrid_labeled = {labels[i]: round(float(hybrid_probs[i]) if i < len(hybrid_probs) else 0.0, 4)
                          for i in range(len(labels))}
        cnn_labeled = {labels[i]: round(float(cnn_probs[i]) if i < len(cnn_probs) else 0.0, 4)
                       for i in range(len(labels))}

        hybrid_seizure = float(hybrid_labeled.get("seizure", 0.0))
        cnn_seizure = float(cnn_labeled.get("seizure", 0.0))

        # Thresholds and weights
        thresholds_path = os.path.join(os.path.dirname(MODEL_METADATA_PATH), 'thresholds.json')
        thresholds = {}
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r') as tfp:
                    thresholds = json.load(tfp)
            except Exception:
                thresholds = {}

        weights = {'hybrid_best.h5': 0.5, 'cnn_best.h5': 0.5}
        try:
            metrics_csv = os.path.join(os.path.dirname(os.path.dirname(MODEL_METADATA_PATH)), 'training', 'results', 'metrics.csv')
            if os.path.exists(metrics_csv):
                with open(metrics_csv, newline='') as f:
                    r = csv.DictReader(f)
                    f1s = {row['model']: float(row.get('f1_score', 0.0)) for row in r}
                w_hybrid = f1s.get('hybrid_cnn_bilstm.h5', f1s.get('hybrid_best.h5', 0.5))
                w_cnn = f1s.get('cnn_baseline.h5', f1s.get('cnn_best.h5', 0.5))
                total = w_hybrid + w_cnn if (w_hybrid + w_cnn) > 0 else 1.0
                weights = {'hybrid_best.h5': w_hybrid / total, 'cnn_best.h5': w_cnn / total}
        except Exception:
            weights = {'hybrid_best.h5': 0.5, 'cnn_best.h5': 0.5}

        ensemble_prob = weights.get('hybrid_best.h5', 0.5) * hybrid_seizure + weights.get('cnn_best.h5', 0.5) * cnn_seizure

        ensemble_thresh = ALERT_THRESHOLD
        if 'ensemble' in thresholds and 'recall_0.9_threshold' in thresholds['ensemble']:
            try:
                ensemble_thresh = float(thresholds['ensemble']['recall_0.9_threshold'])
            except Exception:
                ensemble_thresh = ALERT_THRESHOLD

        decision_status = 'seizure' if ensemble_prob >= ensemble_thresh else 'no_seizure'

        # Log to CSV
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(MODEL_METADATA_PATH)), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, 'inference_log.csv')
            if not os.path.exists(log_file):
                with open(log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'filename', 'hybrid_seizure', 'cnn_seizure', 'ensemble_prob', 'threshold', 'decision'])
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), file.filename if file else 'uploaded', hybrid_seizure, cnn_seizure, ensemble_prob, ensemble_thresh, decision_status])
        except Exception:
            pass

        # Prepare display timestamp based on client info (for UI/report display)
        display_dt = None
        if client_now_iso:
            try:
                if isinstance(client_now_iso, str) and client_now_iso.endswith('Z'):
                    client_now_iso = client_now_iso.replace('Z', '+00:00')
                display_dt = datetime.datetime.fromisoformat(client_now_iso)
            except Exception:
                display_dt = None

        if not display_dt:
            display_dt = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

        # apply client timezone if provided
        if client_tz and ZONEINFO_AVAILABLE:
            try:
                tz = ZoneInfo(client_tz)
                display_dt = display_dt.astimezone(tz)
            except Exception:
                pass
        elif client_tz_offset is not None:
            try:
                display_dt = display_dt - datetime.timedelta(minutes=client_tz_offset)
            except Exception:
                pass

        # Save to MongoDB for authenticated user (non-blocking)
        saved_history = None
        try:
            user_id = get_jwt_identity()
            if user_id:
                record = {
                    'filename': file.filename if file else 'uploaded',
                    'models': {
                        'hybrid': hybrid_labeled,
                        'cnn': cnn_labeled,
                    },
                    'source': {
                        'used_precomputed_spectrogram': used_precomputed_spectrogram,
                        'used_waveform_conversion': used_waveform_conversion,
                    },
                    'uploaded_image_preview': uploaded_data_uri,
                    'spectrogram_image_preview': spectrogram_data_uri,
                    'ensemble': {
                        'probability': ensemble_prob,
                        'threshold': ensemble_thresh,
                        'decision': decision_status,
                    }
                }
                try:
                    save_res = auth_service.save_prediction_history(user_id, record)
                    if isinstance(save_res, dict):
                        ca = save_res.get('created_at')
                        if isinstance(ca, datetime.datetime):
                            try:
                                created_iso = ca.astimezone(datetime.timezone.utc).isoformat()
                            except Exception:
                                created_iso = str(ca)
                        else:
                            created_iso = str(ca) if ca is not None else datetime.datetime.now(datetime.timezone.utc).isoformat()

                        # normalize returned history to include both created_at and timestamp (UTC ISO)
                        save_res['created_at'] = created_iso
                        save_res['timestamp'] = created_iso
                        # include a display-friendly timestamp using client info
                        try:
                            save_res['display_timestamp'] = display_dt.strftime('%d/%m/%Y %H:%M')
                        except Exception:
                            save_res['display_timestamp'] = created_iso
                        # ensure id is a string
                        if 'id' in save_res:
                            save_res['id'] = str(save_res['id'])
                        saved_history = save_res
                except Exception:
                    saved_history = None
        except Exception:
            saved_history = None

        # Debug log timezone inputs and computed times to assist troubleshooting
        try:
            print("[TIME DEBUG] client_now_iso=", client_now_iso)
            print("[TIME DEBUG] client_tz=", client_tz)
            print("[TIME DEBUG] client_tz_offset=", client_tz_offset)
            if saved_history and isinstance(saved_history.get('created_at'), str):
                print("[TIME DEBUG] saved_history.created_at=", saved_history.get('created_at'))
            else:
                print("[TIME DEBUG] no saved_history.created_at, using fallback")
            print("[TIME DEBUG] display_dt=", display_dt.isoformat())
        except Exception:
            pass

        # Build response
        resp = {
            "success": True,
            "message": "Inference completed successfully.",
            "predictions": {
                "hybrid_cnn_bilstm": {
                    "probabilities": hybrid_labeled,
                    "seizure_probability": round(hybrid_seizure, 4),
                },
                "cnn_baseline": {
                    "probabilities": cnn_labeled,
                    "seizure_probability": round(cnn_seizure, 4),
                },
            },
            "ensemble": {
                "probability": round(ensemble_prob, 4),
                "threshold": round(ensemble_thresh, 4),
                "decision": decision_status,
                "weights": weights,
            },
            "source": {
                "used_precomputed_spectrogram": used_precomputed_spectrogram,
                "used_waveform_conversion": used_waveform_conversion,
            },
        }

        if saved_history:
            resp['history'] = saved_history
        else:
            # DB save may have failed or returned nothing — provide a canonical fallback so UI can use server time
            created_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                display_fallback = display_dt.strftime('%d/%m/%Y %H:%M')
            except Exception:
                display_fallback = created_iso
            resp['history'] = {
                'id': None,
                'created_at': created_iso,
                'timestamp': created_iso,
                'display_timestamp': display_fallback,
            }

        return jsonify(resp), 200

    except APIError as e:
        return jsonify({"success": False, "error": e.message}), e.status_code
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500


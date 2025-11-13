from flask import Blueprint, request, jsonify, render_template, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from utils.db import get_db
from datetime import datetime
import os
import base64
from io import BytesIO
import matplotlib
# Use non-interactive backend to avoid tkinter / GUI issues when generating charts on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except Exception:
    HTML = None
    CSS = None
    WEASYPRINT_AVAILABLE = False
import shutil
try:
    import pdfkit
    # require wkhtmltopdf binary available on PATH
    PDFKIT_BINARY = shutil.which('wkhtmltopdf')
    PDFKIT_AVAILABLE = PDFKIT_BINARY is not None
except Exception:
    pdfkit = None
    PDFKIT_AVAILABLE = False
from bson.objectid import ObjectId
try:
    # Python 3.9+ zoneinfo for IANA timezones
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except Exception:
    ZoneInfo = None
    ZONEINFO_AVAILABLE = False

reports_bp = Blueprint('reports', __name__)


def _chart_to_datauri(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('ascii')


def make_model_comparison_chart(prediction):
    # Bar chart comparing per-model seizure probability (percent)
    labels = ['CNN', 'Hybrid']
    cnn = prediction.get('modelProbabilities', {}).get('cnn', 0)
    hybrid = prediction.get('modelProbabilities', {}).get('hybrid', 0)
    values = [cnn, hybrid]
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.bar(labels, values, color=['#4f46e5', '#ec4899'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Model Comparison')
    return _chart_to_datauri(fig)


def make_confidence_trend_chart(prediction):
    # Simple trend: use last 3 confidences if provided, otherwise replicate current confidence
    history = prediction.get('history_confidence', [])
    if not history:
        val = prediction.get('confidence', 0)
        xs = [1, 2, 3]
        ys = [max(0, val - 5), max(0, val - 2), val]
    else:
        xs = list(range(1, len(history) + 1))
        ys = history
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    ax.plot(xs, ys, marker='o', color='#4f46e5')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Recent uploads')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence Trend')
    return _chart_to_datauri(fig)


@reports_bp.route('/reports/generate', methods=['POST'])
@jwt_required()
def generate_report():
    user_id = get_jwt_identity()
    db = get_db()
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'success': False, 'error': 'Missing payload'}), 400

    # Accept full prediction payload from client for fidelity
    prediction = payload.get('prediction') or payload

    # continue even if WeasyPrint isn't installed; we'll try pdfkit/wkhtmltopdf as a fallback

    # generate charts
    try:
        model_comp = make_model_comparison_chart(prediction)
        conf_trend = make_confidence_trend_chart(prediction)
    except Exception as e:
        model_comp = None
        conf_trend = None

    # Determine generation timestamp: prefer client-provided (so PDF shows user's local time),
    # otherwise fall back to server UTC.
    client_now_iso = payload.get('client_now') if isinstance(payload, dict) else None
    gen_dt = None
    if client_now_iso:
        try:
            # Support ISO strings ending with Z by converting Z -> +00:00
            if isinstance(client_now_iso, str) and client_now_iso.endswith('Z'):
                client_now_iso = client_now_iso.replace('Z', '+00:00')
            gen_dt = datetime.fromisoformat(client_now_iso)
        except Exception:
            gen_dt = None

    if not gen_dt:
        # server-side timezone-aware UTC
        gen_dt = datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    # Debug: log received client timezone payload and computed gen/display times
    try:
        print('[REPORT TIME DEBUG] client_now_iso=', client_now_iso)
        print('[REPORT TIME DEBUG] client_timezone=', client_tz)
        print('[REPORT TIME DEBUG] client_tz_offset_minutes=', client_tz_offset)
        print('[REPORT TIME DEBUG] gen_dt=', getattr(gen_dt, 'isoformat', lambda: str(gen_dt))())
        print('[REPORT TIME DEBUG] display_dt=', getattr(display_dt, 'isoformat', lambda: str(display_dt))())
    except Exception:
        pass

    # If client provided a timezone identifier, prefer it (IANA tz like 'Europe/London')
    client_tz = None
    try:
        client_tz = payload.get('client_timezone') if isinstance(payload, dict) else None
    except Exception:
        client_tz = None

    # If timezone not provided, fall back to offset minutes
    client_tz_offset = None
    try:
        client_tz_offset = int(payload.get('client_tz_offset_minutes')) if isinstance(payload, dict) and payload.get('client_tz_offset_minutes') is not None else None
    except Exception:
        client_tz_offset = None

    display_dt = gen_dt
    # Prefer IANA timezone if provided and zoneinfo is available
    if client_tz and ZONEINFO_AVAILABLE:
        try:
            tz = ZoneInfo(client_tz)
            display_dt = gen_dt.astimezone(tz)
        except Exception:
            display_dt = gen_dt
    elif client_tz_offset is not None:
        try:
            # getTimezoneOffset is minutes difference UTC - local, so local = UTC - offset_minutes
            display_dt = gen_dt - datetime.timedelta(minutes=client_tz_offset)
        except Exception:
            display_dt = gen_dt

    # Prepare template context
    ctx = {
        # format date/time as dd/mm/YYYY HH:MM using the chosen gen_dt
    'generated_at': display_dt.strftime("%d/%m/%Y %H:%M"),
        'patientName': prediction.get('patientName', 'unknown'),
        'patientId': prediction.get('patientId', 'unknown'),
        'eegFileName': prediction.get('eegFileName', ''),
        'uploaded_image_preview': prediction.get('uploaded_image_preview') or prediction.get('source', {}).get('uploaded_image_preview'),
        'spectrogram_image_preview': prediction.get('spectrogram_image_preview') or prediction.get('source', {}).get('spectrogram_image_preview'),
        'hybrid': {'probabilities': prediction.get('modelProbabilities') and { 'non_seizure': 0, 'preictal': 0, 'seizure': prediction.get('modelProbabilities', {}).get('hybrid', 0) }},
        'cnn': {'probabilities': prediction.get('modelProbabilities') and { 'non_seizure': 0, 'preictal': 0, 'seizure': prediction.get('modelProbabilities', {}).get('cnn', 0) }},
        'ensemble': prediction.get('ensemble') or {'probability': prediction.get('confidence', 0), 'threshold': 0.7, 'decision': prediction.get('seizureType')},
        'charts': {
            'model_comparison_png': model_comp,
            'confidence_trend_png': conf_trend,
        },
        'clinical_text': payload.get('clinical_text') or '',
        'technical': payload.get('technical') or {},
    }

    # Render HTML template
    html = render_template('report_template.html', **ctx)

    # Save PDF
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(repo_root, 'reports', str(user_id))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"report_{prediction.get('patientId','unknown')}_{int(datetime.utcnow().timestamp())}.pdf"
    pdf_path = os.path.join(out_dir, fname)

    # Try WeasyPrint first
    pdf_generated = False
    last_err = None
    if WEASYPRINT_AVAILABLE:
        try:
            HTML(string=html, base_url=request.host_url).write_pdf(pdf_path, stylesheets=[CSS(string='@page { size: A4; margin: 1cm }')])
            pdf_generated = True
        except Exception as e:
            last_err = str(e)

    # Fallback to wkhtmltopdf via pdfkit if available
    if not pdf_generated and PDFKIT_AVAILABLE:
        try:
            config = pdfkit.configuration(wkhtmltopdf=PDFKIT_BINARY) if PDFKIT_BINARY else None
            options = {'page-size': 'A4', 'margin-top': '10mm', 'margin-bottom': '10mm', 'encoding': 'UTF-8'}
            if config:
                pdfkit.from_string(html, pdf_path, options=options, configuration=config)
            else:
                pdfkit.from_string(html, pdf_path, options=options)
            pdf_generated = True
        except Exception as e:
            last_err = (last_err or '') + '\n' + str(e)

    if not pdf_generated:
        # No backend available or both backends failed
        if not WEASYPRINT_AVAILABLE and not PDFKIT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'PDF generation unavailable: neither WeasyPrint nor wkhtmltopdf (pdfkit) are available on the server. Install WeasyPrint or wkhtmltopdf and the python pdfkit package. See project README for platform-specific instructions.'
            }), 501
        return jsonify({'success': False, 'error': f'PDF generation failed: {last_err}'}), 500

    # Save metadata to DB
    reports_coll = db['reports']
    # Save metadata to DB; store the UTC instant (gen_dt) as created_at
    created_at_dt = gen_dt
    # ensure created_at is timezone-aware datetime
    try:
        if isinstance(created_at_dt, str):
            if created_at_dt.endswith('Z'):
                created_at_dt = created_at_dt.replace('Z', '+00:00')
            created_at_dt = datetime.fromisoformat(created_at_dt)
    except Exception:
        created_at_dt = datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    doc = {
        'user_id': ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
        'patientName': ctx['patientName'],
        'patientId': ctx['patientId'],
        'eegFileName': ctx['eegFileName'],
        'pdf_path': pdf_path,
        'charts': ctx['charts'],
        'created_at': created_at_dt,
    }
    res = reports_coll.insert_one(doc)

    return jsonify({'success': True, 'report_id': str(res.inserted_id), 'download_url': f'/api/reports/{str(res.inserted_id)}/download'}), 201


@reports_bp.route('/reports', methods=['GET'])
@jwt_required()
def list_reports():
    user_id = get_jwt_identity()
    db = get_db()
    reports_coll = db['reports']
    q = {'user_id': ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id}
    rows = []
    for d in reports_coll.find(q).sort('created_at', -1):
        # Convert Mongo types to JSON-serializable values
        out = {}
        for k, v in d.items():
            if k == '_id':
                out['id'] = str(v)
            elif isinstance(v, ObjectId):
                out[k] = str(v)
            elif isinstance(v, datetime):
                out[k] = v.isoformat()
            else:
                out[k] = v
        rows.append(out)
    return jsonify({'success': True, 'reports': rows}), 200


@reports_bp.route('/reports/<report_id>/download', methods=['GET'])
@jwt_required()
def download_report(report_id):
    user_id = get_jwt_identity()
    db = get_db()
    reports_coll = db['reports']
    doc = reports_coll.find_one({'_id': ObjectId(report_id)})
    if not doc:
        return jsonify({'success': False, 'error': 'Report not found'}), 404
    # check ownership
    owner = doc.get('user_id')
    if str(owner) != str(user_id) and not (isinstance(owner, ObjectId) and str(owner) == str(user_id)):
        return jsonify({'success': False, 'error': 'Forbidden'}), 403

    pdf_path = doc.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'File missing'}), 404
    return send_file(pdf_path, as_attachment=True)

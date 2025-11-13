from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.auth_service import auth_service

history_bp = Blueprint('history', __name__)


@history_bp.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    limit = int(request.args.get('limit', 50))
    skip = int(request.args.get('skip', 0))
    try:
        items = auth_service.get_prediction_history(user_id, limit=limit, skip=skip)
        return jsonify({'items': items}), 200
    except Exception as e:
        return jsonify({'error': 'failed to fetch history', 'detail': str(e)}), 500

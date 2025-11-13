from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from services.auth_service import auth_service
from config import settings

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/signup', methods=['POST'])
def signup():
    # Debug: log incoming payload for troubleshooting client/server mismatch
    try:
        data = request.get_json(force=False) or {}
    except Exception:
        # If JSON parsing fails, capture raw data for debugging
        raw = request.get_data(as_text=True)
        print(f"[DEBUG] signup raw body: {raw}")
        data = {}
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'msg': 'email and password required'}), 400
    try:
        user = auth_service.signup(email, password)
    except ValueError as e:
        if str(e) == 'email_exists':
            return jsonify({'msg': 'email already exists'}), 409
        return jsonify({'msg': 'signup failed'}), 500
    # create token
    access_token = create_access_token(identity=str(user['_id']))
    return jsonify({'access_token': access_token, 'user': {'id': str(user['_id']), 'email': user['email']}}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(force=False) or {}
    except Exception:
        raw = request.get_data(as_text=True)
        print(f"[DEBUG] login raw body: {raw}")
        data = {}
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'msg': 'email and password required'}), 400
    try:
        user = auth_service.login(email, password)
    except ValueError:
        return jsonify({'msg': 'invalid credentials'}), 401
    access_token = create_access_token(identity=str(user['_id']))
    return jsonify({'access_token': access_token, 'user': {'id': str(user['_id']), 'email': user['email']}}), 200

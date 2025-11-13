import io
import os
import json
from PIL import Image

from app import create_app


def make_png_bytes(size=(128, 128), color=(255, 0, 0)):
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def run_smoke():
    app = create_app()
    client = app.test_client()

    email = 'smoke_test_user@example.com'
    password = 'TestPass123!'

    # 1) Signup
    rv = client.post('/api/signup', json={'email': email, 'password': password})
    print('signup', rv.status_code, rv.get_json())

    # 2) Login
    rv = client.post('/api/login', json={'email': email, 'password': password})
    print('login', rv.status_code, rv.get_json())
    if rv.status_code != 200 and rv.status_code != 201:
        print('Login failed, aborting smoke test')
        return
    token = rv.get_json().get('access_token')
    headers = {'Authorization': f'Bearer {token}'}

    # 3) Predict (upload PNG)
    img_buf = make_png_bytes()
    data = {
        'file': (img_buf, 'test.png')
    }
    rv = client.post('/api/predict', content_type='multipart/form-data', headers=headers, data=data)
    print('predict', rv.status_code)
    try:
        print(json.dumps(rv.get_json(), indent=2))
    except Exception:
        print('predict: no json')

    # 4) Fetch history
    rv = client.get('/api/history', headers=headers)
    print('history', rv.status_code)
    try:
        print(json.dumps(rv.get_json(), indent=2))
    except Exception:
        print('history: no json')


if __name__ == '__main__':
    run_smoke()

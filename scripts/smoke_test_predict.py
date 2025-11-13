import io
import os
import sys
from PIL import Image

# Ensure repo root is on sys.path so `from app import create_app` works
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from app import create_app

app = create_app()
app.config['TESTING'] = True
client = app.test_client()

img = Image.new('RGB', (128,128), color='white')
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

data = {'file': (buf, 'test.png')}
resp = client.post('/api/predict', data=data, content_type='multipart/form-data')
print('status', resp.status_code)
print('json', resp.get_json())

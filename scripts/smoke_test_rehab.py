import os
import sys
import json

# make repo root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from app import create_app

app = create_app()
app.config['TESTING'] = True
client = app.test_client()

payload = {"probability": 0.75, "patient_id": "test-1"}
resp = client.post('/api/rehab-status', json=payload)
print('status', resp.status_code)
print('json', resp.get_json())

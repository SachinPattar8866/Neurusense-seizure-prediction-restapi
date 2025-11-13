import io
import pytest
from app import create_app
from PIL import Image

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()

def test_predict_no_file(client):
    """POST /predict without file should return 400"""
    resp = client.post("/api/predict")
    assert resp.status_code == 400
    assert not resp.json["success"]

def test_predict_invalid_file(client):
    """POST /predict with non-image file should return 400"""
    data = {"file": (io.BytesIO(b"not-an-image"), "test.txt")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert not resp.json["success"]

def test_predict_valid_png(client):
    """POST /predict with valid PNG should return valid probabilities"""
    img = Image.new("RGB", (128, 128), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {"file": (buf, "test.png")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert resp.json["success"]
    assert "hybrid_cnn_bilstm_probability" in resp.json
    assert "cnn_baseline_probability" in resp.json

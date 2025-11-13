import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()

def test_health_endpoint(client):
    """GET /health should return UP status"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "UP"
    assert "Neurosense" in resp.json["message"]

def test_version_endpoint(client):
    """GET /version should return API version"""
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "api_version" in resp.json

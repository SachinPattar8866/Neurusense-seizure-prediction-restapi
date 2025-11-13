import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()

def test_rehab_status_missing_data(client):
    """POST /rehab-status with no JSON should fail"""
    resp = client.post("/api/rehab-status", json={})
    assert resp.status_code == 400
    assert not resp.json["success"]

def test_rehab_status_valid_probability(client):
    """POST /rehab-status with valid probability"""
    payload = {"probability": 0.85}
    resp = client.post("/api/rehab-status", json=payload)
    assert resp.status_code == 200
    assert resp.json["success"]
    assert "decision" in resp.json
    decision = resp.json["decision"]
    assert "status" in decision
    assert "probability" in decision
    assert "action" in decision

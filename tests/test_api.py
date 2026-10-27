"""Testes da API com FastAPI TestClient."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid():
    r = client.post("/v1/predict", json={"pixels": [0.0] * 784})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["predicted_class"] <= 9
    assert len(body["probabilities"]) == 10


def test_predict_invalid_length():
    r = client.post("/v1/predict", json={"pixels": [0.0] * 10})
    assert r.status_code == 422

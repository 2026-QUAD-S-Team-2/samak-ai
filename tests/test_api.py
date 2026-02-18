from fastapi.testclient import TestClient

from app.main import app


def test_healthz_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_image_missing_input_returns_200_with_template() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/analyze/image", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "analysisId" in data
    assert data["riskLevel"] == "UNKNOWN"
    assert data["fraudProbability"] is None
    assert data["riskScore"] is None
    assert data["riskSignals"] == []
    assert data["travelBanRegionsMatched"] == []
    assert isinstance(data["message"], str)
    assert data["message"].strip() != ""

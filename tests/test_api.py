from fastapi.testclient import TestClient

from app.main import app


def test_healthz_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_image_missing_body_returns_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/analyze/image", json={})
    assert resp.status_code == 422


def test_analyze_image_invalid_url_returns_400() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/analyze/image", json={"imageUrl": "not-a-url", "debug": False})
    assert resp.status_code == 400


def test_analyze_image_download_fail_returns_400(monkeypatch) -> None:
    from fastapi import HTTPException

    import app.routes.analyze as analyze_route

    async def _fail(_url: str) -> bytes:
        raise HTTPException(status_code=400, detail="이미지 다운로드 실패: test")

    monkeypatch.setattr(analyze_route, "_download_image_bytes", _fail)

    with TestClient(app) as client:
        resp = client.post("/v1/analyze/image", json={"imageUrl": "https://example.com/sample.png", "debug": False})
    assert resp.status_code == 400


def test_analyze_image_multiple_urls_returns_results(monkeypatch) -> None:
    import app.routes.analyze as analyze_route

    async def _fake(_url: str, *, debug: bool, background_tasks) -> dict:  # noqa: ANN001
        assert debug is False
        assert background_tasks is not None
        return {
            "analysisId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "fraudProbability": 0.1,
            "riskScore": 10,
            "riskLevel": "LOW",
            "riskSignals": [],
            "travelBanRegionsMatched": [],
            "message": "ok",
        }

    monkeypatch.setattr(analyze_route, "_analyze_one_image_url", _fake)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/analyze/image",
            json={"imageUrls": ["https://example.com/a.png", "https://example.com/b.png"], "debug": False},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2

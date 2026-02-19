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

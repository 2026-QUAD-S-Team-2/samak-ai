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
        resp = client.post("/v1/analyze/image", json={"imageUrl": "not-a-url", "countryCode": "KR", "debug": False})
    assert resp.status_code == 400


def test_analyze_image_download_fail_returns_400(monkeypatch) -> None:
    from fastapi import HTTPException

    import app.routes.analyze as analyze_route

    async def _fail(_url: str) -> bytes:
        raise HTTPException(status_code=400, detail="이미지 다운로드 실패: test")

    monkeypatch.setattr(analyze_route, "_download_image_bytes", _fail)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/analyze/image",
            json={"imageUrls": ["https://example.com/sample.png"], "countryCode": "KR", "debug": False},
        )
    assert resp.status_code == 400


def test_analyze_image_multiple_urls_returns_results(monkeypatch) -> None:
    import app.routes.analyze as analyze_route

    async def _ok(_url: str) -> bytes:
        return b"fake"

    def _ocr(_b: bytes):  # noqa: ANN001
        from app.services.ocr_service import OCRResult

        return OCRResult(
            text="this is long enough to run ml inference " * 3,
            text_preview="preview",
            text_length=100,
            language_guess="en",
            confidence_avg=0.9,
            error=None,
        )

    class _Model:  # noqa: D401
        threshold = 0.5

        @staticmethod
        def load_default():  # noqa: ANN001
            return _Model()

        def get_cleaned_input_from_ocr(self, _t: str) -> str:  # noqa: ANN001
            return "cleaned"

        def predict_proba(self, _t: str) -> float:  # noqa: ANN001
            return 0.2

    monkeypatch.setattr(analyze_route, "_download_image_bytes", _ok)
    monkeypatch.setattr(analyze_route, "ocr_from_bytes", _ocr)
    monkeypatch.setattr(analyze_route, "BaselineModel", _Model)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/analyze/image",
            json={
                "imageUrls": ["https://example.com/a.png", "https://example.com/b.png"],
                "countryCode": "KR",
                "debug": False,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert len(data["results"]) == 2

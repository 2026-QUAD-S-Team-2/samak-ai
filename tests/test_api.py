from fastapi.testclient import TestClient
import pytest
from pathlib import Path

from app.main import app


def test_healthz_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_infer_empty_text_400() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/infer", json={"analysisId": "a-1", "text": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "text must be non-empty"


def test_infer_model_missing_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 로컬에 모델이 존재하면 200이 나올 수 있으니, 일부러 없는 MODEL_DIR로 바꿔서 503을 확인
    from app.settings import get_settings

    # TestClient lifespan에서 get_settings()를 호출하므로, env 변경 전에 cache_clear가 필요
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no-model"))
    get_settings.cache_clear()

    with TestClient(app) as client:
        resp = client.post("/v1/infer", json={"analysisId": "a-1", "text": "hello"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Model not loaded"


def test_external_integrations_disabled_503() -> None:
    with TestClient(app) as client:
        resp_maps = client.post("/v1/external/maps/geocode", json={"address": "Seoul"})
        resp_gemini = client.post("/v1/external/gemini/generate", json={"prompt": "hi"})

    assert resp_maps.status_code == 503
    assert resp_maps.json()["detail"] == "Maps integration disabled"

    assert resp_gemini.status_code == 503
    assert resp_gemini.json()["detail"] == "Gemini integration disabled"

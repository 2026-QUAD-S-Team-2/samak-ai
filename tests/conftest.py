from __future__ import annotations

import sys
from pathlib import Path

# `pytest`를 레포 루트에서 실행할 때도 `import app`이 되도록 경로 보정
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 테스트 환경에서는 로컬 `.env` 영향(외부 API 활성화 등)을 받지 않도록 고정합니다.
import pytest


@pytest.fixture(autouse=True)
def _stable_test_env(monkeypatch: pytest.MonkeyPatch):
    from app.settings import get_settings

    monkeypatch.setenv("ENABLE_MAPS", "false")
    monkeypatch.setenv("ENABLE_GEMINI", "false")
    monkeypatch.delenv("MAPS_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

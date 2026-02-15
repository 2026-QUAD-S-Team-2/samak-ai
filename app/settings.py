from __future__ import annotations

"""
환경변수(Settings) 모듈.

FastAPI 코드에서 `get_settings()`만 보면 어떤 환경변수가 필요한지 바로 알 수 있게 합니다.
`.env.example`를 참고해서 로컬에서 쉽게 실행할 수 있습니다.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _env_bool(name: str, default: str = "false") -> bool:
    # "true/1/yes/on" 등 다양한 문자열을 bool로 변환
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    model_dir: str
    model_version: str
    max_text_chars: int
    log_level: str
    external_timeout_seconds: float
    enable_maps: bool
    maps_api_key: str | None
    enable_gemini: bool
    gemini_api_key: str | None
    gemini_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Settings는 실행 중 변하지 않는 값이라 1회만 만들고 재사용합니다.
    return Settings(
        model_dir=os.environ.get("MODEL_DIR", os.path.join("models", "fraud-baseline")),
        model_version=os.environ.get("MODEL_VERSION", "fraud-baseline-v1.0.0"),
        max_text_chars=int(os.environ.get("MAX_TEXT_CHARS", "20000")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        external_timeout_seconds=float(os.environ.get("EXTERNAL_TIMEOUT_SECONDS", "10")),
        enable_maps=_env_bool("ENABLE_MAPS", "false"),
        maps_api_key=os.environ.get("MAPS_API_KEY") or None,
        enable_gemini=_env_bool("ENABLE_GEMINI", "false"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
    )

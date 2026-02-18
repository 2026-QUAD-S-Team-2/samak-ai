from __future__ import annotations

"""
백엔드로 분석 결과를 push(POST)하는 유틸.

주의:
- 네트워크/인증/엔드포인트는 환경변수로만 설정합니다.
- 실패해도 서비스 전체는 멈추지 않아야 하므로, 예외는 삼키고 상태만 반환합니다.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from app.backend_push_config import (
    BACKEND_PUSH_API_KEY,
    BACKEND_PUSH_ENABLED,
    BACKEND_PUSH_RETRIES,
    BACKEND_PUSH_TIMEOUT_SECONDS,
    BACKEND_PUSH_URL,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PushResult:
    attempted: bool
    sent: bool
    status_code: int | None
    error: str | None


def push_analysis_result(payload: dict[str, Any]) -> PushResult:
    """
    분석 결과를 백엔드로 전송합니다.

    설정은 `app/backend_push_config.py` 하드코딩 값을 사용합니다.
    """
    enabled = bool(BACKEND_PUSH_ENABLED)
    url = (BACKEND_PUSH_URL or "").strip()
    if not enabled or not url:
        return PushResult(attempted=False, sent=False, status_code=None, error=None)

    api_key = (BACKEND_PUSH_API_KEY or "").strip()
    timeout_s = float(BACKEND_PUSH_TIMEOUT_SECONDS)
    retries = int(BACKEND_PUSH_RETRIES)
    retries = max(0, min(5, retries))

    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            if 200 <= resp.status_code < 300:
                logger.info("backend push ok status=%s", resp.status_code)
                return PushResult(attempted=True, sent=True, status_code=resp.status_code, error=None)
            last_err = f"non-2xx status={resp.status_code}"
            logger.warning("backend push failed: %s", last_err)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            logger.warning("backend push exception: %s", last_err)

        if attempt < retries:
            time.sleep(0.2 * (attempt + 1))

    return PushResult(attempted=True, sent=False, status_code=None, error=last_err)

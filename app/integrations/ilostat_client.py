from __future__ import annotations

"""
ILOSTAT (or compatible) 데이터 조회 클라이언트.

주의:
- indicator code/endpoint는 프로젝트에서 확정되지 않을 수 있으므로 env 기반으로 주입합니다.
- 네트워크 실패/데이터 없음은 예외를 던지지 않고 None을 반환합니다.
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _extract_first_number(obj: Any) -> float | None:
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except Exception:
            return None
    if isinstance(obj, dict):
        for v in obj.values():
            out = _extract_first_number(v)
            if out is not None:
                return out
        return None
    if isinstance(obj, list):
        for v in obj:
            out = _extract_first_number(v)
            if out is not None:
                return out
        return None
    return None


async def fetch_series(
    *,
    country_code: str,
    indicator_code: str,
    timeout_s: float | None = None,
    base_url: str | None = None,
) -> float | None:
    """
    지정 국가/indicator에 대한 값을 1개(float)로 반환합니다.

    - API 스펙이 확정되지 않았으므로 응답 JSON에서 첫 번째 숫자를 탐색하는 보수적 파서를 사용합니다.
    - base_url이 비어 있거나 indicator_code가 비어 있으면 None.
    """
    ind = (indicator_code or "").strip()
    if not ind:
        return None

    url = (base_url or os.environ.get("ILOSTAT_BASE_URL") or "").strip()
    if not url:
        return None

    timeout = float(timeout_s if timeout_s is not None else float(os.environ.get("EXTERNAL_TIMEOUT_SECONDS", "10")))

    params = {
        "countryCode": (country_code or "").strip().upper(),
        "indicator": ind,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
        if resp.status_code < 200 or resp.status_code >= 300:
            logger.warning("ilostat fetch non-2xx status=%s", resp.status_code)
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        return _extract_first_number(data)
    except Exception as e:  # noqa: BLE001
        logger.warning("ilostat fetch failed: %s", e)
        return None


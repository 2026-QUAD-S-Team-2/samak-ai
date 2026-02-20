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

def to_alpha3(country_code_alpha2: str) -> str | None:
    try:
        import pycountry  # type: ignore

        country = pycountry.countries.get(alpha_2=(country_code_alpha2 or "").strip().upper())
        return str(country.alpha_3) if country else None
    except Exception:  # noqa: BLE001
        return None


def _build_url(country_alpha3: str, indicator: str) -> str | None:
    base = (os.environ.get("ILOSTAT_BASE_URL") or "").strip()
    ind = (indicator or "").strip()
    cc = (country_alpha3 or "").strip().upper()
    if base == "" or ind == "" or cc == "":
        return None

    # 1) BASE_URL이 완성 URL 템플릿인 경우 ({country}에는 alpha-3 사용)
    if ("{indicator}" in base) or ("{country}" in base) or ("{countryCode}" in base):
        try:
            return base.format(indicator=ind, country=cc, countryCode=cc)
        except Exception:  # noqa: BLE001
            return None

    # 2) fallback: base + /sdmx-json/data/{indicator}/{country}.A
    # (API 제공 형태가 다를 수 있으므로 "유연하게" 구성)
    b = base.rstrip("/")
    return f"{b}/sdmx-json/data/{ind}/{cc}.A"


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip()
        if t == "":
            return None
        try:
            return float(t)
        except Exception:
            return None
    return None


def _parse_sdmx_json_latest(payload: dict[str, Any]) -> float | None:
    """
    SDMX-JSON:
    - payload["dataSets"][0]["series"][<any>]["observations"]에서 latest obs를 찾아 float 반환.
    """
    try:
        data_sets = payload.get("dataSets")
        if not isinstance(data_sets, list) or not data_sets:
            return None
        series = data_sets[0].get("series")  # type: ignore[union-attr]
        if not isinstance(series, dict) or not series:
            return None
        first_series = next(iter(series.values()))
        if not isinstance(first_series, dict):
            return None
        obs = first_series.get("observations")
        if not isinstance(obs, dict) or not obs:
            return None

        # observations 키는 "0","1",... 문자열인 경우가 많음
        def _k_to_int(k: Any) -> int:
            try:
                return int(str(k))
            except Exception:
                return -1

        latest_k = max(obs.keys(), key=_k_to_int)
        latest = obs.get(latest_k)

        # 형태: [value, ...] 또는 {"0":[value]} 등 유연 처리
        if isinstance(latest, list) and latest:
            return _as_float(latest[0])
        if isinstance(latest, dict):
            # {"0":[value]} 같은 케이스
            for vv in latest.values():
                if isinstance(vv, list) and vv:
                    out = _as_float(vv[0])
                    if out is not None:
                        return out
                out = _as_float(vv)
                if out is not None:
                    return out
        return _as_float(latest)
    except Exception:  # noqa: BLE001
        return None


def _parse_list_json_latest(payload: Any) -> float | None:
    """
    단순 list JSON:
    - [{"timePeriod":"2023","value":...}, ...] 등에서 최신 time을 선택.
    """
    if not isinstance(payload, list) or not payload:
        return None

    def _get_time(item: dict[str, Any]) -> str:
        for k in ("timePeriod", "year", "time", "period", "date"):
            v = item.get(k)
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        return ""

    def _time_key(t: str) -> tuple[int, str]:
        s = (t or "").strip()
        digits = ""
        for ch in s:
            if ch.isdigit():
                digits += ch
            else:
                break
        return (int(digits) if digits else -1, s)

    latest_item: dict[str, Any] | None = None
    latest_t = ""
    for it in payload:
        if not isinstance(it, dict):
            continue
        t = _get_time(it)
        if t == "":
            continue
        if latest_item is None or _time_key(t) > _time_key(latest_t):
            latest_item = it
            latest_t = t

    if latest_item is None:
        return None

    for k in ("value", "obs_value", "obsValue", "amount"):
        if k in latest_item:
            return _as_float(latest_item.get(k))
    return None


async def fetch_latest_value(countryCode: str, indicator: str) -> float | None:  # noqa: N802
    alpha3 = to_alpha3(countryCode)
    if alpha3 is None:
        logger.warning("ilostat alpha2->alpha3 failed countryCode=%s", (countryCode or "").strip().upper())
        return None

    url = _build_url(alpha3, indicator)
    if url is None:
        logger.warning("ilostat fetch skipped (missing base/url) indicator=%s countryCode=%s", indicator, countryCode)
        return None

    timeout = float(os.environ.get("EXTERNAL_TIMEOUT_SECONDS", "10"))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code < 200 or resp.status_code >= 300:
            logger.warning(
                "ilostat fetch non-2xx status=%s indicator=%s countryCode=%s",
                resp.status_code,
                indicator,
                (countryCode or "").strip().upper(),
            )
            return None

        try:
            payload: Any = resp.json()
        except Exception:
            logger.warning("ilostat json decode failed indicator=%s countryCode=%s", indicator, countryCode)
            return None

        value: float | None = None
        if isinstance(payload, dict):
            value = _parse_sdmx_json_latest(payload)
            if value is None and isinstance(payload.get("data"), list):
                value = _parse_list_json_latest(payload.get("data"))
        elif isinstance(payload, list):
            value = _parse_list_json_latest(payload)

        if value is not None:
            logger.info(
                "ilostat fetch ok indicator=%s countryCode=%s value=%s",
                indicator,
                (countryCode or "").strip().upper(),
                value,
            )
            return float(value)

        logger.warning("ilostat parse failed indicator=%s countryCode=%s", indicator, countryCode)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("ilostat fetch failed indicator=%s countryCode=%s err=%s", indicator, countryCode, e)
        return None

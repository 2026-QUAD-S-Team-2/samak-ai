from __future__ import annotations

"""
외부 API 호출 모음 (옵션 기능).

주의:
- 이 레포의 핵심은 ML 추론(/v1/infer)이며, 외부 API 호출은 기본 비활성화입니다.
- 실제 운영에서는 보통 백엔드에서 호출/검증하고, 여기서는 필요 시에만 ON 해서 사용합니다.
"""

from dataclasses import dataclass
from typing import Any

import httpx


class ExternalAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeocodeResult:
    status: str
    formatted_address: str | None
    place_id: str | None
    location: dict[str, float] | None
    raw: dict[str, Any]


def geocode_address(*, address: str, api_key: str, timeout_s: float) -> GeocodeResult:
    # Google Maps Geocoding API (REST)
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:  # noqa: BLE001
        # 네트워크/타임아웃/HTTP 에러를 하나의 도메인 에러로 감쌉니다.
        raise ExternalAPIError(f"Google Maps request failed: {e}") from e

    status = str(payload.get("status", ""))
    results = payload.get("results") or []
    first = results[0] if isinstance(results, list) and results else None

    formatted_address = None
    place_id = None
    location = None
    if isinstance(first, dict):
        formatted_address = first.get("formatted_address")
        place_id = first.get("place_id")
        geometry = first.get("geometry") if isinstance(first.get("geometry"), dict) else None
        loc = geometry.get("location") if isinstance(geometry, dict) else None
        if isinstance(loc, dict) and "lat" in loc and "lng" in loc:
            try:
                location = {"lat": float(loc["lat"]), "lng": float(loc["lng"])}
            except Exception:  # noqa: BLE001
                location = None

    return GeocodeResult(
        status=status,
        formatted_address=formatted_address,
        place_id=place_id,
        location=location,
        raw=payload if isinstance(payload, dict) else {"payload": payload},
    )


@dataclass(frozen=True)
class GeminiGenerateResult:
    model: str
    text: str
    raw: dict[str, Any]


def gemini_generate_text(*, api_key: str, model: str, prompt: str, timeout_s: float) -> GeminiGenerateResult:
    # Gemini (Generative Language API) REST 호출
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, params=params, json=body)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:  # noqa: BLE001
        raise ExternalAPIError(f"Gemini request failed: {e}") from e

    # 응답 JSON에서 텍스트만 최대한 안전하게 뽑아옵니다. (형식이 달라져도 예외 방지)
    text = ""
    try:
        candidates = payload.get("candidates") or []
        if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                text = str(parts[0].get("text") or "")
    except Exception:  # noqa: BLE001
        text = ""

    return GeminiGenerateResult(
        model=model,
        text=text,
        raw=payload if isinstance(payload, dict) else {"payload": payload},
    )

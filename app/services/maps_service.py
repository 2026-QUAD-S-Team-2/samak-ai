from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import httpx

from app.env import load_dotenv_once

logger = logging.getLogger(__name__)

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"

SIGNAL_NOT_FOUND = "Google Maps에서 회사명이 검색되지 않습니다."
SIGNAL_MISMATCH = "Google Maps에서 검색한 회사 위치와 공고 표기 위치가 일치하지 않습니다."

MISMATCH_KM_THRESHOLD = 100.0


@dataclass(frozen=True)
class LatLng:
    lat: float
    lng: float


@dataclass(frozen=True)
class LocationResult:
    raw_text: str
    lat: float
    lng: float
    admin_level: str | None
    zoom: int
    status: str  # "company" | "region"
    viewport_ne: LatLng | None = field(default=None)
    viewport_sw: LatLng | None = field(default=None)


@dataclass(frozen=True)
class MapsContext:
    company_found: bool
    company_address: str | None = None
    company_display_name: str | None = None


def _get_api_key() -> str | None:
    load_dotenv_once()
    key = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    return key or None


def _haversine_km(a: LatLng, b: LatLng) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dphi = math.radians(b.lat - a.lat)
    dlam = math.radians(b.lng - a.lng)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _zoom_for_viewport(ne: LatLng, sw: LatLng) -> int:
    lat_span = abs(ne.lat - sw.lat)
    if lat_span < 0.05:
        return 14
    if lat_span < 0.5:
        return 12
    if lat_span < 2.0:
        return 10
    return 8


def _is_name_match(query: str, display_name: str) -> bool:
    q, d = query.strip().lower(), display_name.strip().lower()
    return bool(q and d and (q in d or d in q))


async def _search_places(
    client: httpx.AsyncClient,
    company_name: str,
    api_key: str,
    country_code: str | None = None,
) -> dict | None:
    try:
        body: dict = {"textQuery": company_name, "languageCode": "ko"}
        if country_code:
            body["regionCode"] = country_code.upper()
        resp = await client.post(
            PLACES_URL,
            json=body,
            headers={
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.addressComponents,places.formattedAddress",
            },
        )
        resp.raise_for_status()
        places = resp.json().get("places") or []
        return places[0] if places else None
    except Exception as e:
        logger.warning("Places API 호출 실패: %s", e)
        return None


async def _geocode_region(
    client: httpx.AsyncClient,
    region_text: str,
    api_key: str,
) -> LocationResult | None:
    try:
        resp = await client.get(
            GEOCODING_URL,
            params={"address": region_text, "key": api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None

        result = data["results"][0]
        loc = result["geometry"]["location"]
        viewport = result["geometry"].get("viewport", {})
        ne_raw = viewport.get("northeast", {})
        sw_raw = viewport.get("southwest", {})

        ne = LatLng(lat=ne_raw["lat"], lng=ne_raw["lng"]) if ne_raw else None
        sw = LatLng(lat=sw_raw["lat"], lng=sw_raw["lng"]) if sw_raw else None
        zoom = _zoom_for_viewport(ne, sw) if ne and sw else 8

        admin_level = None
        for comp in result.get("address_components", []):
            if "country" in comp.get("types", []):
                admin_level = comp.get("long_name")
                break
            if "administrative_area_level_1" in comp.get("types", []) and not admin_level:
                admin_level = comp.get("long_name")

        return LocationResult(
            raw_text=region_text,
            lat=loc["lat"],
            lng=loc["lng"],
            admin_level=admin_level,
            zoom=zoom,
            status="region",
            viewport_ne=ne,
            viewport_sw=sw,
        )
    except Exception as e:
        logger.warning("Geocoding API 호출 실패 (region=%s): %s", region_text, e)
        return None


async def _is_location_mismatch(
    places_lat: float,
    places_lng: float,
    places_address: str,
    regions_mentioned: list[str],
    api_key: str,
    client: httpx.AsyncClient,
) -> bool:
    places_point = LatLng(lat=places_lat, lng=places_lng)
    region_result = await _geocode_region(client, regions_mentioned[0], api_key)

    if region_result:
        distance = _haversine_km(places_point, LatLng(lat=region_result.lat, lng=region_result.lng))
        return distance > MISMATCH_KM_THRESHOLD

    # Geocoding 실패 시 문자열 매칭으로 fallback
    addr_lower = places_address.lower()
    return not any(r.lower() in addr_lower for r in regions_mentioned)


async def lookup_location(
    *,
    company_name: str | None,
    regions_mentioned: list[str],
    country_code: str | None = None,
) -> tuple[LocationResult | None, list[str], MapsContext]:
    """
    회사명 또는 언급 지역을 Google Maps로 조회한다.

    Returns:
        (LocationResult | None, extra_risk_signals, MapsContext)
        GOOGLE_MAPS_API_KEY 미설정 또는 전체 실패 시 (None, [], MapsContext(company_found=False)) 반환.
    """
    api_key = _get_api_key()
    if not api_key:
        return None, [], MapsContext(company_found=False)

    signals: list[str] = []
    location_result: LocationResult | None = None
    maps_context = MapsContext(company_found=False)

    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if company_name:
                places_data = await _search_places(client, company_name, api_key, country_code)

                if places_data is None:
                    signals.append(SIGNAL_NOT_FOUND)
                elif not _is_name_match(company_name, places_data.get("displayName", {}).get("text", "")):
                    signals.append(SIGNAL_NOT_FOUND)
                else:
                    loc = places_data.get("location", {})
                    places_lat = loc.get("latitude", 0.0)
                    places_lng = loc.get("longitude", 0.0)
                    formatted_address = places_data.get("formattedAddress", "")
                    display_name = places_data.get("displayName", {}).get("text", "")

                    maps_context = MapsContext(
                        company_found=True,
                        company_address=formatted_address or None,
                        company_display_name=display_name or None,
                    )

                    admin_level = None
                    for comp in places_data.get("addressComponents", []):
                        if "country" in comp.get("types", []):
                            admin_level = comp.get("longText")
                            break

                    if regions_mentioned:
                        mismatch = await _is_location_mismatch(
                            places_lat, places_lng, formatted_address, regions_mentioned, api_key, client
                        )
                        if mismatch:
                            signals.append(SIGNAL_MISMATCH)

                    location_result = LocationResult(
                        raw_text=company_name,
                        lat=places_lat,
                        lng=places_lng,
                        admin_level=admin_level,
                        zoom=14,
                        status="company",
                    )

            if location_result is None and regions_mentioned:
                location_result = await _geocode_region(client, regions_mentioned[0], api_key)

    except Exception as e:
        logger.warning("maps_service.lookup_location 예외: %s", e)
        return None, [], MapsContext(company_found=False)

    return location_result, signals, maps_context

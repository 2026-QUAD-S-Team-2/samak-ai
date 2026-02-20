from __future__ import annotations

"""
여행금지(외교부) 국가/지역 탐지 유틸.

- 규칙 파일: app/ml/risk_regions.txt (한 줄에 한 항목)
- 매칭은 inputCleaned(소문자/공백 정리된 텍스트) 기준으로 수행하는 것을 권장
- false positive를 줄이기 위해, 영문 단어는 word-boundary 기반으로 매칭
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def _is_latin_phrase(s: str) -> bool:
    # 영문/숫자/공백/하이픈/아포스트로피 정도만 포함하면 latin으로 취급
    return re.fullmatch(r"[a-z0-9\-\' ]+", s) is not None


def _normalize_line(s: str) -> str:
    t = (s or "").strip().lower()
    # 줄 끝의 ':' 같은 노이즈 제거
    t = t.strip(" :;\t")
    # 공백 정리
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass(frozen=True)
class RiskRegionPattern:
    key: str
    display_ko: str
    regex: re.Pattern[str] | None


_CACHED: list[RiskRegionPattern] | None = None


def get_risk_region_patterns() -> list[RiskRegionPattern]:
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    path = Path(__file__).resolve().parent / "risk_regions.txt"
    patterns: list[RiskRegionPattern] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                key_raw, display_raw = line.split("\t", 1)
                key = _normalize_line(key_raw)
                display_ko = (display_raw or "").strip()
            else:
                key = _normalize_line(line)
                display_ko = ""

            if not key:
                continue
            if not display_ko:
                # display가 없으면 key를 그대로 사용(추후 파일에 한국어를 채우는 것을 권장)
                display_ko = key

            if _is_latin_phrase(key):
                # multi-word phrase: \bword1\s+word2\b
                tokens = [re.escape(t) for t in key.split(" ") if t]
                if not tokens:
                    continue
                if len(tokens) == 1:
                    pat = rf"\b{tokens[0]}\b"
                else:
                    pat = r"\b" + r"\s+".join(tokens) + r"\b"
                try:
                    rx = re.compile(pat, re.IGNORECASE)
                except re.error as e:
                    logger.warning("risk_regions: regex compile failed key=%s err=%s", key, e)
                    rx = None
                patterns.append(RiskRegionPattern(key=key, display_ko=display_ko, regex=rx))
            else:
                # non-latin: 단순 substring 매칭
                patterns.append(RiskRegionPattern(key=key, display_ko=display_ko, regex=None))

        # 중복 제거(순서 유지)
        seen: set[str] = set()
        uniq: list[RiskRegionPattern] = []
        for p in patterns:
            if p.key in seen:
                continue
            seen.add(p.key)
            uniq.append(p)
        _CACHED = uniq
        if not _CACHED:
            logger.warning("No risk regions loaded from %s", path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to load risk regions from %s: %s", path, e)
        _CACHED = []

    return _CACHED


_ISO2_TO_RISK_REGION_KEY: dict[str, str] = {
    "SO": "somalia",
    "AF": "afghanistan",
    "IQ": "iraq",
    "YE": "yemen",
    "SY": "syria",
    "LY": "libya",
    "UA": "ukraine",
    "SD": "sudan",
    "PS": "palestine",
    "VE": "venezuela",
    "HT": "haiti",
    "LB": "lebanon",
    "IL": "israel",
    "LA": "laos",
    "MM": "myanmar",
    "KH": "cambodia",
    "PH": "philippines",
    "NE": "niger",
    "ML": "mali",
    "CD": "democratic republic of the congo",
    "RU": "russia",
    "BY": "belarus",
    "AM": "armenia",
    "AZ": "azerbaijan",
}


def match_risk_regions_by_country_code(country_code: str) -> list[str]:
    """
    요청 countryCode(ISO 3166-1 alpha-2)가 위험 지역 목록(app/ml/risk_regions.txt)에 대응되면
    해당 display(한국어) 문자열을 반환합니다.
    """
    code = (country_code or "").strip().upper()
    key = _ISO2_TO_RISK_REGION_KEY.get(code)
    if not key:
        return []

    patterns = get_risk_region_patterns()
    display_by_key = {p.key: p.display_ko for p in patterns}
    display = display_by_key.get(_normalize_line(key))
    return [display] if display else []


def find_risk_regions(text: str, *, top_k: int = 5) -> list[str]:
    """
    텍스트에 포함된 여행금지 국가/지역명을 탐지합니다.

    - 반환은 매칭된 region name 목록(중복 제거, 순서 유지)
    - 너무 많이 매칭되면 top_k까지만 반환
    """
    t = (text or "").lower()
    if not t:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for pat in get_risk_region_patterns():
        matched = False
        if pat.regex is not None:
            matched = pat.regex.search(t) is not None
        else:
            matched = pat.key in t
        if not matched:
            continue
        if pat.display_ko in seen:
            continue
        seen.add(pat.display_ko)
        out.append(pat.display_ko)
        if len(out) >= top_k:
            break
    return out

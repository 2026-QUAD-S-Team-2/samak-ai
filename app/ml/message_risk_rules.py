from __future__ import annotations

"""
사기 패턴(위험 신호) 규칙 기반 탐지.

요구사항:
- 패턴은 외부 파일(app/ml/risk_patterns.txt)로 분리
- 영어/한국어 혼합 커버
- 정규식 기반(case-insensitive)
- 반환은 "매칭된 표현" (substring) 목록
- Top 3, 억지로 채우지 않음
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _normalize_signal(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


@dataclass(frozen=True)
class RiskPattern:
    name: str
    regex: re.Pattern[str]


def _load_patterns_from_txt(path: Path) -> list[RiskPattern]:
    patterns: list[RiskPattern] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            logger.warning("risk_patterns.txt: invalid line (missing tab): %r", line)
            continue
        name, pattern = line.split("\t", 1)
        name = name.strip()
        pattern = pattern.strip()
        if not name or not pattern:
            continue
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning("risk_patterns.txt: regex compile failed name=%s err=%s", name, e)
            continue
        patterns.append(RiskPattern(name=name, regex=rx))
    return patterns


_CACHED: list[RiskPattern] | None = None


def get_risk_patterns() -> list[RiskPattern]:
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    path = Path(__file__).resolve().parent / "risk_patterns.txt"
    try:
        _CACHED = _load_patterns_from_txt(path)
        if not _CACHED:
            logger.warning("No risk patterns loaded from %s", path)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to load risk patterns from %s: %s", path, e)
        _CACHED = []
    return _CACHED


def extract_risk_signals(text: str, *, top_k: int = 3) -> list[str]:
    """
    텍스트에서 사기 패턴을 탐지해 매칭된 표현을 반환합니다.

    - 중복 제거(순서 유지)
    - 너무 짧은 토큰은 제외
    - top_k까지만 반환 (억지로 채우지 않음)
    """
    if top_k <= 0:
        return []

    t = (text or "").strip()
    if not t:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for pat in get_risk_patterns():
        for m in pat.regex.finditer(t):
            sig = _normalize_signal(m.group(0))
            if len(sig) < 3:
                continue
            key = sig.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(sig)
            if len(out) >= top_k:
                return out

    return out


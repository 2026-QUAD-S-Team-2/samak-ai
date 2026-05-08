from __future__ import annotations

import re

_SCAM_DOMAINS: tuple[str, ...] = (
    "aloisstaffing.com",
    "arksolutionsinc.com",
    "bluestonestaffing.com",
    "enterprisesolutioninc.com",
    "e-solutionsinc.com",
    "hanstaffing.com",
    "krgtech.com",
    "softpath.net",
    "ustechsolutionsinc.com",
    "vertigonconsulting.com",
)

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (domain, re.compile(re.escape(domain), re.IGNORECASE))
    for domain in _SCAM_DOMAINS
]


def find_scam_domains(text: str) -> list[str]:
    """OCR 텍스트에서 알려진 사기 도메인을 찾아 반환합니다."""
    if not text:
        return []
    return [domain for domain, pattern in _DOMAIN_PATTERNS if pattern.search(text)]

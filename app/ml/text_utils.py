from __future__ import annotations

"""
텍스트 정규화 유틸.

OCR/서빙/학습에서 동일한 방향으로 텍스트를 정리하기 위한 최소 로직만 포함합니다.
"""

import html as html_lib
import re


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}")
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\d{6,}")


def clean_posting_text(text: str, *, max_chars: int) -> str:
    # 1) HTML 태그 제거 2) HTML 엔티티 unescape 3) URL/이메일/전화 토큰 치환
    # 4) 너무 긴 숫자 정리 5) 소문자 6) 공백 정리 7) 길이 제한
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = html_lib.unescape(cleaned)
    cleaned = _URL_RE.sub(" <URL> ", cleaned)
    cleaned = _EMAIL_RE.sub(" <EMAIL> ", cleaned)
    cleaned = _PHONE_RE.sub(" <PHONE> ", cleaned)
    cleaned = _LONG_NUM_RE.sub(" <LONGNUM> ", cleaned)
    cleaned = cleaned.lower()
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


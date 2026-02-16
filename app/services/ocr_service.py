from __future__ import annotations

"""
OCR 서비스 (EasyOCR).

이미지(bytes) 또는 imageUrl → OCR 텍스트 + 간단 진단값 반환.
"""

import html as html_lib
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}")
_WS_RE = re.compile(r"\s+")

# OCR에서 '.'이 빠지거나 공백으로 끊어지는 도메인을 최소 복원하기 위한 whitelist
_TLD_WHITELIST = r"(?:com|net|org|io|ai|co|kr|uk)"
_DOMAIN_PREFIX_RE = re.compile(rf"\b([a-z0-9-]{{3,}})\s+({_TLD_WHITELIST})\b", re.IGNORECASE)
_CO_2LD_RE = re.compile(r"\b(co)\s+(uk|kr)\b", re.IGNORECASE)
_DOMAIN_CO_2LD_RE = re.compile(r"\b([a-z0-9-]{3,})\s+co\s+(uk|kr)\b", re.IGNORECASE)


def _guess_language(text: str) -> str:
    # 한글 비율 > 0.2면 ko, 아니면 en (요구사항)
    t = text.strip()
    if t == "":
        return "en"
    hangul = sum(1 for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3)
    ratio = hangul / max(1, len(t))
    return "ko" if ratio > 0.2 else "en"


def _clean_text(text: str, *, max_chars: int = 20000) -> str:
    # OCR 텍스트는 HTML이 거의 없지만, 엔티티가 들어올 수 있어 unescape 처리
    cleaned = html_lib.unescape(text)

    # 1) "띄어진 도메인" 복원 (예: "capitaldevelopment net" -> "capitaldevelopment.net")
    # 너무 공격적으로 합치지 않도록 앞 토큰을 [a-z0-9-]{3,}로 제한하고, 뒤 토큰은 whitelist 기반
    cleaned = _DOMAIN_CO_2LD_RE.sub(lambda m: f"{m.group(1)}.co.{m.group(2)}", cleaned)
    cleaned = _CO_2LD_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", cleaned)
    cleaned = _DOMAIN_PREFIX_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", cleaned)

    # 2) URL/이메일/전화 토큰화는 기존 유지
    cleaned = _URL_RE.sub(" <URL> ", cleaned)
    cleaned = _EMAIL_RE.sub(" <EMAIL> ", cleaned)
    cleaned = _PHONE_RE.sub(" <PHONE> ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


@dataclass(frozen=True)
class OCRResult:
    text: str
    text_preview: str
    text_length: int
    language_guess: str
    confidence_avg: float | None
    error: str | None = None

    @staticmethod
    def empty(error: str) -> "OCRResult":
        return OCRResult(
            text="",
            text_preview="",
            text_length=0,
            language_guess="en",
            confidence_avg=None,
            error=error,
        )


_READER: Any | None = None


def _get_reader():
    global _READER
    if _READER is not None:
        return _READER
    try:
        import easyocr  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"easyocr import 실패: {e}") from e

    _READER = easyocr.Reader(["ko", "en"], gpu=False)
    return _READER


def _ocr_numpy_image(img) -> tuple[str, float | None]:
    reader = _get_reader()
    # detail=1: (bbox, text, confidence)
    results = reader.readtext(img, detail=1)
    texts: list[str] = []
    confs: list[float] = []
    for item in results:
        try:
            txt = str(item[1])
            conf = float(item[2])
        except Exception:  # noqa: BLE001
            continue
        if txt.strip():
            texts.append(txt.strip())
            confs.append(conf)

    joined = " ".join(texts).strip()
    avg = float(sum(confs) / len(confs)) if confs else None
    return joined, avg


def ocr_from_bytes(image_bytes: bytes) -> OCRResult:
    # bytes → numpy image decode
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # noqa: BLE001
        # 의존성 없으면 OCR을 수행할 수 없음
        return OCRResult.empty(error=f"OCR 의존성(opencv/numpy) 로드 실패: {e}")

    if not image_bytes:
        return OCRResult.empty(error="이미지 바이트가 비어 있습니다.")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return OCRResult.empty(error="이미지 디코딩 실패(지원하지 않는 포맷일 수 있음).")

    try:
        raw_text, conf_avg = _ocr_numpy_image(img)
    except Exception as e:  # noqa: BLE001
        logger.exception("EasyOCR failed: %s", e)
        return OCRResult.empty(error=f"EasyOCR 실패: {e}")

    cleaned = _clean_text(raw_text, max_chars=20000)
    lang = _guess_language(cleaned)
    preview = cleaned[:300]
    return OCRResult(
        text=cleaned,
        text_preview=preview,
        text_length=len(cleaned),
        language_guess=lang,
        confidence_avg=conf_avg,
        error=None,
    )


def ocr_from_url(image_url: str) -> OCRResult:
    try:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return OCRResult.empty(error=f"이미지 다운로드 실패: {e}")

    ctype = (r.headers.get("content-type") or "").lower()
    if ctype and not ctype.startswith("image/"):
        return OCRResult.empty(error=f"content-type이 이미지가 아닙니다: {ctype}")

    return ocr_from_bytes(r.content)

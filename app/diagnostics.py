from __future__ import annotations

"""
입력 텍스트 진단(간단 휴리스틱).

목표:
- 언어를 대략 감지해서 UI에서 "해석 주의" 같은 문구를 노출할 수 있게 함
- 모델이 주로 학습한 도메인(현재는 영어 공고 중심)을 벗어나면 경고

주의:
- 외부 라이브러리 없는 lightweight heuristic입니다.
- 정확한 언어 감지가 필요하면 fastText/langdetect 등으로 교체하세요.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputDiagnostics:
    language: str
    in_domain: bool
    input_confidence: float
    note: str | None = None


def _char_counts(text: str) -> tuple[int, int, int]:
    hangul = 0
    latin = 0
    cjk = 0
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            hangul += 1
        elif (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A):
            latin += 1
        elif 0x4E00 <= o <= 0x9FFF:
            cjk += 1
    return hangul, latin, cjk


def detect_language(text: str) -> str:
    hangul, latin, cjk = _char_counts(text)
    total = hangul + latin + cjk
    if total < 20:
        return "unknown"

    if hangul / total >= 0.6:
        return "ko"
    if latin / total >= 0.6:
        return "en"
    if cjk / total >= 0.6:
        return "zh"
    return "unknown"


def analyze_input(text: str) -> InputDiagnostics:
    text_stripped = text.strip()
    lang = detect_language(text_stripped)

    # MVP 가정: 학습 데이터가 영어 공고 중심인 경우가 많으므로, non-en은 OOD로 표시
    in_domain = lang == "en"

    # input_confidence: 길이(정보량) + 언어 판정 신뢰도를 대략 반영
    hangul, latin, cjk = _char_counts(text_stripped)
    total = hangul + latin + cjk
    dominance = 0.0
    if total > 0:
        dominance = max(hangul, latin, cjk) / total

    length_score = min(1.0, len(text_stripped) / 800.0)
    confidence = float(max(0.0, min(1.0, 0.6 * dominance + 0.4 * length_score)))

    note = None
    if len(text_stripped) < 100:
        in_domain = False
        note = "입력 텍스트가 너무 짧아 해석이 제한될 수 있습니다."
    elif lang in {"ko", "zh", "unknown"}:
        note = "Model trained mostly on English job postings; interpretation may be limited."

    return InputDiagnostics(
        language=lang,
        in_domain=in_domain,
        input_confidence=confidence,
        note=note,
    )


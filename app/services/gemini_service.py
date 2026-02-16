from __future__ import annotations

"""
Gemini polish 서비스.

원칙:
- 템플릿 문장 기반으로 '문장만' 자연스럽게
- 새로운 사실 추가 금지
- 숫자 변경 금지
- 3문장 구조 유지
- 실패/위반 시 템플릿 그대로 반환
"""

import os
from dataclasses import dataclass
import logging
import re

from app.env import load_dotenv_once
from app.services.summary_builder import validate_polished_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiPolishResult:
    # 최종 반환 메시지(검증 실패/예외 시 템플릿으로 fallback 된 값)
    message: str
    used_gemini: bool
    fallback_to_template: bool
    no_change: bool
    error: str | None
    prompt_used: str | None


def _normalize(text: str) -> str:
    # 앞뒤 공백 제거 + 연속 공백 축소 + 줄바꿈 통일
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", "\n", t)
    return t


def _build_prompt(
    *,
    template_message: str,
    trust_score: int,
    trust_label: str,
    fraud_probability: float,
    risk_score: int,
    risk_signals: list[str],
) -> str:
    # 요구사항: 출력은 무조건 한국어 유지 + 3문장 구조 고정 + 숫자 변경 금지 + 사실 추가 금지
    rules = (
        "반드시 한국어로만 작성해.\n"
        "새로운 사실/근거를 절대 추가하지 마.\n"
        "반드시 아래 제공된 신호 목록(riskSignals)과 TEMPLATE 내용만 사용해.\n"
        "숫자(점수/퍼센트)를 절대 변경하지 마.\n"
        "coef/weight/기여도/계수 같은 내부 모델 값은 절대 언급하지 마.\n"
        "템플릿의 의미/구조를 유지하면서 문장만 자연스럽게 다듬어.\n"
        "신호는 '…와 같은 표현이 탐지되어' 수준으로만 부드럽게 언급해.\n"
        "riskSignals가 비어 있으면 '뚜렷한 사기 패턴 표현은 탐지되지 않았습니다' 취지로만 짧게 언급해.\n"
        "TEMPLATE에 '여행금지 지역' 문장이 포함돼 있으면, 그 문장을 삭제하거나 새로운 지역명을 추가하지 마.\n"
        "최종 결과는 한 문단으로만 반환해."
    )
    _ = trust_score, trust_label, fraud_probability, risk_score
    sig_lines = []
    if risk_signals:
        sig_lines.append("riskSignals: " + ", ".join(risk_signals[:3]))
    sig_block = "\n".join(sig_lines) if sig_lines else "riskSignals: (none)"
    return f"{rules}\n\n[SIGNALS]\n{sig_block}\n\n[TEMPLATE]\n{template_message}\n"


def polish_with_gemini(
    *,
    template_message: str,
    trust_score: int,
    trust_label: str,
    fraud_probability: float,
    risk_score: int,
    risk_signals: list[str] | None = None,
    timeout_s: float = 8.0,
) -> GeminiPolishResult:
    # 로컬에서는 .env에 GEMINI_API_KEY를 두는 경우가 많아서, 필요 시 1회 로드
    load_dotenv_once()
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    model = os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash"
    if api_key.strip() == "":
        return GeminiPolishResult(
            message=template_message,
            used_gemini=False,
            fallback_to_template=True,
            no_change=False,
            error="GEMINI_API_KEY not set",
            prompt_used=None,
        )

    prompt = _build_prompt(
        template_message=template_message,
        trust_score=trust_score,
        trust_label=trust_label,
        fraud_probability=fraud_probability,
        risk_score=risk_score,
        risk_signals=list(risk_signals or []),
    )

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        _ = timeout_s  # MVP: SDK timeout 미지원(또는 버전별 상이)이라 직접 제어하지 않음
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
        if not text:
            return GeminiPolishResult(
                message=template_message,
                used_gemini=True,
                fallback_to_template=True,
                no_change=False,
                error="Gemini returned empty text",
                prompt_used=prompt,
            )

        candidate = str(text).strip()
        if not validate_polished_message(template_message, candidate):
            return GeminiPolishResult(
                message=template_message,
                used_gemini=True,
                fallback_to_template=True,
                no_change=False,
                error="Validation failed (rule violation)",
                prompt_used=prompt,
            )

        no_change = _normalize(candidate) == _normalize(template_message)
        if no_change:
            logger.info("Gemini returned identical text (no_change)")

        # 성공 + 검증 통과: 템플릿과 동일해도 fallback은 False
        return GeminiPolishResult(
            message=candidate,
            used_gemini=True,
            fallback_to_template=False,
            no_change=no_change,
            error=None,
            prompt_used=prompt,
        )
    except Exception as e:  # noqa: BLE001
        return GeminiPolishResult(
            message=template_message,
            used_gemini=True,
            fallback_to_template=True,
            no_change=False,
            error=str(e),
            prompt_used=prompt,
        )

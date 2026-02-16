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

from app.env import load_dotenv_once


@dataclass(frozen=True)
class GeminiPolishResult:
    message: str | None
    used_gemini: bool
    prompt_used: str | None


def _build_prompt(
    *,
    template_message: str,
    trust_score: int,
    trust_label: str,
    fraud_probability: float,
    risk_score: int,
) -> str:
    # 요구사항: 출력은 무조건 한국어 유지 + 3문장 구조 고정 + 숫자 변경 금지 + 사실 추가 금지
    rules = (
        "반드시 한국어로만 작성해.\n"
        "반드시 3문장 구조를 그대로 유지해(문장 추가/삭제 금지).\n"
        "새로운 사실/근거를 절대 추가하지 마.\n"
        "숫자(점수/퍼센트)를 절대 변경하지 마.\n"
        "템플릿의 의미/구조를 유지하면서 문장만 자연스럽게 다듬어.\n"
        "최종 결과는 3문장 한 문단만 반환해."
    )
    _ = trust_score, trust_label, fraud_probability, risk_score
    return f"{rules}\n\n[TEMPLATE]\n{template_message}\n"


def polish_with_gemini(
    *,
    template_message: str,
    trust_score: int,
    trust_label: str,
    fraud_probability: float,
    risk_score: int,
    timeout_s: float = 8.0,
) -> GeminiPolishResult:
    # 로컬에서는 .env에 GEMINI_API_KEY를 두는 경우가 많아서, 필요 시 1회 로드
    load_dotenv_once()
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    model = os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash"
    if api_key.strip() == "":
        return GeminiPolishResult(message=None, used_gemini=False, prompt_used=None)

    prompt = _build_prompt(
        template_message=template_message,
        trust_score=trust_score,
        trust_label=trust_label,
        fraud_probability=fraud_probability,
        risk_score=risk_score,
    )

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt, timeout=timeout_s)
        text = getattr(resp, "text", None)
        if not text:
            return GeminiPolishResult(message=None, used_gemini=True, prompt_used=prompt)
        return GeminiPolishResult(message=str(text).strip(), used_gemini=True, prompt_used=prompt)
    except Exception:
        return GeminiPolishResult(message=None, used_gemini=True, prompt_used=prompt)

from __future__ import annotations

"""
Gemini 서비스.

기능:
1. analyze_with_gemini: 경계 구간 케이스에서 Gemini가 직접 사기 여부를 심층 판단 (ML + LLM 앙상블)
2. polish_with_gemini: 템플릿 문장을 자연스럽게 다듬기 (ML 단독 판단 케이스에서 사용)
"""

import json
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


@dataclass(frozen=True)
class GeminiAnalysisResult:
    fraud_probability: float
    risk_signals: list[str]
    reasoning: str
    used_gemini: bool
    error: str | None


def _build_analysis_prompt(
    *,
    ocr_text: str,
    ml_probability: float,
    ml_risk_signals: list[str],
) -> str:
    signals_str = ", ".join(ml_risk_signals) if ml_risk_signals else "(없음)"
    return (
        "당신은 채용 사기 탐지 전문가입니다. 아래 채용 공고 텍스트를 분석하고 "
        "정확히 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.\n\n"
        f"[ML 1차 판단]\n사기 확률: {ml_probability:.1%}\n탐지된 패턴: {signals_str}\n\n"
        f"[채용 공고 텍스트]\n{ocr_text[:3000]}\n\n"
        "응답 형식 (JSON만, 마크다운 코드블록 없이):\n"
        "{\n"
        '  "fraud_probability": <0.0~1.0 사이 소수>,\n'
        '  "risk_signals": ["신호1", "신호2"],\n'
        '  "reasoning": "<판단 근거 2~3문장, 한국어>"\n'
        "}"
    )


def analyze_with_gemini(
    *,
    ocr_text: str,
    ml_probability: float,
    ml_risk_signals: list[str] | None = None,
) -> GeminiAnalysisResult:
    """경계 구간(20%~80%) 케이스에 대해 Gemini가 ML 판단을 검토하고 최종 확률을 제시합니다."""
    load_dotenv_once()
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    model = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

    if api_key.strip() == "":
        return GeminiAnalysisResult(
            fraud_probability=ml_probability,
            risk_signals=list(ml_risk_signals or []),
            reasoning="",
            used_gemini=False,
            error="GEMINI_API_KEY not set",
        )

    prompt = _build_analysis_prompt(
        ocr_text=ocr_text,
        ml_probability=ml_probability,
        ml_risk_signals=list(ml_risk_signals or []),
    )

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
        if not text:
            return GeminiAnalysisResult(
                fraud_probability=ml_probability,
                risk_signals=list(ml_risk_signals or []),
                reasoning="",
                used_gemini=True,
                error="Gemini returned empty text",
            )

        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        data = json.loads(clean)
        fraud_prob = float(data.get("fraud_probability", ml_probability))
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        signals = [str(s) for s in data.get("risk_signals", [])][:5]
        reasoning = str(data.get("reasoning", "")).strip()

        return GeminiAnalysisResult(
            fraud_probability=fraud_prob,
            risk_signals=signals,
            reasoning=reasoning,
            used_gemini=True,
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        return GeminiAnalysisResult(
            fraud_probability=ml_probability,
            risk_signals=list(ml_risk_signals or []),
            reasoning="",
            used_gemini=True,
            error=str(e),
        )


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
    model = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
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

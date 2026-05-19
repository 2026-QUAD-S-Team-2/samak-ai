from __future__ import annotations

"""
Gemini 서비스.

기능:
1. analyze_image_with_gemini_vision: 이미지를 Gemini Vision으로 직접 분석 (멀티모달, 모든 요청에서 실행)
2. polish_with_gemini: 템플릿 문장을 자연스럽게 다듬기
"""

import json
import os
from dataclasses import dataclass, field
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
class GeminiVisionResult:
    fraud_probability: float
    risk_signals: list[str]
    reasoning: str
    used_gemini: bool
    error: str | None
    domains_found: list[str] = field(default_factory=list)
    regions_mentioned: list[str] = field(default_factory=list)
    summary_message: str = ""
    risk_quotes: list[str] = field(default_factory=list)


def _detect_mime_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _build_vision_prompt() -> str:
    return (
        "당신은 채용 사기 탐지 전문가입니다. 첨부된 이미지는 채용 공고 또는 채팅 캡처본입니다.\n"
        "이미지를 보고 아래 기준으로 사기 여부를 판단하여 정확히 JSON 형식으로만 응답하세요.\n"
        "다른 텍스트나 마크다운 코드블록은 절대 포함하지 마세요.\n\n"
        "[분석 기준]\n"
        "- 비정상적으로 높은 급여 약속\n"
        "- 선입금/보증금/장비 구매 요구\n"
        "- 개인정보 즉시 요구 (계좌번호, 주민번호 등)\n"
        "- 회사 정보 불명확 (이름/주소/연락처 없음)\n"
        "- 문법 오류, 번역 투 문체\n"
        "- 여행금지 국가·지역 파견 근무 제안\n"
        "- 카카오톡/텔레그램 등 비공식 채널 연락 요구\n"
        "- 레이아웃이 조잡하거나 로고가 위조처럼 보임\n\n"
        "[추가 추출 항목]\n"
        "- domains_found: 이미지에 보이는 모든 URL·도메인 주소 (예: example.com, t.me/xxx). 없으면 빈 배열.\n"
        "- regions_mentioned: 이미지에 언급된 모든 국가·지역명을 영문 소문자로 추출 (예: myanmar, cambodia, myawaddy). 없으면 빈 배열.\n"
        "- risk_quotes: 공고 이미지에서 사기 신호와 직접 관련된 의심 문구를 원문 그대로 최대 3개 추출. 없으면 빈 배열. 문구를 지어내지 말고 이미지에 실제로 보이는 텍스트만 추출할 것.\n\n"
        "[summary_message 작성 규칙]\n"
        "- 구직자에게 전달하는 자연스러운 한국어 문장 1~2개.\n"
        "- 반드시 한국어로만 작성. 숫자·퍼센트·점수는 절대 포함 금지.\n"
        "- 탐지된 주요 위험 신호를 구체적으로 언급 (없으면 '특이한 사기 패턴은 발견되지 않았습니다' 수준으로).\n"
        "- 구직자에게 권고 행동을 간략히 포함할 것.\n\n"
        "응답 형식 (JSON만, 마크다운 코드블록 없이):\n"
        "{\n"
        '  "fraud_probability": <0.0~1.0 사이 소수>,\n'
        '  "risk_signals": ["신호1", "신호2"],\n'
        '  "reasoning": "<판단 근거 2~3문장, 한국어>",\n'
        '  "domains_found": ["domain1.com"],\n'
        '  "regions_mentioned": ["country1", "region1"],\n'
        '  "risk_quotes": ["이미지에서 그대로 발췌한 의심 문구1", "의심 문구2"],\n'
        '  "summary_message": "<구직자 대상 자연스러운 한국어 요약. 숫자/% 제외.>"\n'
        "}"
    )


def analyze_image_with_gemini_vision(image_bytes: bytes) -> GeminiVisionResult:
    """이미지 bytes를 Gemini Vision으로 직접 분석하여 사기 확률을 반환합니다."""
    load_dotenv_once()
    api_key = os.environ.get("GEMINI_API_KEY") or ""
    model = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

    if not api_key.strip():
        return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=False, error="GEMINI_API_KEY not set")
    if not image_bytes:
        return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=False, error="image_bytes is empty")
    if len(image_bytes) > 20 * 1024 * 1024:
        return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=False, error="이미지 크기 20MB 초과")

    mime_type = _detect_mime_type(image_bytes)
    prompt_text = _build_vision_prompt()

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        resp = client.models.generate_content(model=model, contents=[prompt_text, image_part])
        text = getattr(resp, "text", None)
        if not text:
            return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=True, error="Gemini Vision returned empty text")

        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        data = json.loads(clean)
        fraud_prob = max(0.0, min(1.0, float(data.get("fraud_probability", 0.5))))
        signals = [str(s) for s in data.get("risk_signals", [])][:5]
        reasoning = str(data.get("reasoning", "")).strip()
        domains_found = [str(d) for d in data.get("domains_found", [])][:10]
        regions_mentioned = [str(r).lower() for r in data.get("regions_mentioned", [])][:10]
        risk_quotes = [str(q).strip() for q in data.get("risk_quotes", []) if str(q).strip()][:3]
        summary_message = str(data.get("summary_message", "")).strip()

        return GeminiVisionResult(
            fraud_probability=fraud_prob,
            risk_signals=signals,
            reasoning=reasoning,
            used_gemini=True,
            error=None,
            domains_found=domains_found,
            regions_mentioned=regions_mentioned,
            summary_message=summary_message,
            risk_quotes=risk_quotes,
        )
    except Exception as e:  # noqa: BLE001
        return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=True, error=str(e))


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

from __future__ import annotations

"""
이미지 기반 공고/채팅 분석 API 라우트.

입력:
- application/json: { "imageUrls": ["https://..."], "countryCode": "KR", "salary": "..." } (단일 입력 호환: imageUrl)

처리:
이미지 → OCR → ML 추론 → scoring → 템플릿 summary → (옵션) Gemini polish → 응답
"""

import logging
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api_models import AnalyzeImageRequest, AnalyzeImageResponse, AnalyzeImagesResponse
from app.ml.message_risk_rules import extract_risk_signals
from app.ml.ml_baseline import BaselineModel, ModelArtifactsError
from app.ml.risk_regions import match_risk_regions_by_country_code
from app.services.backend_push import push_analysis_result
from app.services.gemini_service import polish_with_gemini
from app.services.ocr_service import ocr_from_bytes
from app.services.scoring_service import PredictionScores, score_prediction, ui_policy_from_probability
from app.services.summary_builder import build_template_message
from app.services.wage_service import WageScores, apply_wage_adjustments, cap_scores, decide_wage_warning


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/analyze", tags=["analyze"])


def _is_valid_image_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    if p.scheme not in {"http", "https"}:
        return False
    if not p.netloc:
        return False
    return True


async def _download_image_bytes(image_url: str) -> bytes:
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(image_url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패: {e}") from e

    if resp.status_code < 200 or resp.status_code >= 300:
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패: HTTP {resp.status_code}")

    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패: content-type={content_type}")

    data = resp.content or b""
    if not data:
        raise HTTPException(status_code=400, detail="이미지 다운로드 실패: empty body")
    return data


@router.post("/image", response_model=AnalyzeImageResponse | AnalyzeImagesResponse)
async def analyze_image(payload: AnalyzeImageRequest, background_tasks: BackgroundTasks) -> dict:
    """
    POST /v1/analyze/image

    - application/json: imageUrls(필수, 1개 이상), countryCode(필수), salary(선택), debug(선택)
      - 단일 입력 호환: imageUrl
    """
    image_urls = [str(u or "").strip() for u in (payload.imageUrls or [])]
    image_urls = [u for u in image_urls if u]
    if not image_urls:
        raise HTTPException(status_code=422, detail="imageUrls는 최소 1개 이상이어야 합니다.")

    for i, u in enumerate(image_urls):
        if not _is_valid_image_url(u):
            raise HTTPException(status_code=400, detail=f"imageUrls[{i}]가 올바르지 않습니다. (http/https URL 필요)")

    country_code = payload.countryCode
    travel_ban_matched = match_risk_regions_by_country_code(country_code)

    salary_text = payload.salaryText
    has_salary = salary_text is not None and str(salary_text).strip() != ""
    salary_line = f"\n[salary] {str(salary_text)}" if has_salary else ""

    wage_decision = await decide_wage_warning(country_code=country_code, salary_text=salary_text)
    wage_message = wage_decision.warning_message
    if wage_decision.warning_kind in {"min_wage_low", "high_salary"}:
        wage_message_type = "WARNING"
    elif wage_decision.warning_kind in {"parse_error", "mismatch"}:
        wage_message_type = "ERROR"
    elif wage_decision.warning_kind == "missing":
        wage_message_type = "INFO" if wage_message else "NONE"
    else:
        wage_message_type = "NONE"

    async def _analyze_one_image_url(image_url: str) -> dict:
        image_bytes = await _download_image_bytes(image_url)

        try:
            ocr = ocr_from_bytes(image_bytes)
        except Exception as e:  # noqa: BLE001
            logger.exception("OCR failed: %s", e)
            # 기존 비즈니스 로직 유지: OCR 실패 자체는 200 응답으로 처리(UNKNOWN)
            from app.services.ocr_service import OCRResult  # local import to avoid unused import in happy path

            ocr = OCRResult.empty(error=f"OCR 실패: {e}")

        company_name = None

        # 안전장치: OCR 텍스트가 너무 짧으면 ML 추론을 건너뜁니다.
        if ocr.text_length < 30:
            analysis_id = str(uuid4())
            message = "텍스트 추출에 실패했습니다. 더 선명한 이미지로 다시 시도해 주세요."
            resp: dict = {
                "analysisId": analysis_id,
                "fraudProbability": None,
                "riskScore": None,
                "riskLevel": "UNKNOWN",
                "riskSignals": [],
                "travelBanRegionsMatched": travel_ban_matched,
                "wageMessageType": wage_message_type,
                "wageMessage": wage_message,
                "message": message,
            }
            background_tasks.add_task(push_analysis_result, resp)
            return resp

        # ML baseline (로컬 아티팩트)
        used_model = False
        try:
            model = BaselineModel.load_default()
            cleaned_input = model.get_cleaned_input_from_ocr(ocr.text)
            cleaned_input = cleaned_input + salary_line
            fraud_prob = model.predict_proba(cleaned_input)
            risk_signals = extract_risk_signals(cleaned_input, top_k=3)
            threshold_used = model.threshold
            used_model = True
        except ModelArtifactsError as e:
            logger.error("Model load failed: %s", e)
            fraud_prob = 0.0
            risk_signals = []
            threshold_used = 0.5

        scores: PredictionScores = score_prediction(fraud_prob, threshold_used)

        # 점수 cap (100은 사용하지 않음)
        base_risk_score = int(min(99, scores.risk_score))
        if travel_ban_matched:
            base_risk_score = int(min(99, round(base_risk_score * 1.15)))
        trust_score = int(min(99, max(0, 100 - base_risk_score)))

        # 임금 경고 기반 점수 조정(점수 None-safe)
        adjusted = apply_wage_adjustments(
            scores=WageScores(risk_score=base_risk_score, trust_score=trust_score, fraud_probability=float(fraud_prob)),
            warning_kind=wage_decision.warning_kind,
        )
        adjusted = cap_scores(adjusted)
        base_risk_score = adjusted.risk_score if adjusted.risk_score is not None else base_risk_score
        trust_score = adjusted.trust_score if adjusted.trust_score is not None else trust_score
        fraud_prob = adjusted.fraud_probability if adjusted.fraud_probability is not None else fraud_prob
        ui_risk_level, ui_trust_label = ui_policy_from_probability(float(fraud_prob))

        template_message = build_template_message(
            company_name=company_name,
            trust_score=trust_score,
            risk_score=base_risk_score,
            ui_trust_label=ui_trust_label,
            has_signals=bool(risk_signals),
            travel_ban_regions=travel_ban_matched,
        )
        # 임금 메시지는 기본적으로 별도 필드로 전달하고,
        # "경고"에 해당하는 경우만 분석 요약(message)에 덧붙입니다.
        if wage_message_type == "WARNING" and wage_message:
            template_message = template_message + " " + wage_message

        polished = template_message
        prompt_used: str | None = None
        used_gemini = False
        fallback_to_template = True
        no_change = False
        gemini_error: str | None = None
        try:
            gemini_out = polish_with_gemini(
                template_message=template_message,
                trust_score=trust_score,
                trust_label=scores.ui_trust_label,
                fraud_probability=fraud_prob,
                risk_score=base_risk_score,
                risk_signals=risk_signals,
            )
            prompt_used = gemini_out.prompt_used
            polished = gemini_out.message
            used_gemini = gemini_out.used_gemini
            fallback_to_template = gemini_out.fallback_to_template
            no_change = gemini_out.no_change
            gemini_error = gemini_out.error
        except Exception as e:  # noqa: BLE001
            logger.exception("Gemini polish failed: %s", e)

        analysis_id = str(uuid4())

        resp: dict = {
            "analysisId": analysis_id,
            "fraudProbability": float(min(0.99, max(0.0, float(fraud_prob)))),
            "riskScore": base_risk_score,
            "riskLevel": ui_risk_level,
            "riskSignals": risk_signals,
            "travelBanRegionsMatched": travel_ban_matched,
            "wageMessageType": wage_message_type,
            "wageMessage": wage_message,
            "message": polished,
        }

        background_tasks.add_task(push_analysis_result, resp)

        if payload.debug:
            logger.info(
                "debug: usedModel=%s usedGemini=%s fallback=%s noChange=%s geminiError=%s ocrError=%s promptUsed_len=%s",
                used_model,
                used_gemini,
                fallback_to_template,
                no_change,
                gemini_error,
                ocr.error,
                len(prompt_used or ""),
            )

        return resp

    if len(image_urls) == 1:
        return await _analyze_one_image_url(image_urls[0])

    results = [await _analyze_one_image_url(u) for u in image_urls]
    return {"results": results}

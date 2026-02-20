from __future__ import annotations

"""
이미지 기반 공고/채팅 분석 API 라우트.

입력:
- application/json: { "imageUrls": ["https://..."] } (단일 입력 호환: imageUrl)

처리:
이미지 → OCR → ML 추론 → scoring → 템플릿 summary → (옵션) Gemini polish → 응답
"""

import logging
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api_models import AnalyzeImageRequest, AnalyzeImageResponse, AnalyzeImagesResponse
from app.ml.ml_baseline import BaselineModel, ModelArtifactsError
from app.ml.risk_regions import find_risk_regions
from app.services.backend_push import push_analysis_result
from app.services.gemini_service import polish_with_gemini
from app.services.ocr_service import ocr_from_bytes
from app.services.scoring_service import PredictionScores, score_prediction
from app.services.summary_builder import build_template_message


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


async def _analyze_one_image_url(image_url: str, *, debug: bool, background_tasks: BackgroundTasks) -> dict:
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
            "travelBanRegionsMatched": [],
            "message": message,
        }
        # 백엔드로 push (실패해도 응답은 유지)
        background_tasks.add_task(push_analysis_result, resp)
        return resp

    # ML baseline (로컬 아티팩트)
    try:
        model = BaselineModel.load_default()
        fraud_prob = model.predict_proba_from_ocr(ocr.text)
        risk_signals = model.risk_signals_from_ocr(ocr.text, top_k=3)
        cleaned_input = model.get_cleaned_input_from_ocr(ocr.text)
        risk_regions = find_risk_regions(cleaned_input, top_k=5)
        threshold_used = model.threshold
    except ModelArtifactsError as e:
        # 모델 로딩 실패해도 API는 200을 유지(템플릿 message로 fallback)
        logger.error("Model load failed: %s", e)
        fraud_prob = 0.0
        risk_signals = []
        risk_regions = []
        threshold_used = 0.5

    scores: PredictionScores = score_prediction(fraud_prob, threshold_used)

    template_message = build_template_message(
        company_name=company_name,
        trust_score=scores.trust_score,
        risk_score=scores.risk_score,
        ui_trust_label=scores.ui_trust_label,
        has_signals=bool(risk_signals),
        travel_ban_regions=risk_regions,
    )

    polished = template_message
    prompt_used: str | None = None
    used_gemini = False
    fallback_to_template = True
    no_change = False
    gemini_error: str | None = None
    try:
        gemini_out = polish_with_gemini(
            template_message=template_message,
            trust_score=scores.trust_score,
            trust_label=scores.ui_trust_label,
            fraud_probability=fraud_prob,
            risk_score=scores.risk_score,
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
        "fraudProbability": float(fraud_prob),
        "riskScore": scores.risk_score,
        "riskLevel": scores.ui_risk_level,
        "riskSignals": risk_signals,
        "travelBanRegionsMatched": risk_regions,
        "message": polished,
    }

    # 백엔드로 push (실패해도 응답은 유지). 응답 지연을 줄이기 위해 background task로 실행.
    background_tasks.add_task(push_analysis_result, resp)

    if debug:
        # 응답 스키마는 flat JSON로 고정하고, 디버그는 로그로만 남깁니다.
        logger.info(
            "debug: usedGemini=%s fallback=%s noChange=%s geminiError=%s ocrError=%s promptUsed_len=%s",
            used_gemini,
            fallback_to_template,
            no_change,
            gemini_error,
            ocr.error,
            len(prompt_used or ""),
        )

    return resp


@router.post("/image", response_model=AnalyzeImageResponse | AnalyzeImagesResponse)
async def analyze_image(payload: AnalyzeImageRequest, background_tasks: BackgroundTasks) -> dict:
    """
    POST /v1/analyze/image

    - application/json: imageUrls (필수, 1개 이상)
      - 단일 입력 호환: imageUrl
    """
    image_urls = [str(u or "").strip() for u in (payload.imageUrls or [])]
    image_urls = [u for u in image_urls if u]
    if not image_urls:
        raise HTTPException(status_code=422, detail="imageUrls는 최소 1개 이상이어야 합니다.")

    for i, u in enumerate(image_urls):
        if not _is_valid_image_url(u):
            raise HTTPException(status_code=400, detail=f"imageUrls[{i}]가 올바르지 않습니다. (http/https URL 필요)")

    if len(image_urls) == 1:
        return await _analyze_one_image_url(image_urls[0], debug=payload.debug, background_tasks=background_tasks)

    results = [
        await _analyze_one_image_url(u, debug=payload.debug, background_tasks=background_tasks) for u in image_urls
    ]
    return {"results": results}

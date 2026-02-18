from __future__ import annotations

"""
이미지 기반 공고/채팅 분석 API 라우트.

입력:
- multipart/form-data: multipartFile (이미지)
- 또는 application/json: { "imageUrl": "https://..." }

처리:
이미지 → OCR → ML 추론 → scoring → 템플릿 summary → (옵션) Gemini polish → 응답
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Request

from app.api_models import AnalyzeImageResponse
from app.ml.ml_baseline import BaselineModel, ModelArtifactsError
from app.ml.risk_regions import find_risk_regions
from app.services.backend_push import push_analysis_result
from app.services.gemini_service import polish_with_gemini
from app.services.ocr_service import OCRResult, ocr_from_bytes, ocr_from_url
from app.services.scoring_service import PredictionScores, score_prediction
from app.services.summary_builder import build_template_message


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/analyze", tags=["analyze"])


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _get_str(meta: dict[str, object], key: str) -> str | None:
    v = meta.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


@router.post("/image", response_model=AnalyzeImageResponse)
async def analyze_image(request: Request, debug: bool = False, background_tasks: BackgroundTasks = None) -> dict:
    """
    POST /v1/analyze/image

    - multipart/form-data: multipartFile (필수) + (옵션) meta fields
    - application/json: imageUrl (필수) + (옵션) meta fields
    """
    content_type = (request.headers.get("content-type") or "").lower()

    image_bytes: bytes | None = None
    image_url: str | None = None
    meta: dict[str, object] = {}

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("multipartFile")
        if file is not None:
            # UploadFile
            image_bytes = await file.read()  # type: ignore[union-attr]
        # meta optional
        for k in ["companyName", "countryCode", "region", "channel", "sourceUrl", "type"]:
            if k in form:
                meta[k] = str(form.get(k) or "")
    else:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if isinstance(body, dict):
            image_url = _get_str(body, "imageUrl")
            meta_obj = body.get("meta")
            if isinstance(meta_obj, dict):
                meta = meta_obj
            else:
                # 메타를 최상위로 주는 케이스도 허용(편의)
                for k in ["companyName", "countryCode", "region", "channel", "sourceUrl", "type"]:
                    if k in body:
                        meta[k] = body.get(k)

    if image_bytes is None and not image_url:
        # 요구사항: 실패해도 200. (백엔드가 UI 표시를 계속 할 수 있게)
        ocr: OCRResult = OCRResult.empty(error="이미지 파일(multipartFile) 또는 imageUrl이 필요합니다.")
    else:
        try:
            if image_bytes is not None:
                ocr = ocr_from_bytes(image_bytes)
            else:
                ocr = ocr_from_url(image_url or "")
        except Exception as e:  # noqa: BLE001
            logger.exception("OCR failed: %s", e)
            ocr = OCRResult.empty(error=f"OCR 실패: {e}")

    analysis_type = (_get_str(meta, "type") or "JOB_POST").upper()
    if analysis_type not in {"JOB_POST", "MESSAGE"}:
        analysis_type = "JOB_POST"

    company_name = _get_str(meta, "companyName")

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
        if background_tasks is not None:
            background_tasks.add_task(push_analysis_result, resp)
        else:
            push_analysis_result(resp)
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
        model = None  # type: ignore[assignment]

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
    if background_tasks is not None:
        background_tasks.add_task(push_analysis_result, resp)
    else:
        push_analysis_result(resp)

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

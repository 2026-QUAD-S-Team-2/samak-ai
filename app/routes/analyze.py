from __future__ import annotations

"""
이미지 기반 공고/채팅 분석 API 라우트.

입력:
- multipart/form-data: multipartFile (이미지)
- 또는 application/json: { "imageUrl": "..." } (단일) 또는 { "imageUrls": [...] } (복수)

처리:
이미지 → OCR → ML 추론 → Confidence Gating → (경계 구간) Gemini 심층 분석
→ 점수 계산 → 메시지 생성 → 응답
"""

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Request

from app.ml.ml_baseline import BaselineModel, ModelArtifactsError
from app.ml.risk_regions import find_risk_regions
from app.services.gemini_service import analyze_with_gemini, polish_with_gemini
from app.services.ocr_service import OCRResult, ocr_from_bytes, ocr_from_url
from app.services.scoring_service import PredictionScores, score_prediction
from app.services.summary_builder import build_template_message


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/analyze", tags=["analyze"])

# Confidence gating 임계값: 이 구간의 케이스만 Gemini 심층 분석 호출
_BORDER_LOW = 0.20
_BORDER_HIGH = 0.80


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


async def _run_analysis(
    image_bytes: bytes | None,
    image_url: str | None,
    meta: dict[str, object],
    debug: bool,
) -> dict:
    """단일 이미지에 대한 전체 분석 파이프라인."""
    if image_bytes is None and not image_url:
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
            "type": analysis_type,
            "ocr": {
                "textPreview": ocr.text_preview,
                "textLength": ocr.text_length,
                "languageGuess": ocr.language_guess,
                "confidenceAvg": ocr.confidence_avg,
            },
            "mlPrediction": {
                "modelVersion": "fraud-baseline-v1.0.0",
                "fraudProbability": None,
                "riskScore": None,
                "riskLevel": None,
                "thresholdUsed": None,
            },
            "travelBanRegionsMatched": [],
            "explanation": {
                "riskSignals": [],
                "note": "Signals are matched against predefined scam-pattern rules.",
            },
            "ui": {"riskLevel": "UNKNOWN", "trustLabel": None, "trustScore": None},
            "analysisSummary": {"score": None, "label": None, "message": message},
        }
        if debug:
            resp["debug"] = {"ocrError": ocr.error}
        return resp

    # ML 1차 판단
    used_model = False
    cleaned_input = ""
    model = None  # type: ignore[assignment]
    try:
        model = BaselineModel.load_default()
        fraud_prob = model.predict_proba_from_ocr(ocr.text)
        risk_signals = model.risk_signals_from_ocr(ocr.text, top_k=3)
        cleaned_input = model.get_cleaned_input_from_ocr(ocr.text)
        risk_regions = find_risk_regions(cleaned_input, top_k=5)
        threshold_used = model.threshold
        model_version = model.model_version
        used_model = True
    except ModelArtifactsError as e:
        logger.error("Model load failed: %s", e)
        fraud_prob = 0.0
        risk_signals = []
        risk_regions = []
        threshold_used = 0.5
        model_version = "fraud-baseline-v1.0.0"

    # Confidence gating: 경계 구간(20%~80%)에서만 Gemini 심층 분석 (LLM as a Judge)
    gemini_reasoning = ""
    used_gemini_for_analysis = False
    if used_model and _BORDER_LOW <= float(fraud_prob) <= _BORDER_HIGH:
        try:
            gemini_analysis = analyze_with_gemini(
                ocr_text=cleaned_input,
                ml_probability=float(fraud_prob),
                ml_risk_signals=risk_signals,
            )
            if gemini_analysis.used_gemini and not gemini_analysis.error:
                # 가중 평균: ML 40% + Gemini 60%
                fraud_prob = 0.4 * float(fraud_prob) + 0.6 * gemini_analysis.fraud_probability
                if gemini_analysis.risk_signals:
                    risk_signals = gemini_analysis.risk_signals
                gemini_reasoning = gemini_analysis.reasoning
                used_gemini_for_analysis = True
        except Exception as e:  # noqa: BLE001
            logger.exception("Gemini analysis failed: %s", e)

    scores: PredictionScores = score_prediction(fraud_prob, threshold_used)

    # 메시지 생성
    polished = ""
    prompt_used: str | None = None
    used_gemini = False
    fallback_to_template = True
    no_change = False
    gemini_error: str | None = None

    if used_gemini_for_analysis and gemini_reasoning:
        # Gemini가 직접 심층 판단한 케이스: reasoning을 최종 메시지로 사용
        polished = gemini_reasoning
        used_gemini = True
        fallback_to_template = False
    else:
        # ML 단독 판단 케이스: 템플릿 생성 후 Gemini polish
        template_message = build_template_message(
            company_name=company_name,
            trust_score=scores.trust_score,
            risk_score=scores.risk_score,
            ui_trust_label=scores.ui_trust_label,
            has_signals=bool(risk_signals),
            travel_ban_regions=risk_regions,
        )
        polished = template_message
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
        "type": analysis_type,
        "travelBanRegionsMatched": risk_regions,
        "ocr": {
            "textPreview": ocr.text_preview,
            "textLength": ocr.text_length,
            "languageGuess": ocr.language_guess,
            "confidenceAvg": ocr.confidence_avg,
        },
        "mlPrediction": {
            "modelVersion": model_version,
            "fraudProbability": float(fraud_prob),
            "riskScore": scores.risk_score,
            "riskLevel": scores.model_risk_level,
            "thresholdUsed": float(threshold_used),
        },
        "explanation": {
            "riskSignals": risk_signals,
            "note": "Signals are matched against predefined scam-pattern rules.",
        },
        "ui": {
            "riskLevel": scores.ui_risk_level,
            "trustLabel": scores.ui_trust_label,
            "trustScore": scores.trust_score,
        },
        "analysisSummary": {
            "score": scores.trust_score,
            "label": scores.ui_trust_label,
            "message": polished,
        },
    }

    if debug:
        resp["debug"] = {
            "usedGeminiAnalysis": used_gemini_for_analysis,
            "usedGemini": used_gemini,
            "fallbackToTemplate": fallback_to_template,
            "noChange": no_change,
            "promptUsed": prompt_used,
            "geminiError": gemini_error,
            "ocrError": ocr.error,
        }
        if model is not None:
            resp["debug"]["inputStructured"] = model.structure_ocr_text(ocr.text)
            resp["debug"]["inputCleaned"] = model.get_cleaned_input_from_ocr(ocr.text)
            resp["debug"]["explanation"] = {"riskSignals": risk_signals}
            resp["debug"]["riskRegionsMatched"] = risk_regions

    return resp


@router.post("/image")
async def analyze_image(request: Request, debug: bool = False) -> dict:
    """
    POST /v1/analyze/image

    - multipart/form-data: multipartFile (필수) + (옵션) meta fields
    - application/json: imageUrl (단일) 또는 imageUrls (복수, 5장 이상 병렬 처리) + (옵션) meta fields
    """
    content_type = (request.headers.get("content-type") or "").lower()

    image_bytes: bytes | None = None
    image_url: str | None = None
    image_urls: list[str] = []
    meta: dict[str, object] = {}

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("multipartFile")
        if file is not None:
            image_bytes = await file.read()  # type: ignore[union-attr]
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
            urls_raw = body.get("imageUrls")
            if isinstance(urls_raw, list):
                image_urls = [str(u).strip() for u in urls_raw if u and str(u).strip()]
            meta_obj = body.get("meta")
            if isinstance(meta_obj, dict):
                meta = meta_obj
            else:
                for k in ["companyName", "countryCode", "region", "channel", "sourceUrl", "type"]:
                    if k in body:
                        meta[k] = body.get(k)

    # 복수 이미지 처리
    if image_urls:
        if len(image_urls) >= 5:
            # 5장 이상: asyncio.gather로 병렬 처리 (I/O bound 구간에서 속도 향상)
            results = list(await asyncio.gather(
                *[_run_analysis(None, url, meta, debug) for url in image_urls]
            ))
        else:
            results = [await _run_analysis(None, url, meta, debug) for url in image_urls]
        return {"results": results}

    # 단일 이미지 처리
    return await _run_analysis(image_bytes, image_url, meta, debug)

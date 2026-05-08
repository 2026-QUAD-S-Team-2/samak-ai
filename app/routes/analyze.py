from __future__ import annotations

"""
이미지 기반 공고/채팅 분석 API 라우트.

입력:
- multipart/form-data: multipartFile (이미지)
- 또는 application/json: { "imageUrl": "..." } (단일) 또는 { "imageUrls": [...] } (복수)

처리:
이미지 → [OCR + Gemini Vision 병렬] → 규칙 기반 신호 추출 → 점수 계산 → 메시지 생성 → 응답
"""

import asyncio
import logging
import re
import requests
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.ml.message_risk_rules import extract_risk_signals
from app.ml.risk_regions import find_risk_regions
from app.ml.scam_domains import find_scam_domains
from app.services.gemini_service import GeminiVisionResult, analyze_image_with_gemini_vision, polish_with_gemini
from app.services.ocr_service import OCRResult, ocr_from_bytes
from app.services.scoring_service import PredictionScores, score_prediction
from app.services.summary_builder import build_template_message


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/analyze", tags=["analyze"])

_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
MAX_IMAGE_BATCH = 10
FRAUD_THRESHOLD = 0.6242
MODEL_VERSION = "gemini-rule-v1.0.0"


def _get_str(meta: dict[str, object], key: str) -> str | None:
    v = meta.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


async def _download_image_bytes(url: str) -> bytes:
    try:
        r = await asyncio.to_thread(requests.get, url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패: {e}") from e
    return r.content


async def _run_analysis(
    image_bytes: bytes | None,
    meta: dict[str, object],
    debug: bool,
) -> dict:
    """단일 이미지에 대한 전체 분석 파이프라인."""

    async def _run_ocr() -> OCRResult:
        if not image_bytes:
            return OCRResult.empty(error="이미지 바이트가 없습니다.")
        try:
            return await asyncio.to_thread(ocr_from_bytes, image_bytes)
        except Exception as e:  # noqa: BLE001
            logger.exception("OCR failed: %s", e)
            return OCRResult.empty(error=f"OCR 실패: {e}")

    async def _run_vision() -> GeminiVisionResult:
        if not image_bytes:
            return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=False, error="image_bytes is empty")
        try:
            return await asyncio.to_thread(analyze_image_with_gemini_vision, image_bytes)
        except Exception as e:  # noqa: BLE001
            logger.exception("Gemini Vision failed: %s", e)
            return GeminiVisionResult(fraud_probability=0.5, risk_signals=[], reasoning="", used_gemini=True, error=str(e))

    ocr, vision_result = await asyncio.gather(_run_ocr(), _run_vision())

    analysis_type = (_get_str(meta, "type") or "JOB_POST").upper()
    if analysis_type not in {"JOB_POST", "MESSAGE"}:
        analysis_type = "JOB_POST"

    company_name = _get_str(meta, "companyName")

    # OCR도 실패하고 Gemini도 실패한 경우에만 UNKNOWN 조기 반환
    if ocr.text_length < 30 and (not vision_result.used_gemini or vision_result.error):
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
                "modelVersion": MODEL_VERSION,
                "fraudProbability": None,
                "riskScore": None,
                "riskLevel": None,
                "thresholdUsed": None,
            },
            "travelBanRegionsMatched": [],
            "scamDomainsMatched": [],
            "explanation": {
                "riskSignals": [],
                "note": "Signals are matched against predefined scam-pattern rules.",
            },
            "ui": {"riskLevel": "UNKNOWN", "trustLabel": None, "trustScore": None},
            "analysisSummary": {"score": None, "label": None, "message": message},
        }
        if debug:
            resp["debug"] = {"ocrError": ocr.error, "visionError": vision_result.error}
        return resp

    # 규칙 기반 신호 추출 (OCR 텍스트가 충분한 경우)
    fraud_prob: float = 0.5
    risk_signals: list[str] = []
    risk_regions: list[str] = []
    scam_domains: list[str] = []

    if ocr.text_length >= 30:
        risk_signals = extract_risk_signals(ocr.text, top_k=3)
        risk_regions = find_risk_regions(ocr.text, top_k=5)
        scam_domains = find_scam_domains(ocr.text)

    # Gemini Vision 결과 적용
    used_vision = False
    vision_error: str | None = None
    if vision_result.used_gemini and not vision_result.error:
        fraud_prob = vision_result.fraud_probability
        if vision_result.risk_signals:
            seen = {s.lower() for s in vision_result.risk_signals}
            extra = [s for s in risk_signals if s.lower() not in seen]
            risk_signals = (vision_result.risk_signals + extra)[:5]
        used_vision = True
    else:
        vision_error = vision_result.error
        if vision_error:
            logger.warning("Gemini Vision 실패, 규칙 기반으로만 진행: %s", vision_error)

    # 알려진 사기 도메인 감지 시 확률 1.0으로 확정
    if scam_domains:
        fraud_prob = 1.0

    scores: PredictionScores = score_prediction(fraud_prob, FRAUD_THRESHOLD)

    # 메시지 생성: 템플릿 → polish_with_gemini
    prompt_used: str | None = None
    used_gemini = False
    fallback_to_template = True
    no_change = False
    gemini_error: str | None = None

    template_message = build_template_message(
        company_name=company_name,
        trust_score=scores.trust_score,
        risk_score=scores.risk_score,
        ui_trust_label=scores.ui_trust_label,
        has_signals=bool(risk_signals),
        travel_ban_regions=risk_regions,
        scam_domains=scam_domains,
    )
    polished = template_message
    try:
        gemini_out = await asyncio.to_thread(
            polish_with_gemini,
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

    resp = {
        "analysisId": analysis_id,
        "type": analysis_type,
        "travelBanRegionsMatched": risk_regions,
        "scamDomainsMatched": scam_domains,
        "ocr": {
            "textPreview": ocr.text_preview,
            "textLength": ocr.text_length,
            "languageGuess": ocr.language_guess,
            "confidenceAvg": ocr.confidence_avg,
        },
        "mlPrediction": {
            "modelVersion": MODEL_VERSION,
            "fraudProbability": float(fraud_prob),
            "riskScore": scores.risk_score,
            "riskLevel": scores.model_risk_level,
            "thresholdUsed": float(FRAUD_THRESHOLD),
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
            "usedGeminiVision": used_vision,
            "visionFraudProbability": vision_result.fraud_probability if used_vision else None,
            "visionError": vision_error,
            "usedGemini": used_gemini,
            "fallbackToTemplate": fallback_to_template,
            "noChange": no_change,
            "promptUsed": prompt_used,
            "geminiError": gemini_error,
            "ocrError": ocr.error,
            "riskRegionsMatched": risk_regions,
        }

    return resp


@router.post("/image")
async def analyze_image(request: Request, debug: bool = False) -> dict:
    """
    POST /v1/analyze/image

    - multipart/form-data: multipartFile (필수) + (옵션) meta fields
    - application/json: imageUrl (단일) 또는 imageUrls (복수, 최대 10장, 5장 이상 병렬 처리) + (옵션) meta fields
    """
    content_type = (request.headers.get("content-type") or "").lower()

    image_bytes: bytes | None = None
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
            elif image_url:
                image_urls = [image_url]
            meta_obj = body.get("meta")
            if isinstance(meta_obj, dict):
                meta = meta_obj
            else:
                for k in ["companyName", "countryCode", "region", "channel", "sourceUrl", "type"]:
                    if k in body:
                        meta[k] = body.get(k)

    # 이미지 입력 필수 검증
    if not image_urls and image_bytes is None:
        raise HTTPException(status_code=422, detail="imageUrl, imageUrls, 또는 multipartFile이 필요합니다.")

    # URL 형식 검증
    for url in image_urls:
        if not _URL_SCHEME_RE.match(url):
            raise HTTPException(status_code=400, detail=f"유효하지 않은 이미지 URL: {url}")

    # 이미지 수 제한
    if len(image_urls) > MAX_IMAGE_BATCH:
        raise HTTPException(status_code=400, detail=f"최대 {MAX_IMAGE_BATCH}장까지 지원합니다.")

    # URL 이미지 처리
    if image_urls:
        if len(image_urls) == 1:
            bts = await _download_image_bytes(image_urls[0])
            return await _run_analysis(bts, meta, debug)
        if len(image_urls) >= 5:
            all_bytes = await asyncio.gather(*[_download_image_bytes(u) for u in image_urls])
            results = list(await asyncio.gather(*[_run_analysis(b, meta, debug) for b in all_bytes]))
        else:
            results = []
            for url in image_urls:
                b = await _download_image_bytes(url)
                results.append(await _run_analysis(b, meta, debug))
        return {"results": results}

    # multipart 파일 처리
    return await _run_analysis(image_bytes, meta, debug)

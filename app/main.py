# 백엔드 input 처리해서 결과 반환(추론 서비스)

from __future__ import annotations

"""
FastAPI 엔트리포인트.

이 파일 하나만 보면 서비스의 "입력 → 전처리 → 모델 추론 → 출력" 흐름이 보이도록 구성했습니다.

- /v1/infer: 공고 전체 텍스트 → 사기 확률/점수/레벨 반환 (모델 미로딩 시 503)
- /healthz: 단순 헬스체크
- /v1/external/*: (옵션) Google Maps / Gemini 외부 호출 (기본 OFF, 환경변수로 ON)
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

import logging

from app.diagnostics import analyze_input
from app.external_apis import ExternalAPIError, geocode_address, gemini_generate_text
from app.ml_baseline import ModelLoadError, TfidfLogRegPredictor, clean_posting_text, make_structured_posting_text
from app.schemas import (
    GeminiGenerateRequest,
    GeminiGenerateResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    InputDiagnosticsModel,
    ModelPolicy,
    MapsGeocodeRequest,
    MapsGeocodeResponse,
)
from app.settings import get_settings

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # LOG_LEVEL 환경변수로 로그 레벨을 제어합니다. (기본: INFO)
    level = getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # .env 파일이 있으면 로드합니다. (운영 환경에서는 보통 환경변수를 직접 주입)
    load_dotenv(override=False)
    _configure_logging()
    try:
        # 모델은 앱 시작 시 1회 로딩합니다. (요청마다 로딩하면 느리고 비용 큼)
        app.state.predictor = TfidfLogRegPredictor.from_dir(get_settings().model_dir)
        logger.info("Model loaded")
    except ModelLoadError as e:
        # 더미 모델로 graceful fallback 하지 않고, 명확하게 503을 반환하도록 None으로 둡니다.
        app.state.predictor = None
        logger.error("Model not loaded: %s", e)
    yield


app = FastAPI(
    title="Samak AI - Fraud Baseline Inference",
    version=get_settings().model_version,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    # 백엔드/로드밸런서에서 살아있음 확인 용도
    return HealthResponse()


@app.post("/v1/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    # 필수 입력 검증: 텍스트가 비어있으면 400
    if payload.text.strip() == "":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text must be non-empty")

    # 모델 로딩 실패/미배포 상태면 503
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    settings = get_settings()
    # 학습과 동일한 "문서 템플릿" 형태로 감싸서 추론 입력을 만듭니다.
    # 백엔드가 필드를 따로 주지 않기 때문에, 최소한 DESCRIPTION에 전체 텍스트를 넣고
    # 나머지 필드는 비워둡니다.
    augmented = make_structured_posting_text(payload.text)
    if payload.meta:
        meta_lines: list[str] = []
        if payload.meta.sourceType:
            meta_lines.append(f"[META_SOURCE_TYPE] {payload.meta.sourceType.value}")
        if payload.meta.language:
            meta_lines.append(f"[META_LANGUAGE] {payload.meta.language}")
        if payload.meta.offeredCompensation:
            oc = payload.meta.offeredCompensation
            meta_lines.append(
                f"[META_OFFERED_COMP] amount={oc.amount} currency={oc.currency} period={oc.period.value}"
            )
        if meta_lines:
            augmented = augmented + "\n" + "\n".join(meta_lines)

    # HTML 제거/URL/이메일 치환/소문자/공백 정리/길이 제한 (CPU 추론 속도/안정성 확보)
    cleaned = clean_posting_text(augmented, max_chars=settings.max_text_chars)

    # 민감정보 유출 방지: 본문 전체를 남기지 않고 길이만 로그로 남깁니다.
    logger.info("Infer request analysisId=%s text=<len=%d>", payload.analysisId, len(cleaned))

    # 확률(0~1) 산출
    proba = predictor.predict_proba(cleaned)
    # riskScore는 round(probability*100)
    risk_score = int(round(proba * 100))
    # riskLevel 정책(기본): riskScore 구간으로만 결정 (UI에서 직관적)
    # - HIGH: 80 이상
    # - MEDIUM: 50 이상
    # - LOW: 그 미만
    risk_level = "HIGH" if risk_score >= 80 else ("MEDIUM" if risk_score >= 50 else "LOW")

    # 입력 진단(언어/도메인 적합도). 모델 추론 자체에는 영향 없음.
    diag = analyze_input(payload.text)

    return InferResponse(
        analysisId=payload.analysisId,
        modelVersion=settings.model_version,
        fraudProbability=proba,
        riskScore=risk_score,
        riskLevel=risk_level,
        modelPolicy=ModelPolicy(
            threshold=float(getattr(predictor, "threshold", 0.5)),
            highPrecisionThreshold=getattr(predictor, "high_precision_threshold", None),
        ),
        inputDiagnostics=InputDiagnosticsModel(
            language=diag.language,
            in_domain=diag.in_domain,
            input_confidence=diag.input_confidence,
            note=diag.note,
        ),
        notes=getattr(predictor, "notes", None),
    )


@app.post("/v1/external/maps/geocode", response_model=MapsGeocodeResponse)
def maps_geocode(payload: MapsGeocodeRequest) -> MapsGeocodeResponse:
    settings = get_settings()
    # 외부 API는 기본 OFF. 서비스 요구사항에 따라 백엔드가 호출할 수도 있고,
    # 필요 시 이 AI 서비스에서 직접 호출하도록 옵션으로만 제공합니다.
    if not settings.enable_maps:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Maps integration disabled")
    if not settings.maps_api_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MAPS_API_KEY not configured")

    try:
        result = geocode_address(
            address=payload.address,
            api_key=settings.maps_api_key,
            timeout_s=settings.external_timeout_seconds,
        )
    except ExternalAPIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    return MapsGeocodeResponse(
        status=result.status,
        formattedAddress=result.formatted_address,
        placeId=result.place_id,
        location=result.location,
    )


@app.post("/v1/external/gemini/generate", response_model=GeminiGenerateResponse)
def gemini_generate(payload: GeminiGenerateRequest) -> GeminiGenerateResponse:
    settings = get_settings()
    if not settings.enable_gemini:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gemini integration disabled")
    if not settings.gemini_api_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="GEMINI_API_KEY not configured")

    try:
        result = gemini_generate_text(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            prompt=payload.prompt,
            timeout_s=settings.external_timeout_seconds,
        )
    except ExternalAPIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    return GeminiGenerateResponse(model=result.model, text=result.text)

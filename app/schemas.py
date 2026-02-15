from __future__ import annotations

"""
API 요청/응답 스키마(Pydantic).

백엔드(Spring)와의 계약을 명확히 하기 위해 타입/필드 제약을 여기에서 관리합니다.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    TEXT = "TEXT"
    URL = "URL"
    CHAT = "CHAT"


class CompensationPeriod(str, Enum):
    YEAR = "YEAR"
    MONTH = "MONTH"
    HOUR = "HOUR"


class OfferedCompensation(BaseModel):
    amount: float = Field(..., ge=0)
    currency: str = Field(..., min_length=1, max_length=16)
    period: CompensationPeriod


class InferMeta(BaseModel):
    sourceType: SourceType | None = None
    language: str | None = None
    offeredCompensation: OfferedCompensation | None = None


class InferRequest(BaseModel):
    analysisId: str = Field(..., min_length=1)
    # 공고 전체 텍스트(제목+본문+요구사항 등 합친 문자열)
    text: str = Field(..., description="Full job posting text (title + description + requirements + etc.)")
    # meta는 "텍스트에서 파싱 가능한 정보만" 전달한다는 전제 (MVP에서는 추론에 직접 사용하지 않음)
    meta: InferMeta | None = None


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class InputDiagnosticsModel(BaseModel):
    # 입력 텍스트 적합도/도메인 경고용 진단 정보
    language: str = Field(..., description="ko/en/zh/unknown 등 (휴리스틱)")
    in_domain: bool = Field(..., description="모델 학습 도메인(현재: 영어 공고 중심) 내 입력인지 여부")
    input_confidence: float = Field(..., ge=0.0, le=1.0, description="입력 적합도(휴리스틱)")
    note: str | None = Field(None, description="UI에 표시할 경고/주의 문구")


class ModelPolicy(BaseModel):
    # 배포 폴더(models/fraud-baseline/metadata.json)의 정책 값을 그대로 노출합니다.
    threshold: float = Field(..., ge=0.0, le=1.0, description="F1-max 등 기본 정책 threshold")
    highPrecisionThreshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="(옵션) precision 정책 threshold (예: precision>=0.90)",
    )


class InferResponse(BaseModel):
    analysisId: str
    modelVersion: str
    fraudProbability: float = Field(..., ge=0.0, le=1.0)
    riskScore: int = Field(..., ge=0, le=100)
    riskLevel: RiskLevel
    modelPolicy: ModelPolicy
    inputDiagnostics: InputDiagnosticsModel | None = None
    notes: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class MapsGeocodeRequest(BaseModel):
    # 지도 API로 지오코딩할 주소 문자열
    address: str = Field(..., min_length=1)


class MapsGeocodeResponse(BaseModel):
    status: str
    formattedAddress: str | None = None
    placeId: str | None = None
    location: dict[str, float] | None = None


class GeminiGenerateRequest(BaseModel):
    # Gemini에 전달할 프롬프트
    prompt: str = Field(..., min_length=1)


class GeminiGenerateResponse(BaseModel):
    model: str
    text: str

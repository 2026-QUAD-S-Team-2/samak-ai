from __future__ import annotations

"""
Swagger(OpenAPI) 문서용 응답 스키마.

백엔드가 바로 쓰기 쉬운 "플랫(flat) JSON" 형태로 계약을 고정합니다.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# Swagger에서 enum으로 노출되도록 Literal로 고정합니다.
# - CRITICAL: (선택) 향후 정책 확장용
# - UNKNOWN: OCR 실패 등으로 판단 불가 케이스
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]


class AnalyzeImageRequest(BaseModel):
    imageUrl: str = Field(
        ...,
        description="분석할 이미지 URL (http/https). 파일 업로드는 지원하지 않습니다.",
        examples=["https://example.com/sample.png"],
    )
    debug: bool = Field(
        default=False,
        description="true면 서버 로그에 디버그 정보를 남깁니다(응답 스키마는 변경되지 않음).",
        examples=[False],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "imageUrl": "https://example.com/sample.png",
                    "debug": False,
                }
            ]
        }
    )


class AnalyzeImageResponse(BaseModel):
    analysisId: UUID = Field(
        ...,
        description="분석 요청의 고유 ID (UUID)",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    fraudProbability: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="사기 확률(0~1). OCR 실패 등으로 추론을 생략한 경우 null.",
        examples=[0.8735],
        json_schema_extra={"type": "number", "format": "float"},
    )
    riskScore: int | None = Field(None, ge=0, le=100)
    riskLevel: RiskLevel
    riskSignals: list[str] = Field(default_factory=list, max_length=3)
    travelBanRegionsMatched: list[str] = Field(default_factory=list)
    message: str

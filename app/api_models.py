from __future__ import annotations

"""
Swagger(OpenAPI) 문서용 응답 스키마.

백엔드가 바로 쓰기 쉬운 "플랫(flat) JSON" 형태로 계약을 고정합니다.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    countryCode: str = Field(
        ...,
        description='요청 국가 코드(ISO 3166-1 alpha-2). 예: "KR", "UA"',
        examples=["UA"],
    )
    salary: str | None = Field(
        default=None,
        description='(옵션) 급여 문자열. 예: "3000000 KRW", "$300/day". 값은 그대로 사용됩니다.',
        examples=["3000000 KRW"],
    )
    debug: bool = Field(
        default=False,
        description="true면 서버 로그에 디버그 정보를 남깁니다(응답 스키마는 변경되지 않음).",
        examples=[False],
    )

    @field_validator("countryCode", mode="before")
    @classmethod
    def _normalize_country_code(cls, v: Any) -> str:
        t = str(v or "").strip().upper()
        if len(t) != 2 or not t.isalpha():
            raise ValueError("countryCode must be ISO 3166-1 alpha-2 (2 letters), e.g. 'KR'")
        return t

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "debug": False,
                    "imageUrl": "https://example.com/sample.png",
                    "countryCode": "UA",
                    "salary": "3000000 KRW",
                },
                {
                    "debug": False,
                    "imageUrl": "https://example.com/sample.png",
                    "countryCode": "KR",
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

from __future__ import annotations

"""
Swagger(OpenAPI) 문서용 응답 스키마.

백엔드가 바로 쓰기 쉬운 "플랫(flat) JSON" 형태로 계약을 고정합니다.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


# Swagger에서 enum으로 노출되도록 Literal로 고정합니다.
# - CRITICAL: (선택) 향후 정책 확장용
# - UNKNOWN: Gemini 분석 실패 등으로 판단 불가 케이스
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
WageMessageType = Literal["NONE", "INFO", "WARNING", "ERROR"]


class AnalyzeImageRequest(BaseModel):
    imageUrls: list[str] = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("imageUrls", "imageUrl"),
        description="분석할 이미지 URL 목록 (http/https). 파일 업로드는 지원하지 않습니다.",
        examples=[["https://example.com/sample.png"]],
    )
    countryCode: str = Field(
        ...,
        description='요청 국가 코드(ISO 3166-1 alpha-2). 예: "KR", "UA"',
        examples=["UA"],
    )
    salaryText: str | None = Field(
        default=None,
        validation_alias=AliasChoices("salaryText", "salary"),
        description='(옵션) 급여 문자열. 예: "3000000 KRW", "$300/day". 값은 그대로 사용됩니다.',
        examples=["3000000 KRW"],
    )
    debug: bool = Field(
        default=False,
        description="true면 서버 로그에 디버그 정보를 남깁니다(응답 스키마는 변경되지 않음).",
        examples=[False],
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_image_urls(cls, data: Any) -> Any:
        # Backward-compat:
        # - 신규: {"imageUrls": ["https://...", "..."]}
        # - 기존: {"imageUrl": "https://..."}
        if not isinstance(data, dict):
            return data

        if "imageUrls" in data and isinstance(data["imageUrls"], str):
            data["imageUrls"] = [data["imageUrls"]]

        if "imageUrl" in data and "imageUrls" not in data:
            v = data.get("imageUrl")
            if isinstance(v, str):
                data["imageUrls"] = [v]
            elif isinstance(v, list):
                data["imageUrls"] = v

        return data

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
                    "imageUrls": ["https://example.com/sample.png"],
                    "countryCode": "UA",
                    "salaryText": "3000000 KRW",
                },
                {
                    "debug": False,
                    "imageUrls": ["https://example.com/sample.png"],
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
        description="사기 확률(0~1). Gemini 분석 실패 등으로 추론을 생략한 경우 null.",
        examples=[0.8735],
        json_schema_extra={"type": "number", "format": "float"},
    )
    riskScore: int | None = Field(None, ge=0, le=100)
    riskLevel: RiskLevel
    riskSignals: list[str] = Field(default_factory=list, max_length=3)
    travelBanRegionsMatched: list[str] = Field(default_factory=list)
    wageMessageType: WageMessageType = Field(
        default="NONE",
        description="임금 관련 메시지 유형. NONE이면 wageMessage는 null.",
        examples=["WARNING"],
    )
    wageMessage: str | None = Field(
        default=None,
        description="임금 관련 메시지(경고/안내/오류). 해당 없으면 null.",
        examples=["제안된 시급(5000)이(가) KR의 법정 최저임금(10320)보다 낮습니다. 공고의 급여/근로조건을 다시 확인해 주세요."],
    )
    message: str


class AnalyzeImagesResponse(BaseModel):
    results: list[AnalyzeImageResponse] = Field(
        default_factory=list,
        description="이미지별 분석 결과 목록(요청 순서 보장).",
    )


class WageWarningRequest(BaseModel):
    countryCode: str = Field(
        ...,
        description='요청 국가 코드(ISO 3166-1 alpha-2). 예: "KR", "UA"',
        examples=["UA"],
    )
    salaryText: str | None = Field(
        default=None,
        description='(옵션) 시급(hourly) 기준 급여 텍스트(통화 포함). 예: "KRW 12000/h", "₩12,000/hour", "USD 25/h". 미제공 시 최저시급 안내 또는 null.',
        examples=["UAH 150/h"],
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
                {"countryCode": "UA", "salaryText": "UAH 150/h"},
                {"countryCode": "KR", "salaryText": "KRW 12000/hour"},
                {"countryCode": "UA"},
            ]
        }
    )


class WageWarningData(BaseModel):
    warningMessage: str | None = Field(
        default=None,
        description="임금 경고 메시지. 경고 조건에 해당하지 않으면 null.",
    )


class WageWarningResponse(BaseModel):
    code: str
    message: str
    data: WageWarningData

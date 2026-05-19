from __future__ import annotations

"""
Swagger(OpenAPI) 문서용 응답 스키마.

백엔드가 바로 쓰기 쉬운 "플랫(flat) JSON" 형태로 계약을 고정합니다.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]


class AnalyzeImageRequest(BaseModel):
    imageUrls: list[str] = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("imageUrls", "imageUrl"),
        description="분석할 이미지 URL 목록 (http/https). 파일 업로드는 지원하지 않습니다.",
        examples=[["https://example.com/sample.png"]],
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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "debug": False,
                    "imageUrls": ["https://example.com/sample.png"],
                }
            ]
        }
    )


class LocationLatLng(BaseModel):
    lat: float
    lng: float


class LocationData(BaseModel):
    rawText: str = Field(description="검색에 사용된 회사명 또는 지역명")
    lat: float
    lng: float
    adminLevel: str | None = None
    zoom: int = Field(description="Flutter 지도 초기 줌 레벨. 회사 핀=14, 지역 bounds=8~12")
    status: str = Field(description='"company" | "region"')
    viewportNe: LocationLatLng | None = None
    viewportSw: LocationLatLng | None = None


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
    message: str
    location: LocationData | None = Field(
        default=None,
        description="지도 표시용 위치 정보. GOOGLE_MAPS_API_KEY 미설정 또는 조회 실패 시 null.",
    )


class AnalyzeImagesResponse(BaseModel):
    results: list[AnalyzeImageResponse] = Field(
        default_factory=list,
        description="이미지별 분석 결과 목록(요청 순서 보장).",
    )



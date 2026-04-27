from __future__ import annotations

from pydantic import BaseModel


class AnalysisRequestMessage(BaseModel):
    analysisItemId: int
    imageUrls: list[str]
    countryCode: str
    salaryText: str | None = None
    debug: bool = False


class AnalysisResultMessage(BaseModel):
    analysisId: str
    fraudProbability: float | None
    riskScore: int | None
    riskLevel: str | None
    riskSignals: list[str]
    travelBanRegionsMatched: list[str]
    message: str | None

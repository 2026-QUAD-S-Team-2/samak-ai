from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class MinWageRecord(BaseModel):
    currency: str = Field(..., description="ISO 4217 currency code (e.g. KRW, UAH)")
    hourly: float = Field(..., ge=0, description="Hourly minimum wage (numeric)")
    asOf: str = Field(..., description="Effective date (YYYY-MM-DD)")
    source: str = Field(..., description="Data source identifier")

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Any) -> str:
        t = str(v or "").strip().upper()
        if len(t) != 3 or not t.isalpha():
            raise ValueError("currency must be ISO 4217 (3 letters)")
        return t


MinWageDataset = dict[str, MinWageRecord]


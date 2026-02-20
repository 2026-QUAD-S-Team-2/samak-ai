from __future__ import annotations

"""
임금 경고(wage warning) 단독 엔드포인트.

- salaryText가 없으면 경고 생성 불가 안내만 반환(점수 변화 없음)
- salaryText는 시급(hourly)만 지원하며, 통화 포함 텍스트여야 함
- 외부 데이터(ILOSTAT) 조회 실패 시에도 HTTP 200 + 안내 메시지로 응답
"""

import logging

from fastapi import APIRouter

from app.api_models import WageWarningRequest, WageWarningResponse
from app.services.wage_service import evaluate_wage_warning

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["wage"])


@router.post("/wage-warning", response_model=WageWarningResponse)
async def wage_warning(payload: WageWarningRequest) -> dict:
    try:
        out = await evaluate_wage_warning(country_code=payload.countryCode, salary_text=payload.salaryText)
        msg = out.warning_message
    except Exception as e:  # noqa: BLE001
        logger.exception("wage-warning failed: %s", e)
        msg = "임금 기준 데이터 조회에 실패하여 경고를 생성할 수 없습니다."

    return {"code": "200", "message": "API 요청 성공", "data": {"warningMessage": msg}}

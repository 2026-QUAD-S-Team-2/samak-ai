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
    """
    POST /v1/wage-warning

    필드:
    - countryCode (필수): ISO 3166-1 alpha-2 (예: KR, US)
    - salaryText (옵션): 시급(hourly) 텍스트(통화 포함). 예: "KRW 12000/h", "USD 25/hour"

    동작:
    - salaryText 미제공:
      - 내부 최저시급 데이터가 있으면: 최저시급 안내 문구 반환
      - 없으면: warningMessage=null
    - salaryText 제공:
      - 파싱 실패(시급 아님/통화/숫자 해석 불가): 형식 안내 문구 반환
      - 통화가 해당 국가 최저시급 통화와 다름: 비교 보류 안내 문구 반환 (환율 변환은 하지 않음)
      - 시급 < 최저시급: 최저임금 미달 경고 문구 반환
      - 시급 >= (최저시급 * 4): 고임금 경고 문구 반환
      - 그 외: warningMessage=null

    참고:
    - 이 엔드포인트는 경고 메시지만 반환합니다(분석 점수는 /v1/analyze/image에서만 조정).
    """
    try:
        out = await evaluate_wage_warning(country_code=payload.countryCode, salary_text=payload.salaryText)
        msg = out.warning_message
    except Exception as e:  # noqa: BLE001
        logger.exception("wage-warning failed: %s", e)
        msg = "임금 기준 데이터 조회에 실패하여 경고를 생성할 수 없습니다."

    return {"code": "200", "message": "API 요청 성공", "data": {"warningMessage": msg}}

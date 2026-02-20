from __future__ import annotations

"""
임금 경고(wage warning) 서비스 레이어.

- salaryText 파싱(시급 only)
- 통화 정규화(ISO 4217)
- 국가 기본 통화 불일치 시 비교 보류
- ILOSTAT(또는 호환 데이터 소스) 조회
- 경고 메시지 생성(템플릿에 숫자만 삽입)
- 점수 조정 + 상한(cap) 적용
"""

from dataclasses import dataclass
import logging
import os
import re

from app.integrations import ilostat_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedSalary:
    amount_hourly: float
    currency_code: str


@dataclass(frozen=True)
class WageScores:
    risk_score: int | None
    trust_score: int | None
    fraud_probability: float | None


@dataclass(frozen=True)
class WageWarningResult:
    warning_message: str
    scores: WageScores


@dataclass(frozen=True)
class WageWarningDecision:
    warning_kind: str
    warning_message: str
    parsed_salary: ParsedSalary | None
    min_wage: float | None
    avg_wage: float | None
    currency_mismatch: bool


_HOURLY_RE = re.compile(r"(/h|/hr|/hour|per\s*hour|hourly|hour|hr)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(\d[\d,]*\.?\d*)")

# 통화 토큰 탐지(기호/코드/단어)
_CURRENCY_TOKENS = [
    "KRW",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CNY",
    "RMB",
    "CN¥",
    "UAH",
    "won",
    "dollar",
    "dollars",
    "yen",
    "euro",
    "pound",
    "₩",
    "$",
    "€",
    "£",
    "¥",
    "元",
]

_DEFAULT_CURRENCY_BY_COUNTRY: dict[str, str] = {
    "KR": "KRW",
    "UA": "UAH",
    "US": "USD",
    "GB": "GBP",
    "JP": "JPY",
    "CN": "CNY",
    "DE": "EUR",
    "FR": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "NL": "EUR",
    "BE": "EUR",
    "PT": "EUR",
    "IE": "EUR",
    "CA": "CAD",
    "AU": "AUD",
    "SG": "SGD",
    "NZ": "NZD",
    "HK": "HKD",
}

_DOLLAR_DISAMBIGUATION: dict[str, str] = {
    "CA": "CAD",
    "AU": "AUD",
    "SG": "SGD",
    "NZ": "NZD",
    "HK": "HKD",
}


def cap_scores(scores: WageScores) -> WageScores:
    rs = scores.risk_score
    ts = scores.trust_score
    fp = scores.fraud_probability

    if rs is not None:
        rs = int(min(99, max(0, round(float(rs)))))
    if ts is not None:
        ts = int(min(99, max(0, round(float(ts)))))
    if fp is not None:
        fp = float(min(0.99, max(0.0, float(fp))))
    return WageScores(risk_score=rs, trust_score=ts, fraud_probability=fp)


def get_country_default_currency(country_code: str) -> str | None:
    code = (country_code or "").strip().upper()
    return _DEFAULT_CURRENCY_BY_COUNTRY.get(code)


def normalize_currency(country_code: str, raw_currency_token: str) -> str | None:
    tok = (raw_currency_token or "").strip()
    if tok == "":
        return None

    upper = tok.upper()
    if re.fullmatch(r"[A-Z]{3}", upper):
        return upper

    cc = (country_code or "").strip().upper()

    if tok in {"₩"} or upper == "KRW" or upper == "WON" or "won" in tok.lower():
        return "KRW"
    if tok in {"€"} or upper == "EUR" or "euro" in tok.lower():
        return "EUR"
    if tok in {"£"} or upper == "GBP" or "pound" in tok.lower():
        return "GBP"

    if tok in {"¥"} or upper in {"JPY", "YEN"} or "yen" in tok.lower():
        if cc == "CN":
            return "CNY"
        return "JPY"
    if upper in {"CNY", "RMB", "CN¥"} or tok == "元":
        return "CNY"

    if tok == "$" or upper in {"USD", "US$"} or "dollar" in tok.lower():
        return _DOLLAR_DISAMBIGUATION.get(cc, "USD")

    if upper == "UAH":
        return "UAH"

    return None


def parse_salary_text(*, country_code: str, salary_text: str) -> ParsedSalary | None:
    s = (salary_text or "").strip()
    if s == "":
        return None

    if _HOURLY_RE.search(s) is None:
        return None

    # currency token: first match from predefined list (longer token first)
    token = None
    for t in sorted(_CURRENCY_TOKENS, key=len, reverse=True):
        if t.isalpha():
            if re.search(rf"\b{re.escape(t)}\b", s, flags=re.IGNORECASE):
                token = t
                break
        else:
            if t in s:
                token = t
                break
    if token is None:
        return None

    currency = normalize_currency(country_code, token)
    if currency is None:
        return None

    m = _AMOUNT_RE.search(s)
    if not m:
        return None
    num_raw = m.group(1)
    try:
        amount = float(num_raw.replace(",", ""))
    except Exception:
        return None
    if amount <= 0:
        return None

    return ParsedSalary(amount_hourly=amount, currency_code=currency)


async def get_min_wage(country_code: str) -> float | None:
    ind = (os.environ.get("ILOSTAT_MIN_WAGE_INDICATOR") or "").strip()
    if not ind:
        return None
    return await ilostat_client.fetch_series(country_code=country_code, indicator_code=ind)


async def get_avg_wage(country_code: str) -> float | None:
    ind_median = (os.environ.get("ILOSTAT_MEDIAN_WAGE_INDICATOR") or "").strip()
    ind_avg = (os.environ.get("ILOSTAT_AVG_WAGE_INDICATOR") or "").strip()
    ind = ind_median or ind_avg
    if not ind:
        return None
    return await ilostat_client.fetch_series(country_code=country_code, indicator_code=ind)


def _fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(round(float(x), 2))


def build_warning_message(
    *,
    country_code: str,
    salary_text: str | None,
    parsed_salary: ParsedSalary | None,
    min_wage: float | None,
    avg_wage: float | None,
    currency_mismatch: bool,
    warning_kind: str,
) -> str:
    cc = (country_code or "").strip().upper()

    if not salary_text:
        return "제안 임금 정보가 제공되지 않아 최저임금/평균임금 기준 경고를 생성할 수 없습니다."

    if warning_kind == "parse_error":
        return (
            "제안 임금 형식을 해석할 수 없습니다. 시급(hourly) 기준으로 통화와 함께 입력해 주세요. "
            "예: KRW 12000/h, USD 25/hour"
        )

    if parsed_salary is None:
        return (
            "제안 임금 형식을 해석할 수 없습니다. 시급(hourly) 기준으로 통화와 함께 입력해 주세요. "
            "예: KRW 12000/h, USD 25/hour"
        )

    if currency_mismatch:
        return (
            f"제안된 임금의 통화({parsed_salary.currency_code})가 {cc}의 기본 통화와 달라 임금 기준 비교를 보류했습니다. "
            "해당 국가 통화 기준 시급으로 다시 입력해 주세요."
        )

    if warning_kind == "min_wage_low" and min_wage is not None:
        return (
            f"제안된 시급({_fmt_num(parsed_salary.amount_hourly)})이(가) {cc}의 법정 최저임금({_fmt_num(min_wage)})보다 낮습니다. "
            "공고의 급여/근로조건을 다시 확인해 주세요."
        )

    if warning_kind == "high_salary" and avg_wage is not None:
        ratio = parsed_salary.amount_hourly / avg_wage if avg_wage > 0 else 0.0
        ratio_s = _fmt_num(ratio)
        return (
            f"제안된 시급({_fmt_num(parsed_salary.amount_hourly)})이(가) {cc}의 평균(또는 중위) 시급({_fmt_num(avg_wage)}) 대비 약 {ratio_s}배로 높습니다. "
            "비현실적 보상 제안은 사기 공고에서 흔히 나타나므로 추가 검증을 권장합니다."
        )

    if min_wage is None and avg_wage is None:
        return "해당 국가의 임금 기준 데이터가 충분하지 않아 경고를 생성할 수 없습니다."

    return "제공된 정보로는 임금 경고 조건에 해당하지 않습니다."


def apply_wage_adjustments(
    *,
    scores: WageScores,
    warning_kind: str,
) -> WageScores:
    rs = scores.risk_score
    ts = scores.trust_score
    fp = scores.fraud_probability

    if warning_kind == "min_wage_low":
        if rs is not None:
            rs = int(round(float(rs) * 1.10))
        if ts is not None:
            ts = int(round(float(ts) * 0.90))
        if fp is not None:
            fp = float(fp) * 1.05
    elif warning_kind == "high_salary":
        if rs is not None:
            rs = int(round(float(rs) * 1.15))
        if ts is not None:
            ts = int(round(float(ts) * 0.90))
        if fp is not None:
            fp = float(fp) * 1.10

    return WageScores(risk_score=rs, trust_score=ts, fraud_probability=fp)


async def decide_wage_warning(*, country_code: str, salary_text: str | None) -> WageWarningDecision:
    cc = (country_code or "").strip().upper()
    st = (salary_text or "").strip() if salary_text is not None else None

    if not st:
        msg = build_warning_message(
            country_code=cc,
            salary_text=None,
            parsed_salary=None,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=False,
            warning_kind="missing",
        )
        return WageWarningDecision(
            warning_kind="missing",
            warning_message=msg,
            parsed_salary=None,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=False,
        )

    parsed = parse_salary_text(country_code=cc, salary_text=st)
    if parsed is None:
        msg = build_warning_message(
            country_code=cc,
            salary_text=st,
            parsed_salary=None,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=False,
            warning_kind="parse_error",
        )
        return WageWarningDecision(
            warning_kind="parse_error",
            warning_message=msg,
            parsed_salary=None,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=False,
        )

    default_currency = get_country_default_currency(cc)
    if default_currency and parsed.currency_code != default_currency:
        msg = build_warning_message(
            country_code=cc,
            salary_text=st,
            parsed_salary=parsed,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=True,
            warning_kind="mismatch",
        )
        return WageWarningDecision(
            warning_kind="mismatch",
            warning_message=msg,
            parsed_salary=parsed,
            min_wage=None,
            avg_wage=None,
            currency_mismatch=True,
        )

    min_wage = await get_min_wage(cc)
    avg_wage = await get_avg_wage(cc)

    warning_kind = "none"
    if min_wage is not None and parsed.amount_hourly < min_wage:
        warning_kind = "min_wage_low"
    elif avg_wage is not None and avg_wage > 0 and (parsed.amount_hourly / avg_wage) >= 2.5:
        warning_kind = "high_salary"

    msg = build_warning_message(
        country_code=cc,
        salary_text=st,
        parsed_salary=parsed,
        min_wage=min_wage,
        avg_wage=avg_wage,
        currency_mismatch=False,
        warning_kind=warning_kind,
    )

    return WageWarningDecision(
        warning_kind=warning_kind,
        warning_message=msg,
        parsed_salary=parsed,
        min_wage=min_wage,
        avg_wage=avg_wage,
        currency_mismatch=False,
    )


async def evaluate_wage_warning(
    *,
    country_code: str,
    salary_text: str | None,
    existing_scores: WageScores | None = None,
) -> WageWarningResult:
    """
    경고 메시지 생성 + (선택) 점수 조정까지 수행합니다.

    existing_scores가 None이면 점수 조정 없이 message만 반환합니다.
    """
    base_scores = existing_scores or WageScores(risk_score=None, trust_score=None, fraud_probability=None)
    decision = await decide_wage_warning(country_code=country_code, salary_text=salary_text)
    adjusted = apply_wage_adjustments(scores=base_scores, warning_kind=decision.warning_kind)
    return WageWarningResult(warning_message=decision.warning_message, scores=cap_scores(adjusted))

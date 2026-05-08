from __future__ import annotations

"""
템플릿 기반 요약 문장 생성 + (옵션) Gemini 출력 검증 로직.
"""

import re


def build_template_message(
    *,
    company_name: str | None,
    trust_score: int,
    risk_score: int,
    ui_trust_label: str,
    risk_signals: list[str] | None = None,
    # backward compatibility: older callers only tell whether signals exist
    has_signals: bool | None = None,
    travel_ban_regions: list[str] | None = None,
    scam_domains: list[str] | None = None,
) -> str:
    # 요구사항: 자연어는 앞으로 무조건 한국어로 고정 (언어 감지와 무관)
    # NOTE: 회사명은 UI 레이아웃/정책이 정리되기 전까지 메시지에 넣지 않습니다.
    _ = company_name
    _ = risk_score

    s1 = f"AI 분석 결과, 해당 공고는 {ui_trust_label} 단계로 분류되었습니다."
    s2 = f"신뢰도는 약 {trust_score}%로 분석되었습니다."
    parts: list[str] = [s1, s2]

    domains = [d.strip() for d in (scam_domains or []) if d and d.strip()]
    if domains:
        shown = ", ".join(domains[:3])
        parts.append(f"경고: 해당 공고에서 알려진 사기 도메인({shown})이 발견되었습니다. 이 도메인은 구직 사기에 활용되는 가짜 도메인으로 알려져 있으니 절대 응하지 마세요.")

    signals = [s.strip() for s in (risk_signals or []) if s and s.strip()]
    if not signals and has_signals:
        # 신호의 구체 문자열을 알 수 없을 때(구버전 호출자)
        parts.append("특히 사기 공고에서 자주 사용되는 표현이 포함되어 있습니다.")
    elif signals:
        shown = ", ".join([f"‘{s}’" for s in signals[:3]])
        parts.append(f"특히 {shown} 등 사기 공고에서 자주 사용되는 표현이 포함되어 있습니다.")
    else:
        parts.append("뚜렷한 사기 공고 패턴 표현은 탐지되지 않았습니다.")

    # 위험 신호(사기 패턴) 문장 뒤에 여행금지 문장을 붙이고, 마지막에는 안내 문장을 고정
    regions = [r.strip() for r in (travel_ban_regions or []) if r and r.strip()]
    if regions:
        shown = ", ".join(regions[:5])
        parts.append(
            f"또한 공고 텍스트에서 대한민국 외교부가 여행금지 지역으로 지정한 국가/지역({shown})이(가) 언급되었습니다. "
            "해당 지역과 관련된 활동은 법적·안전상의 위험이 있을 수 있으므로, 관련 정보를 충분히 확인하시기 바랍니다."
        )
    return " ".join(parts)


def validate_polished_message(template: str, candidate: str) -> bool:
    """
    Gemini 결과가 형식을 깨거나 숫자를 바꾸면 fallback 해야 합니다.

    검증 규칙(안정성 우선, 문장 수 고정 X):
    1) 한국어인지(한글 포함)
    2) 숫자(점수) 그대로인지
    3) 새로운 사실 추가 금지(키워드 기반)
    4) 길이 제한(너무 길거나 짧으면 탈락)
    """
    t = (template or "").strip()
    c = (candidate or "").strip()
    if not c:
        return False

    # 4) 길이 제한
    # 너무 짧으면 정보가 빠졌을 가능성이 크고, 너무 길면 Gemini가 과도한 내용을 추가했을 가능성이 큼
    if not (60 <= len(c) <= 600):
        return False

    # 1) 한국어인지(아주 가벼운 체크): 한글이 거의 없으면 실패로 간주
    hangul = sum(1 for ch in c if 0xAC00 <= ord(ch) <= 0xD7A3)
    if hangul < 10:
        return False

    # 한글 비율(공백 제외)이 너무 낮으면 영어/잡문일 가능성
    compact = "".join(ch for ch in c if not ch.isspace())
    if compact and (hangul / len(compact) < 0.2):
        return False

    # 2) 숫자 보존: 템플릿에 들어간 점수(0~100) 및 % 형태가 그대로 포함되어야 함
    # 템플릿에 있는 "58%" 같은 토큰을 그대로 유지하도록 강제
    score_tokens = re.findall(r"\b\d{1,3}%\b", t)
    for tok in score_tokens:
        # "58 %"처럼 공백이 끼는 경우를 허용
        digits = tok.replace("%", "")
        if not re.search(rf"\b{re.escape(digits)}\s*%\b", c):
            return False

    # (보조) 템플릿의 0~100 숫자(정수)가 candidate에서 사라지면 실패
    nums = re.findall(r"\b\d{1,3}\b", t)
    must_keep_ints = [n for n in nums if 0 <= int(n) <= 100]
    for n in must_keep_ints:
        if re.search(rf"\b{re.escape(n)}\b", c) is None:
            return False

    # 3) 새로운 사실 추가 금지(키워드 기반)
    # 템플릿에는 없는 "외부 근거/사실" 키워드가 들어오면 실패 처리
    disallowed = [
        "지도",
        "맵",
        "구글",
        "google",
        "maps",
        "리뷰",
        "평점",
        "직원",
        "직원수",
        "매출",
        "등기",
        "법인",
        "주소",
        "위치 확인",
        "신고",
        "수사",
        "경찰",
        "wage",
        "wageindicator",
        "확정",
        "100%",
    ]
    t_lower = t.lower()
    c_lower = c.lower()
    for kw in disallowed:
        if kw.lower() in c_lower and kw.lower() not in t_lower:
            return False

    return True

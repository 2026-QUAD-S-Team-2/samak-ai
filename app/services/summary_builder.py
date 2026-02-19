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
    has_signals: bool = False,
    travel_ban_regions: list[str] | None = None,
) -> str:
    # 요구사항: 자연어는 앞으로 무조건 한국어로 고정 (언어 감지와 무관)
    if company_name:
        s1 = f"AI 신뢰도는 {ui_trust_label} 입니다. '{company_name}' 공고는 {trust_score}% 신뢰할 수 있어요."
    else:
        s1 = f"AI 신뢰도는 {ui_trust_label} 입니다. 해당 공고는 {trust_score}% 신뢰할 수 있어요."
    s2 = f"텍스트 패턴 분석 결과, 사기 가능성은 {risk_score}% 수준으로 추정됩니다."
    parts: list[str] = [s1, s2]
    if has_signals:
        parts.append("또한, 텍스트에서 사기 패턴으로 해석될 수 있는 표현이 일부 탐지되었습니다.")
    else:
        parts.append("뚜렷한 사기 패턴 표현은 탐지되지 않았습니다.")

    # 위험 신호(사기 패턴) 문장 뒤에 여행금지 문장을 붙이고, 마지막에는 안내 문장을 고정
    regions = [r.strip() for r in (travel_ban_regions or []) if r and r.strip()]
    if regions:
        shown = ", ".join(regions[:5])
        parts.append(f"또한 공고 텍스트에서 대한민국 외교부가 여행금지 지역으로 지정한 국가/지역({shown})이(가) 언급되었습니다.")
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
    score_tokens = re.findall(r"\\b\\d{1,3}%\\b", t)
    for tok in score_tokens:
        # "58 %"처럼 공백이 끼는 경우를 허용
        digits = tok.replace("%", "")
        if not re.search(rf"\\b{re.escape(digits)}\\s*%\\b", c):
            return False

    # (보조) 템플릿의 0~100 숫자(정수)가 candidate에서 사라지면 실패
    nums = re.findall(r"\\b\\d{1,3}\\b", t)
    must_keep_ints = [n for n in nums if 0 <= int(n) <= 100]
    for n in must_keep_ints:
        if re.search(rf"\\b{re.escape(n)}\\b", c) is None:
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

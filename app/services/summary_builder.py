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
) -> str:
    # 요구사항: 자연어는 앞으로 무조건 한국어로 고정 (언어 감지와 무관)
    if company_name:
        s1 = f"AI 신뢰도는 {ui_trust_label} 입니다. '{company_name}' 공고는 {trust_score}% 신뢰할 수 있어요."
    else:
        s1 = f"AI 신뢰도는 {ui_trust_label} 입니다. 해당 공고는 {trust_score}% 신뢰할 수 있어요."
    s2 = f"텍스트 패턴 분석 결과, 사기 가능성은 {risk_score}% 수준으로 추정됩니다."
    # 4문장 구조를 맞추기 위해 기존 3번째 문장을 의미 그대로 2문장으로 분리
    s3 = "다만 AI는 텍스트 기반 판단입니다."
    s4 = "지원 전 공식 채용 페이지/연락처/요구 사항을 추가로 확인해 주세요."
    return f"{s1} {s2} {s3} {s4}"


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

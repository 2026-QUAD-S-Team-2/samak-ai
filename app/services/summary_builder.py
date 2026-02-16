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
    s3 = "다만 AI는 텍스트 기반 판단이므로, 지원 전 공식 채용 페이지/연락처/요구 사항을 추가로 확인해 주세요."
    return f"{s1} {s2} {s3}"


def validate_polished_message(template: str, candidate: str) -> bool:
    """
    Gemini 결과가 형식을 깨거나 숫자를 바꾸면 fallback 해야 합니다.

    최소 검증:
    - 문장 수(마침표 기준) 3개 유지
    - template의 퍼센트 숫자(예: 12.3%)와 trustScore(정수)가 candidate에 그대로 포함
    """
    t = (template or "").strip()
    c = (candidate or "").strip()
    if not c:
        return False

    # 문장 수: '.' 기준으로 정확히 3문장 유지
    sentences = [s.strip() for s in c.split(".") if s.strip()]
    if len(sentences) != 3:
        return False

    # 한국어 유지(아주 가벼운 체크): 한글이 거의 없으면 실패로 간주
    hangul = sum(1 for ch in c if 0xAC00 <= ord(ch) <= 0xD7A3)
    if hangul < 10:
        return False

    # 숫자 보존: 템플릿에 들어간 숫자(0~100)들이 그대로 포함되어야 함
    nums = re.findall(r"\\b\\d{1,3}\\b", t)
    must_keep = [n for n in nums if 0 <= int(n) <= 100]
    for n in must_keep:
        if n not in c:
            return False

    return True

from __future__ import annotations

"""
.env 로딩 유틸 (로컬 개발 편의).

운영에서는 보통 환경변수를 직접 주입하므로, override=False로 "없으면 로드"만 수행합니다.
"""

from dotenv import load_dotenv

_LOADED = False


def load_dotenv_once() -> None:
    global _LOADED
    if _LOADED:
        return
    load_dotenv(override=False)
    _LOADED = True


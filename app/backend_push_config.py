from __future__ import annotations

"""
백엔드 Push 설정(하드코딩).

주의:
- 운영/외부 공유 환경에서 "API 키"를 하드코딩하는 것은 매우 위험합니다.
- 현재는 팀 내부 MVP 용도로만, URL/옵션을 코드로 고정하고 싶을 때 사용합니다.
- 가능하면 API 키는 비워두고(인증 없이 내부망), 추후 Secret/환경변수로 옮기세요.
"""

# 분석 결과를 백엔드로 전송할지 여부
BACKEND_PUSH_ENABLED: bool = False

# 백엔드 수신 엔드포인트 URL (예: https://backend.example.com/api/ai/analysis)
BACKEND_PUSH_URL: str = ""

# (옵션) Bearer 토큰. 되도록 하드코딩하지 말 것.
BACKEND_PUSH_API_KEY: str = ""

# requests timeout / retries (MVP 기본)
BACKEND_PUSH_TIMEOUT_SECONDS: float = 5.0
BACKEND_PUSH_RETRIES: int = 1


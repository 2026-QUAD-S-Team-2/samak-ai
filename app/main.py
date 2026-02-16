from __future__ import annotations

"""
FastAPI 엔트리포인트.

MVP: "이미지 기반 공고/채팅 분석 + Gemini 자연어 생성"

- /v1/analyze/image: image(multipart) 또는 imageUrl(JSON) 입력 → OCR → ML → 점수 → 요약문 생성
- /healthz: 헬스체크

주의:
- 텍스트 직접 입력 추론 API(/v1/infer)는 이 MVP에서는 제공하지 않습니다.
"""

from fastapi import FastAPI

import logging
import os

from app.env import load_dotenv_once
from app.routes.analyze import router as analyze_router

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # LOG_LEVEL=DEBUG/INFO/WARNING...
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


load_dotenv_once()
_configure_logging()

app = FastAPI(title="Samak AI - Image Analyze MVP", version="0.1.0")
app.include_router(analyze_router)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}

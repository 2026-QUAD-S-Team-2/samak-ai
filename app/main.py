from __future__ import annotations

"""
FastAPI 엔트리포인트.

- /v1/analyze/image: 이미지 입력 → Gemini Vision → 규칙 기반 신호 추출 → 점수 → 요약문 생성
- /healthz: 헬스체크
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

import asyncio
import logging
import os

from app.env import load_dotenv_once
from app.pubsub.consumer import start_consumer
from app.routes.analyze import router as analyze_router

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # LOG_LEVEL=DEBUG/INFO/WARNING...
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


load_dotenv_once()
_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    consumer_task = asyncio.create_task(start_consumer())
    yield
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Samak AI - Image Analyze MVP", version="0.1.0", lifespan=lifespan)
app.include_router(analyze_router)


@app.get("/healthz")
def healthz() -> dict:
    from app.pubsub.pubsub_config import GCP_PROJECT_ID
    pubsub_status = "configured" if GCP_PROJECT_ID else "not_configured"
    status = "ok" if pubsub_status == "configured" else "degraded"
    return {"status": status, "checks": {"pubsub": pubsub_status}}


# 백엔드/인프라에서 `/health`로 확인하는 경우도 많아서 alias를 제공합니다.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

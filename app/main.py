from __future__ import annotations

"""
FastAPI 엔트리포인트.

MVP: "이미지 기반 공고/채팅 분석 + Gemini 자연어 생성"

- /v1/analyze/image: imageUrl/countryCode(JSON) 입력 → 이미지 다운로드 → OCR → ML → 점수 → 요약문 생성
- /healthz: 헬스체크

주의:
- 텍스트 직접 입력 추론 API(/v1/infer)는 이 MVP에서는 제공하지 않습니다.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

import asyncio
import logging
import os

from app.env import load_dotenv_once
from app.mq.consumer import start_consumer
from app.routes.analyze import router as analyze_router, _init_model
from app.routes.wage_warning import router as wage_warning_router

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
    _init_model()
    consumer_task = asyncio.create_task(start_consumer())
    yield
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Samak AI - Image Analyze MVP", version="0.1.0", lifespan=lifespan)
app.include_router(analyze_router)
app.include_router(wage_warning_router)


@app.get("/healthz")
def healthz() -> dict:
    checks: dict[str, str] = {}

    model_dir = os.environ.get("MODEL_DIR", "models/fraud-baseline")
    checks["model"] = "ok" if os.path.isdir(model_dir) else "missing"

    min_wage_env = os.environ.get("MIN_WAGE_DATA_PATH", "resources/min_wage_hourly.json")
    min_wage_p = Path(min_wage_env) if Path(min_wage_env).is_absolute() else Path(__file__).resolve().parent.parent / min_wage_env
    checks["minWageData"] = "ok" if min_wage_p.exists() else "missing"

    from app.mq.mq_config import GCP_PROJECT_ID
    checks["pubsub"] = "configured" if GCP_PROJECT_ID else "not_configured"

    status = "ok" if all(v in ("ok", "configured") for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


# 백엔드/인프라에서 `/health`로 확인하는 경우도 많아서 alias를 제공합니다.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

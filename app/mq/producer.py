from __future__ import annotations

import asyncio
import logging

from google.cloud import pubsub_v1

from app.mq.mq_config import PUBSUB_RESULT_TOPIC_PATH
from app.mq.schemas import AnalysisResultMessage

logger = logging.getLogger(__name__)

_publisher: pubsub_v1.PublisherClient | None = None


def _get_publisher() -> pubsub_v1.PublisherClient:
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


async def publish_result(result: AnalysisResultMessage) -> None:
    data = result.model_dump_json().encode()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _get_publisher().publish(PUBSUB_RESULT_TOPIC_PATH, data=data).result(),
    )
    logger.info(
        "분석 결과 발행 완료: analysisId=%s riskLevel=%s",
        result.analysisId,
        result.riskLevel,
    )

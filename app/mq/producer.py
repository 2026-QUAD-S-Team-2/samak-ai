from __future__ import annotations

import json
import logging

import aio_pika

from app.mq.mq_config import RABBITMQ_RESULT_ROUTING_KEY
from app.mq.schemas import AnalysisResultMessage

logger = logging.getLogger(__name__)


async def publish_result(
    result_exchange: aio_pika.abc.AbstractExchange,
    result: AnalysisResultMessage,
) -> None:
    body = result.model_dump_json().encode()
    await result_exchange.publish(
        aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=RABBITMQ_RESULT_ROUTING_KEY,
    )
    logger.info("분석 결과 발행 완료: analysisId=%s riskLevel=%s", result.analysisId, result.riskLevel)

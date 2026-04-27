from __future__ import annotations

import asyncio
import logging

import aio_pika

from app.mq.mq_config import (
    RABBITMQ_PREFETCH,
    RABBITMQ_RECONNECT_DELAY,
    RABBITMQ_REQUEST_EXCHANGE,
    RABBITMQ_REQUEST_QUEUE,
    RABBITMQ_REQUEST_ROUTING_KEY,
    RABBITMQ_RESULT_EXCHANGE,
    RABBITMQ_RESULT_QUEUE,
    RABBITMQ_RESULT_ROUTING_KEY,
    RABBITMQ_URL,
)
from app.mq.producer import publish_result
from app.mq.schemas import AnalysisRequestMessage, AnalysisResultMessage
from app.routes.analyze import _run_analysis

logger = logging.getLogger(__name__)


async def _process_message(
    message: aio_pika.abc.AbstractIncomingMessage,
    result_exchange: aio_pika.abc.AbstractExchange,
) -> None:
    async with message.process(requeue=False):
        req = AnalysisRequestMessage.model_validate_json(message.body)
        logger.info(
            "분석 요청 수신: analysisItemId=%d, 이미지 %d장",
            req.analysisItemId,
            len(req.imageUrls),
        )

        tasks = [
            _run_analysis(None, url, {}, req.debug)
            for url in req.imageUrls
        ]
        if len(tasks) >= 5:
            raw_results = list(await asyncio.gather(*tasks))
        else:
            raw_results = [await t for t in tasks]

        # riskScore 최고값 결과를 대표값으로 선택
        best = max(raw_results, key=lambda r: r["mlPrediction"]["riskScore"] or 0)

        result = AnalysisResultMessage(
            analysisId=str(req.analysisItemId),
            fraudProbability=best["mlPrediction"]["fraudProbability"],
            riskScore=best["mlPrediction"]["riskScore"],
            riskLevel=best["mlPrediction"]["riskLevel"],
            riskSignals=best["explanation"]["riskSignals"],
            travelBanRegionsMatched=best.get("travelBanRegionsMatched", []),
            message=best["analysisSummary"]["message"],
        )
        await publish_result(result_exchange, result)


async def start_consumer() -> None:
    while True:
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            logger.info("RabbitMQ 연결 성공: %s", RABBITMQ_URL)

            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=RABBITMQ_PREFETCH)

                req_exchange = await channel.declare_exchange(
                    RABBITMQ_REQUEST_EXCHANGE,
                    aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )
                res_exchange = await channel.declare_exchange(
                    RABBITMQ_RESULT_EXCHANGE,
                    aio_pika.ExchangeType.DIRECT,
                    durable=True,
                )

                req_queue = await channel.declare_queue(RABBITMQ_REQUEST_QUEUE, durable=True)
                await req_queue.bind(req_exchange, routing_key=RABBITMQ_REQUEST_ROUTING_KEY)

                res_queue = await channel.declare_queue(RABBITMQ_RESULT_QUEUE, durable=True)
                await res_queue.bind(res_exchange, routing_key=RABBITMQ_RESULT_ROUTING_KEY)

                async def on_message(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
                    await _process_message(msg, res_exchange)

                await req_queue.consume(on_message)
                logger.info("분석 요청 큐 소비 시작: %s", RABBITMQ_REQUEST_QUEUE)

                await asyncio.Future()  # 연결 유지

        except asyncio.CancelledError:
            logger.info("RabbitMQ consumer 종료")
            raise
        except Exception as e:
            logger.error(
                "RabbitMQ 연결 오류: %s — %d초 후 재연결 시도",
                e,
                RABBITMQ_RECONNECT_DELAY,
            )
            await asyncio.sleep(RABBITMQ_RECONNECT_DELAY)

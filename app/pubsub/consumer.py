from __future__ import annotations

import asyncio
import logging

from google.cloud import pubsub_v1

from app.pubsub.pubsub_config import (
    PUBSUB_DLQ_SUBSCRIPTION_PATH,
    PUBSUB_RECONNECT_DELAY,
    PUBSUB_REQUEST_SUBSCRIPTION_PATH,
)
from app.pubsub.producer import publish_result
from app.pubsub.schemas import AnalysisRequestMessage, AnalysisResultMessage
from app.routes.analyze import _download_image_bytes, _run_analysis

logger = logging.getLogger(__name__)


async def _process_message(message: pubsub_v1.subscriber.message.Message) -> None:
    req: AnalysisRequestMessage | None = None
    try:
        req = AnalysisRequestMessage.model_validate_json(message.data.decode())
        logger.info(
            "분석 요청 수신: analysisItemId=%d, 이미지 %d장",
            req.analysisItemId,
            len(req.imageUrls),
        )

        async def _fetch_and_analyze(url: str) -> dict:
            from fastapi import HTTPException
            from uuid import uuid4
            try:
                bts = await _download_image_bytes(url)
            except HTTPException as e:
                logger.error("이미지 다운로드 실패: url=%s detail=%s", url, e.detail)
                return {
                    "analysisId": str(uuid4()),
                    "mlPrediction": {"fraudProbability": None, "riskScore": None, "riskLevel": None, "modelVersion": "gemini-rule-v1.0.0", "thresholdUsed": None},
                    "explanation": {"riskSignals": []},
                    "travelBanRegionsMatched": [],
                    "analysisSummary": {"score": None, "label": None, "message": str(e.detail)},
                }
            meta: dict[str, object] = {}
            if req.companyName:
                meta["companyName"] = req.companyName
            if req.countryCode:
                meta["countryCode"] = req.countryCode
            return await _run_analysis(bts, meta, req.debug)

        if len(req.imageUrls) >= 5:
            raw_results = list(await asyncio.gather(*[_fetch_and_analyze(u) for u in req.imageUrls]))
        else:
            raw_results = [await _fetch_and_analyze(u) for u in req.imageUrls]

        best = max(raw_results, key=lambda r: r["mlPrediction"]["riskScore"] or 0)

        result = AnalysisResultMessage(
            analysisId=str(req.analysisItemId),
            fraudProbability=best["mlPrediction"]["fraudProbability"],
            riskScore=best["mlPrediction"]["riskScore"],
            riskLevel=best["mlPrediction"]["riskLevel"],
            trustScore=best["ui"]["trustScore"],
            riskSignals=best["explanation"]["riskSignals"],
            travelBanRegionsMatched=best.get("travelBanRegionsMatched", []),
            message=best["analysisSummary"]["message"],
            location=best.get("location"),
        )
        await publish_result(result)
        message.ack()

    except Exception as e:
        logger.error(
            "분석 처리 실패: analysisItemId=%s error=%s",
            req.analysisItemId if req else "?",
            e,
        )
        message.nack()


async def start_consumer() -> None:
    loop = asyncio.get_running_loop()
    flow_control = pubsub_v1.types.FlowControl(max_messages=1)

    def _make_callback(event_loop: asyncio.AbstractEventLoop):
        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            future = asyncio.run_coroutine_threadsafe(_process_message(message), event_loop)
            try:
                future.result(timeout=300)
            except TimeoutError:
                logger.error("메시지 처리 타임아웃 (300s 초과)")
                message.nack()
            except Exception as e:
                logger.error("콜백 오류: %s", e)
                message.nack()
        return callback

    def _make_dlq_callback(event_loop: asyncio.AbstractEventLoop):
        async def _log_dlq(message: pubsub_v1.subscriber.message.Message) -> None:
            body_preview = message.data[:300].decode("utf-8", errors="replace")
            logger.error(
                "ERROR DLQ 메시지 수신 — 처리에 실패한 요청입니다. "
                "attributes=%s body=%s",
                dict(message.attributes),
                body_preview,
            )
            message.ack()

        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            future = asyncio.run_coroutine_threadsafe(_log_dlq(message), event_loop)
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.error("DLQ 콜백 오류: %s", e)
                message.nack()
        return callback

    while True:
        subscriber = pubsub_v1.SubscriberClient()
        dlq_future = None
        req_future = None
        try:
            dlq_future = subscriber.subscribe(
                PUBSUB_DLQ_SUBSCRIPTION_PATH,
                callback=_make_dlq_callback(loop),
            )
            req_future = subscriber.subscribe(
                PUBSUB_REQUEST_SUBSCRIPTION_PATH,
                callback=_make_callback(loop),
                flow_control=flow_control,
            )
            logger.info(
                "Pub/Sub 구독 시작: %s (DLQ: %s)",
                PUBSUB_REQUEST_SUBSCRIPTION_PATH,
                PUBSUB_DLQ_SUBSCRIPTION_PATH,
            )
            await loop.run_in_executor(None, req_future.result)

        except asyncio.CancelledError:
            if req_future:
                req_future.cancel()
            if dlq_future:
                dlq_future.cancel()
            subscriber.close()
            logger.info("Pub/Sub consumer 종료")
            raise
        except Exception as e:
            logger.error(
                "Pub/Sub 연결 오류: %s — %d초 후 재연결 시도",
                e,
                PUBSUB_RECONNECT_DELAY,
            )
            if req_future:
                req_future.cancel()
            if dlq_future:
                dlq_future.cancel()
            subscriber.close()
            await asyncio.sleep(PUBSUB_RECONNECT_DELAY)

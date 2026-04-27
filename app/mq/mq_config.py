from __future__ import annotations

import os

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

RABBITMQ_REQUEST_EXCHANGE = os.environ.get("RABBITMQ_REQUEST_EXCHANGE", "analysis-request-exchange")
RABBITMQ_RESULT_EXCHANGE  = os.environ.get("RABBITMQ_RESULT_EXCHANGE",  "analysis-result-exchange")

RABBITMQ_REQUEST_QUEUE = os.environ.get("RABBITMQ_REQUEST_QUEUE", "analysis-request-queue")
RABBITMQ_RESULT_QUEUE  = os.environ.get("RABBITMQ_RESULT_QUEUE",  "analysis-result-queue")

RABBITMQ_REQUEST_ROUTING_KEY = os.environ.get("RABBITMQ_REQUEST_ROUTING_KEY", "analysis-request-routing-key")
RABBITMQ_RESULT_ROUTING_KEY  = os.environ.get("RABBITMQ_RESULT_ROUTING_KEY",  "analysis-result-routing-key")

RABBITMQ_PREFETCH        = int(os.environ.get("RABBITMQ_PREFETCH", "1"))
RABBITMQ_RECONNECT_DELAY = int(os.environ.get("RABBITMQ_RECONNECT_DELAY", "5"))

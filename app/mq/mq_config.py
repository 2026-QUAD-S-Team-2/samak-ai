from __future__ import annotations

import os

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")

PUBSUB_REQUEST_SUBSCRIPTION  = os.environ.get("PUBSUB_REQUEST_SUBSCRIPTION",  "analysis-request-subscription")
PUBSUB_RESULT_TOPIC          = os.environ.get("PUBSUB_RESULT_TOPIC",          "analysis-result-topic")
PUBSUB_DLQ_SUBSCRIPTION      = os.environ.get("PUBSUB_DLQ_SUBSCRIPTION",      "analysis-request-dead-letter-subscription")

PUBSUB_RECONNECT_DELAY = int(os.environ.get("PUBSUB_RECONNECT_DELAY", "5"))

PUBSUB_REQUEST_SUBSCRIPTION_PATH = (
    f"projects/{GCP_PROJECT_ID}/subscriptions/{PUBSUB_REQUEST_SUBSCRIPTION}"
)
PUBSUB_RESULT_TOPIC_PATH = (
    f"projects/{GCP_PROJECT_ID}/topics/{PUBSUB_RESULT_TOPIC}"
)
PUBSUB_DLQ_SUBSCRIPTION_PATH = (
    f"projects/{GCP_PROJECT_ID}/subscriptions/{PUBSUB_DLQ_SUBSCRIPTION}"
)

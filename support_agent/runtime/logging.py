from __future__ import annotations

import json
import logging
from typing import Any


LOGGER_NAME = "support_agent"
SENSITIVE_KEYS = {
    "mobile",
    "phone",
    "phone_number",
    "email",
    "payment_id",
    "transaction_id",
    "payment_session_id",
    "vin",
    "order_id",
    "order_number",
}


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, **fields: Any) -> None:
    logger = get_logger()
    payload = {"event": event, **redact_for_logging(fields)}
    logger.info(json.dumps(payload, default=str))


def redact_for_logging(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: (_mask_value(item) if key.lower() in SENSITIVE_KEYS else redact_for_logging(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_for_logging(item) for item in value]
    return value


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}***{value[-2:]}"
    return "***"


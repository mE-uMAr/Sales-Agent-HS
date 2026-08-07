"""Structured JSON logging with session correlation.

Two rules this module exists to enforce:

1. Every log line is one JSON object, so it can be shipped anywhere without a
   custom parser.
2. Personal data never lands in logs. Use :func:`hash_pii` for anything that
   identifies a visitor; the hash is stable, so you can still correlate.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

#: Set once per request so every downstream log line carries the session id.
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        session_id = session_id_var.get()
        if session_id:
            payload["session_id"] = session_id

        # Anything passed via logger.info("...", extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty and rarely tell us anything we do not already log.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Telemetry failures are not our problem and log at ERROR.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def hash_pii(value: str | None) -> str | None:
    """Stable, non-reversible identifier for logs. Never log the raw value."""
    if not value:
        return None
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()[:16]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

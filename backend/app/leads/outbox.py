"""Background delivery of captured leads to an external system.

The visitor never waits on the CRM. ``LeadService.capture`` writes the lead
locally and returns; this worker drains the pending rows afterwards with
exponential backoff. If the CRM is down for a day, nothing is lost — the leads
sit in the table until it comes back.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import UTC, datetime, timedelta

from app.config import Settings, get_settings
from app.leads import repository
from app.leads.sink import LeadDeliveryError, LeadSink
from app.observability import get_logger
from app.persistence.db import session_scope

logger = get_logger(__name__)

BASE_BACKOFF_SECONDS = 30
MAX_BACKOFF_SECONDS = 3600


def next_attempt_delay(attempts: int) -> timedelta:
    """Exponential backoff with jitter, capped at an hour.

    ``attempts`` is the number already made, so the first retry waits
    ``BASE_BACKOFF_SECONDS``.
    """
    raw = BASE_BACKOFF_SECONDS * (2**max(0, attempts))
    capped = min(raw, MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0.8, 1.2)
    return timedelta(seconds=capped * jitter)


class OutboxWorker:
    """Polls for undelivered leads and pushes them through the sink."""

    def __init__(self, sink: LeadSink, settings: Settings | None = None) -> None:
        self._sink = sink
        self._settings = settings or get_settings()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def run_once(self) -> int:
        """Process one batch. Returns how many leads were attempted."""
        async with session_scope() as session:
            due = await repository.claim_due_deliveries(
                session, limit=self._settings.outbox_batch_size
            )
            if not due:
                return 0

            for lead in due:
                record = lead.to_record()
                try:
                    reference = await self._sink.deliver(record)
                except LeadDeliveryError as exc:
                    exhausted = (
                        not exc.retryable
                        or lead.delivery_attempts + 1 >= self._settings.outbox_max_attempts
                    )
                    await repository.mark_delivery_failed(
                        session,
                        lead,
                        error=str(exc),
                        next_attempt_at=(
                            None
                            if exhausted
                            else datetime.now(UTC)
                            + next_attempt_delay(lead.delivery_attempts)
                        ),
                    )
                    logger.warning(
                        "lead delivery failed",
                        extra={
                            "lead_id": lead.id,
                            "attempts": lead.delivery_attempts,
                            "retryable": exc.retryable,
                            "exhausted": exhausted,
                            "error": str(exc)[:300],
                        },
                    )
                except Exception:
                    # An unexpected bug must not poison the batch or the loop.
                    await repository.mark_delivery_failed(
                        session,
                        lead,
                        error="unexpected error during delivery",
                        next_attempt_at=datetime.now(UTC)
                        + next_attempt_delay(lead.delivery_attempts),
                    )
                    logger.exception(
                        "unexpected lead delivery error", extra={"lead_id": lead.id}
                    )
                else:
                    await repository.mark_delivered(session, lead, reference)

            return len(due)

    async def _loop(self) -> None:
        interval = self._settings.outbox_poll_seconds
        while not self._stopping.is_set():
            try:
                processed = await self.run_once()
                if processed:
                    logger.info("outbox batch processed", extra={"count": processed})
            except Exception:
                logger.exception("outbox loop iteration failed")

            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except TimeoutError:
                continue

    def start(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="lead-outbox")
        logger.info(
            "outbox worker started",
            extra={"sink": self._sink.name, "interval_s": self._settings.outbox_poll_seconds},
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        # Shutdown is best-effort: a worker dying badly must not stop the
        # process from exiting.
        with contextlib.suppress(Exception):
            await self._task
        self._task = None
        logger.info("outbox worker stopped")

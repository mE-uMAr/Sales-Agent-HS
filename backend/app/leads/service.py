"""The one entry point the rest of the application uses to capture a lead."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.leads import repository
from app.leads.models import DeliveryStatus, LeadRecord
from app.leads.scoring import explain_score, score_lead
from app.leads.sink import LeadSink, build_sink
from app.observability import get_logger, hash_pii
from app.persistence.db import session_scope

logger = get_logger(__name__)


class LeadService:
    """Persists leads locally, then queues them for forwarding if configured.

    ``capture`` is idempotent on ``session_id``: a conversation can be closed
    twice — by the user, by the idle sweeper, by a retried request — and still
    produce exactly one lead.
    """

    def __init__(self, sink: LeadSink, settings: Settings | None = None) -> None:
        self._sink = sink
        self._settings = settings or get_settings()

    @property
    def sink(self) -> LeadSink:
        """The configured forwarding target — the outbox worker drives it."""
        return self._sink

    @property
    def forwarding_enabled(self) -> bool:
        return self._settings.lead_sink != "sqlite"

    async def capture(self, record: LeadRecord) -> LeadRecord:
        record.lead_score = score_lead(record)

        async with session_scope() as session:
            existing = await repository.get_by_session_id(session, record.session_id)
            if existing is not None:
                logger.info(
                    "lead already captured for session; skipping duplicate",
                    extra={"lead_id": existing.id, "session_id": record.session_id},
                )
                return existing.to_record()

            status = (
                DeliveryStatus.PENDING
                if self.forwarding_enabled
                else DeliveryStatus.NOT_REQUIRED
            )
            await repository.insert_lead(session, record, delivery_status=status)

        logger.info(
            "lead captured",
            extra={
                "lead_id": record.id,
                "session_id": record.session_id,
                "handoff_reason": record.handoff_reason.value,
                "lead_score": record.lead_score,
                "score_breakdown": explain_score(record),
                "contact_email_hash": hash_pii(record.contact.email),
                "unanswered_count": len(record.unanswered_questions),
                "quoted_count": len(record.quoted_prices),
                "forwarding": self.forwarding_enabled,
            },
        )
        return record

    async def aclose(self) -> None:
        await self._sink.aclose()


_service: LeadService | None = None


def get_lead_service() -> LeadService:
    global _service
    if _service is None:
        _service = LeadService(build_sink())
    return _service


async def close_lead_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
    _service = None

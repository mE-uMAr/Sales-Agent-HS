"""The forwarding seam.

A :class:`LeadSink` delivers a lead somewhere *outside* this service. The local
database is not a sink — it is written unconditionally by
:class:`~app.leads.service.LeadService` before any delivery is attempted, so a
lead is durable even when every downstream system is unreachable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.leads.models import LeadRecord


class LeadDeliveryError(RuntimeError):
    """Delivery failed. Raise this to have the outbox retry with backoff."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class LeadSink(Protocol):
    """Forward a lead to an external system.

    Implementations must be idempotent on ``record.id`` — the outbox retries, so
    the same lead can legitimately arrive twice.
    """

    name: str

    async def deliver(self, record: LeadRecord) -> str:
        """Return a downstream reference id. Raise :class:`LeadDeliveryError`."""
        ...

    async def aclose(self) -> None:
        """Release any held resources."""
        ...


def build_sink(settings: Settings | None = None) -> LeadSink:
    """Resolve the configured forwarding target."""
    settings = settings or get_settings()

    if settings.lead_sink == "http":
        from app.leads.sinks.http import HttpLeadSink

        if not settings.crm_webhook_url:
            raise ValueError("LEAD_SINK=http requires CRM_WEBHOOK_URL to be set")
        return HttpLeadSink(
            url=settings.crm_webhook_url,
            secret=(
                settings.crm_webhook_secret.get_secret_value()
                if settings.crm_webhook_secret
                else None
            ),
        )

    from app.leads.sinks.null import NullLeadSink

    return NullLeadSink()

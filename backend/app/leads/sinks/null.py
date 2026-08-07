"""No-op sink — used when the local database *is* the destination."""

from __future__ import annotations

from app.leads.models import LeadRecord


class NullLeadSink:
    """Accepts every lead and forwards it nowhere."""

    name = "null"

    async def deliver(self, record: LeadRecord) -> str:
        return record.id

    async def aclose(self) -> None:
        return None

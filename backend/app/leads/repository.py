"""Data access for the ``leads`` table.

Plain functions over an ``AsyncSession`` — no hidden session ownership, so the
caller controls the transaction boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.leads.models import DeliveryStatus, Lead, LeadRecord


def _now() -> datetime:
    return datetime.now(UTC)


async def get_by_session_id(session: AsyncSession, session_id: str) -> Lead | None:
    result = await session.execute(select(Lead).where(Lead.session_id == session_id))
    return result.scalar_one_or_none()


async def insert_lead(
    session: AsyncSession,
    record: LeadRecord,
    *,
    delivery_status: DeliveryStatus,
) -> Lead:
    lead = Lead(
        id=record.id,
        session_id=record.session_id,
        created_at=record.captured_at,
        contact_name=record.contact.name,
        contact_email=record.contact.email,
        contact_company=record.contact.company,
        handoff_reason=record.handoff_reason.value,
        lead_score=record.lead_score,
        payload=record.model_dump(mode="json"),
        delivery_status=delivery_status.value,
        delivery_attempts=0,
        delivery_next_attempt_at=(
            _now() if delivery_status is DeliveryStatus.PENDING else None
        ),
    )
    session.add(lead)
    await session.flush()
    return lead


async def claim_due_deliveries(
    session: AsyncSession, *, limit: int, now: datetime | None = None
) -> list[Lead]:
    """Pending leads whose backoff window has elapsed, oldest first."""
    now = now or _now()
    query = (
        select(Lead)
        .where(
            Lead.delivery_status == DeliveryStatus.PENDING.value,
            Lead.delivery_next_attempt_at <= now,
        )
        .order_by(Lead.delivery_next_attempt_at.asc())
        .limit(limit)
    )
    # SQLite has no row locking; on Postgres this keeps multiple workers from
    # picking up the same lead.
    if session.bind is not None and session.bind.dialect.name != "sqlite":
        query = query.with_for_update(skip_locked=True)

    result = await session.execute(query)
    return list(result.scalars().all())


async def mark_delivered(session: AsyncSession, lead: Lead, reference: str) -> None:
    lead.delivery_status = DeliveryStatus.SENT.value
    lead.delivered_at = _now()
    lead.delivery_attempts += 1
    lead.delivery_last_error = None
    lead.delivery_next_attempt_at = None
    payload = dict(lead.payload)
    payload["crm_reference"] = reference
    lead.payload = payload


async def mark_delivery_failed(
    session: AsyncSession,
    lead: Lead,
    *,
    error: str,
    next_attempt_at: datetime | None,
) -> None:
    """Reschedule, or give up when ``next_attempt_at`` is ``None``."""
    lead.delivery_attempts += 1
    lead.delivery_last_error = error[:2000]
    if next_attempt_at is None:
        lead.delivery_status = DeliveryStatus.FAILED.value
        lead.delivery_next_attempt_at = None
    else:
        lead.delivery_next_attempt_at = next_attempt_at


async def list_leads(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    since: datetime | None = None,
    min_score: int | None = None,
) -> list[Lead]:
    query = select(Lead).order_by(Lead.created_at.desc())
    if since is not None:
        query = query.where(Lead.created_at >= since)
    if min_score is not None:
        query = query.where(Lead.lead_score >= min_score)
    result = await session.execute(query.limit(limit).offset(offset))
    return list(result.scalars().all())


async def count_by_delivery_status(session: AsyncSession) -> dict[str, int]:
    from sqlalchemy import func

    result = await session.execute(
        select(Lead.delivery_status, func.count()).group_by(Lead.delivery_status)
    )
    return dict(result.all())

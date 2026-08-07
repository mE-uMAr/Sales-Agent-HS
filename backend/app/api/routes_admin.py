"""Operational endpoints. All behind the admin key except the liveness probe."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query

from app import __version__
from app.api.schemas import HealthOut, LeadOut, ReindexOut
from app.api.security import require_admin_key
from app.config import get_settings
from app.knowledge.ingest import build_index
from app.knowledge.retriever import get_retriever, reset_retriever
from app.leads import repository
from app.leads.models import DeliveryStatus
from app.observability import get_logger
from app.persistence.db import session_scope

logger = get_logger(__name__)

public_router = APIRouter(tags=["ops"])
router = APIRouter(
    prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin_key)]
)


@public_router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Deliberately says nothing about internals — it is publicly reachable."""
    return {"status": "ok"}


@router.get("/health", response_model=HealthOut, summary="Detailed health")
async def detailed_health() -> HealthOut:
    settings = get_settings()

    async with session_scope() as db:
        counts = await repository.count_by_delivery_status(db)

    return HealthOut(
        status="ok",
        version=__version__,
        knowledge_base_ready=get_retriever().is_ready(),
        llm_provider=settings.llm_provider,
        lead_sink=settings.lead_sink,
        pending_deliveries=counts.get(DeliveryStatus.PENDING.value, 0),
    )


@router.post("/reindex", response_model=ReindexOut, summary="Rebuild the vector index")
async def reindex() -> ReindexOut:
    """Re-read ``content/public`` and rebuild the index.

    Lets a content edit go live without a restart. Pricing changes need no
    reindex at all — the catalog is read from disk per request.
    """
    report = build_index(reset=True)
    reset_retriever()
    logger.info("index rebuilt via admin endpoint", extra=report.as_dict())
    return ReindexOut(files=report.files, chunks=report.chunks, skipped=report.skipped)


@router.get("/leads", response_model=list[LeadOut], summary="List captured leads")
async def list_leads(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    days: int | None = Query(default=None, ge=1, le=365),
    min_score: int | None = Query(default=None, ge=0, le=100),
) -> list[LeadOut]:
    since = datetime.now(UTC) - timedelta(days=days) if days else None

    async with session_scope() as db:
        leads = await repository.list_leads(
            db, limit=limit, offset=offset, since=since, min_score=min_score
        )

    results: list[LeadOut] = []
    for lead in leads:
        record = lead.to_record()
        results.append(
            LeadOut(
                id=lead.id,
                session_id=lead.session_id,
                created_at=lead.created_at,
                contact_name=lead.contact_name,
                contact_email=lead.contact_email,
                contact_company=lead.contact_company,
                lead_score=lead.lead_score,
                handoff_reason=lead.handoff_reason,
                delivery_status=lead.delivery_status,
                concern=record.concern,
                budget_stated=record.budget_stated,
                chat_summary=record.chat_summary,
                unanswered_questions=record.unanswered_questions,
                quoted_prices=[q.model_dump(mode="json") for q in record.quoted_prices],
            )
        )
    return results


@router.post("/sweep", summary="Close idle sessions now")
async def sweep() -> dict[str, int]:
    """Runs on a timer already; exposed for testing and manual cleanup."""
    from app.chat.service import get_chat_service

    closed = await get_chat_service().sweep_idle_sessions()
    return {"closed": closed}

"""Lead capture, forwarding and the outbox.

A lead is the product of this system. These tests are about not losing one and
not writing two.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
import respx

from app.leads import repository
from app.leads.models import Contact, DeliveryStatus, HandoffReason, LeadRecord
from app.leads.outbox import (
    BASE_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    OutboxWorker,
    next_attempt_delay,
)
from app.leads.service import LeadService
from app.leads.sink import LeadDeliveryError, build_sink
from app.leads.sinks.http import HttpLeadSink
from app.leads.sinks.null import NullLeadSink
from app.persistence.db import session_scope

WEBHOOK = "https://crm.example.com/hooks/leads"


def make_record(session_id: str = "sess-1", **kwargs) -> LeadRecord:
    payload = {
        "session_id": session_id,
        "contact": Contact(name="Dana Reyes", email="dana@brightpath.io"),
        "concern": "needs a client portal",
        "use_case": "client portal with order history",
        "budget_stated": "20-30k",
        "chat_summary": "Dana wants a portal.",
    }
    payload.update(kwargs)
    return LeadRecord(**payload)


# ── capture ──────────────────────────────────────────────────────────
async def test_capture_persists_and_scores(database) -> None:
    service = LeadService(NullLeadSink())
    record = await service.capture(make_record())

    assert record.lead_score > 0

    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")
    assert stored is not None
    assert stored.contact_email == "dana@brightpath.io"
    assert stored.to_record().use_case == "client portal with order history"


async def test_capture_is_idempotent_per_session(database) -> None:
    """Two closes racing must still produce exactly one lead."""
    service = LeadService(NullLeadSink())
    first = await service.capture(make_record())
    second = await service.capture(make_record(handoff_reason=HandoffReason.MAX_TURNS))

    assert second.id == first.id

    async with session_scope() as db:
        leads = await repository.list_leads(db, limit=50)
    assert len(leads) == 1


async def test_local_only_sink_marks_delivery_not_required(database) -> None:
    service = LeadService(NullLeadSink())
    await service.capture(make_record())

    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")
    assert stored.delivery_status == DeliveryStatus.NOT_REQUIRED.value


async def test_forwarding_sink_queues_for_delivery(database, monkeypatch) -> None:
    monkeypatch.setenv("LEAD_SINK", "http")
    from app.config import get_settings

    get_settings.cache_clear()

    service = LeadService(NullLeadSink())
    await service.capture(make_record())

    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")
    assert stored.delivery_status == DeliveryStatus.PENDING.value
    assert stored.delivery_next_attempt_at is not None


# ── http sink ────────────────────────────────────────────────────────
@respx.mock
async def test_http_sink_signs_and_keys_the_request() -> None:
    route = respx.post(WEBHOOK).mock(
        return_value=httpx.Response(200, json={"id": "crm-99"})
    )
    sink = HttpLeadSink(url=WEBHOOK, secret="topsecret")
    record = make_record()

    reference = await sink.deliver(record)
    await sink.aclose()

    assert reference == "crm-99"
    request = route.calls[0].request
    assert request.headers["Idempotency-Key"] == record.id

    timestamp = request.headers["X-Hashed-Timestamp"]
    expected = hmac.new(
        b"topsecret", f"{timestamp}.".encode() + request.content, hashlib.sha256
    ).hexdigest()
    assert request.headers["X-Hashed-Signature"] == f"sha256={expected}"
    assert json.loads(request.content)["session_id"] == "sess-1"


@respx.mock
async def test_http_sink_accepts_a_bodyless_ack() -> None:
    respx.post(WEBHOOK).mock(return_value=httpx.Response(204))
    sink = HttpLeadSink(url=WEBHOOK)
    record = make_record()
    assert await sink.deliver(record) == record.id
    await sink.aclose()


@respx.mock
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (503, True), (400, False), (422, False)],
)
async def test_http_sink_classifies_failures(status: int, retryable: bool) -> None:
    respx.post(WEBHOOK).mock(return_value=httpx.Response(status, text="nope"))
    sink = HttpLeadSink(url=WEBHOOK)

    with pytest.raises(LeadDeliveryError) as caught:
        await sink.deliver(make_record())
    await sink.aclose()

    assert caught.value.retryable is retryable


@respx.mock
async def test_transport_error_is_retryable() -> None:
    respx.post(WEBHOOK).mock(side_effect=httpx.ConnectError("boom"))
    sink = HttpLeadSink(url=WEBHOOK)

    with pytest.raises(LeadDeliveryError) as caught:
        await sink.deliver(make_record())
    await sink.aclose()

    assert caught.value.retryable is True


def test_http_sink_requires_a_url(monkeypatch) -> None:
    monkeypatch.setenv("LEAD_SINK", "http")
    monkeypatch.delenv("CRM_WEBHOOK_URL", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(ValueError, match="CRM_WEBHOOK_URL"):
        build_sink()


# ── outbox ───────────────────────────────────────────────────────────
def test_backoff_grows_and_is_capped() -> None:
    first = next_attempt_delay(0).total_seconds()
    second = next_attempt_delay(1).total_seconds()
    huge = next_attempt_delay(20).total_seconds()

    # Jittered, so compare against the jitter band rather than exact values.
    assert BASE_BACKOFF_SECONDS * 0.8 <= first <= BASE_BACKOFF_SECONDS * 1.2
    assert second > first
    assert huge <= MAX_BACKOFF_SECONDS * 1.2


@respx.mock
async def test_outbox_delivers_pending_leads(database, monkeypatch) -> None:
    monkeypatch.setenv("LEAD_SINK", "http")
    from app.config import get_settings

    get_settings.cache_clear()

    respx.post(WEBHOOK).mock(return_value=httpx.Response(200, json={"id": "crm-1"}))
    sink = HttpLeadSink(url=WEBHOOK)

    await LeadService(sink).capture(make_record())
    processed = await OutboxWorker(sink).run_once()
    await sink.aclose()

    assert processed == 1
    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")
    assert stored.delivery_status == DeliveryStatus.SENT.value
    assert stored.delivered_at is not None
    assert stored.payload["crm_reference"] == "crm-1"


@respx.mock
async def test_outbox_reschedules_a_retryable_failure(database, monkeypatch) -> None:
    monkeypatch.setenv("LEAD_SINK", "http")
    from app.config import get_settings

    get_settings.cache_clear()

    respx.post(WEBHOOK).mock(return_value=httpx.Response(503))
    sink = HttpLeadSink(url=WEBHOOK)

    await LeadService(sink).capture(make_record())
    await OutboxWorker(sink).run_once()
    await sink.aclose()

    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")

    # Still pending, still ours to retry — the lead is not lost.
    assert stored.delivery_status == DeliveryStatus.PENDING.value
    assert stored.delivery_attempts == 1
    assert stored.delivery_last_error is not None


@respx.mock
async def test_outbox_gives_up_on_a_permanent_failure(database, monkeypatch) -> None:
    monkeypatch.setenv("LEAD_SINK", "http")
    from app.config import get_settings

    get_settings.cache_clear()

    respx.post(WEBHOOK).mock(return_value=httpx.Response(400, text="malformed"))
    sink = HttpLeadSink(url=WEBHOOK)

    await LeadService(sink).capture(make_record())
    await OutboxWorker(sink).run_once()
    await sink.aclose()

    async with session_scope() as db:
        stored = await repository.get_by_session_id(db, "sess-1")

    # Marked failed for a human to look at, never silently dropped.
    assert stored.delivery_status == DeliveryStatus.FAILED.value
    assert "malformed" in stored.delivery_last_error


async def test_outbox_is_a_noop_when_nothing_is_pending(database) -> None:
    assert await OutboxWorker(NullLeadSink()).run_once() == 0

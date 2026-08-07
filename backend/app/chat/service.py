"""Session lifecycle: start, turn, close, sweep.

The graph produces a reply; this service owns everything around it — persistence,
the turn ceiling, the idle sweeper, and the single path by which a conversation
becomes a lead.

Closing is idempotent. A conversation can be ended by the visitor, by the
agent, by the turn cap and by the idle sweeper more or less simultaneously, and
must still produce exactly one lead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.chat.copy import MAX_TURNS_CLOSING, greeting
from app.chat.graph import get_graph
from app.chat.guards import sanitise_input
from app.chat.records import ChatMessage, ChatSession, MessageRole, SessionStatus
from app.chat.state import Stage, compute_stage, new_state
from app.chat.summarizer import summarise_conversation
from app.config import Settings, get_settings
from app.leads import Contact, HandoffReason, LeadRecord, QuotedPrice, get_lead_service
from app.observability import get_logger, hash_pii, session_id_var
from app.persistence.db import session_scope

logger = get_logger(__name__)


class SessionNotFoundError(LookupError):
    """No such session."""


class SessionClosedError(RuntimeError):
    """The conversation has already ended."""


@dataclass(slots=True)
class StartSessionRequest:
    name: str
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    page_url: str | None = None
    utm: dict[str, str] = field(default_factory=dict)
    client_hash: str | None = None


@dataclass(slots=True)
class TurnResult:
    session_id: str
    reply: str
    stage: str
    closed: bool
    handoff_reason: str | None = None


def _now() -> datetime:
    return datetime.now(UTC)


class ChatService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ── lifecycle ────────────────────────────────────────────────────
    async def start_session(self, request: StartSessionRequest) -> tuple[str, str]:
        """Create a session and return ``(session_id, opening_message)``.

        The greeting is canned rather than generated: it is instant, costs
        nothing, and there is no useful variation to be had in "hello".
        """
        contact = Contact(name=request.name, email=request.email)
        opening = greeting(contact.first_name, self._settings.company_name)

        async with session_scope() as db:
            session = ChatSession(
                contact_name=request.name.strip(),
                contact_email=request.email,
                contact_phone=request.phone,
                contact_company=request.company,
                page_url=request.page_url,
                utm=request.utm or {},
                client_hash=request.client_hash,
                stage=Stage.GREETING.value,
                slots={},
                quoted_prices=[],
                unanswered=[],
            )
            db.add(session)
            await db.flush()

            db.add(
                ChatMessage(
                    session_id=session.id,
                    role=MessageRole.ASSISTANT,
                    content=opening,
                )
            )
            session_id = session.id

        logger.info(
            "session started",
            extra={
                "session_id": session_id,
                "contact_email_hash": hash_pii(request.email),
                "page_url": request.page_url,
            },
        )
        return session_id, opening

    async def send_message(self, session_id: str, text: str) -> TurnResult:
        session_id_var.set(session_id)
        cleaned = sanitise_input(text)
        if not cleaned:
            raise ValueError("message is empty")

        async with session_scope() as db:
            session = await self._load(db, session_id)
            if not session.is_active:
                raise SessionClosedError(session_id)

            # Turn ceiling is checked before spending a model call on it.
            if session.turn_count >= self._settings.max_turns:
                history = [(m.role, m.content) for m in session.messages]
                snapshot = _snapshot(session)
                db.add(
                    ChatMessage(
                        session_id=session.id,
                        role=MessageRole.ASSISTANT,
                        content=MAX_TURNS_CLOSING,
                    )
                )
                await self._finalise(
                    db,
                    session,
                    reason=HandoffReason.MAX_TURNS,
                    history=history,
                    snapshot=snapshot,
                )
                return TurnResult(
                    session_id=session_id,
                    reply=MAX_TURNS_CLOSING,
                    stage=Stage.HANDOFF.value,
                    closed=True,
                    handoff_reason=HandoffReason.MAX_TURNS.value,
                )

            history = [(m.role, m.content) for m in session.messages]
            db.add(
                ChatMessage(
                    session_id=session.id, role=MessageRole.HUMAN, content=cleaned
                )
            )

            state = new_state(
                session_id=session.id,
                first_name=Contact(name=session.contact_name).first_name,
                stage=session.stage,
                slots=dict(session.slots or {}),
                quoted_prices=list(session.quoted_prices or []),
                unanswered=list(session.unanswered or []),
                unanswered_streak=session.unanswered_streak,
                turn_count=session.turn_count,
                messages=[*_to_langchain(history), HumanMessage(content=cleaned)],
            )

            result: dict[str, Any] = await get_graph().ainvoke(state)

            reply = ""
            for message in reversed(result.get("messages", [])):
                if isinstance(message, AIMessage) and isinstance(message.content, str):
                    reply = message.content.strip()
                    break

            session.slots = result.get("slots", session.slots)
            session.quoted_prices = _dedupe_quotes(result.get("quoted_prices", []))
            session.unanswered = result.get("unanswered", session.unanswered)
            session.unanswered_streak = result.get("unanswered_streak", 0)
            session.turn_count += 1
            session.last_activity_at = _now()
            session.stage = compute_stage(
                session.slots,
                turn_count=session.turn_count,
                terminating=bool(result.get("terminate")),
            ).value

            db.add(
                ChatMessage(
                    session_id=session.id,
                    role=MessageRole.ASSISTANT,
                    content=reply,
                    tool_trace=result.get("tool_trace", []),
                )
            )

            terminating = bool(result.get("terminate"))
            reached_cap = session.turn_count >= self._settings.max_turns

            if terminating or reached_cap:
                reason = HandoffReason(
                    result.get("handoff_reason") or HandoffReason.MAX_TURNS.value
                )
                await self._finalise(
                    db,
                    session,
                    reason=reason,
                    history=[*history, ("human", cleaned), ("assistant", reply)],
                    snapshot=_snapshot(session),
                )
                return TurnResult(
                    session_id=session_id,
                    reply=reply,
                    stage=Stage.HANDOFF.value,
                    closed=True,
                    handoff_reason=reason.value,
                )

            logger.info(
                "turn completed",
                extra={
                    "session_id": session_id,
                    "stage": session.stage,
                    "turn": session.turn_count,
                    "tools": [t.get("tool") for t in result.get("tool_trace", [])],
                },
            )
            return TurnResult(
                session_id=session_id,
                reply=reply,
                stage=session.stage,
                closed=False,
            )

    async def close_session(
        self,
        session_id: str,
        reason: HandoffReason = HandoffReason.COMPLETED,
    ) -> LeadRecord | None:
        """End a conversation and capture the lead. Safe to call repeatedly."""
        session_id_var.set(session_id)
        async with session_scope() as db:
            session = await self._load(db, session_id)
            if session.lead_saved_at is not None:
                logger.info("session already closed", extra={"session_id": session_id})
                return None
            return await self._finalise(
                db,
                session,
                reason=reason,
                history=[(m.role, m.content) for m in session.messages],
                snapshot=_snapshot(session),
            )

    async def sweep_idle_sessions(self) -> int:
        """Close conversations the visitor walked away from."""
        cutoff = _now() - timedelta(minutes=self._settings.session_idle_minutes)

        async with session_scope() as db:
            result = await db.execute(
                select(ChatSession).where(
                    ChatSession.status == SessionStatus.ACTIVE.value,
                    ChatSession.last_activity_at < cutoff,
                )
            )
            stale = list(result.scalars().all())

        closed = 0
        for session in stale:
            try:
                await self.close_session(session.id, HandoffReason.IDLE_TIMEOUT)
                closed += 1
            except Exception:
                logger.exception(
                    "failed to close idle session", extra={"session_id": session.id}
                )

        if closed:
            logger.info("idle sessions closed", extra={"count": closed})
        return closed

    # ── internals ────────────────────────────────────────────────────
    async def _load(self, db: Any, session_id: str) -> ChatSession:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def _finalise(
        self,
        db: Any,
        session: ChatSession,
        *,
        reason: HandoffReason,
        history: list[tuple[str, str]],
        snapshot: dict[str, Any],
    ) -> LeadRecord | None:
        """Summarise, build the lead, hand it to the leads module, close up.

        Guarded by ``lead_saved_at`` so concurrent closes cannot duplicate.
        """
        if session.lead_saved_at is not None:
            return None

        summary = await summarise_conversation(
            messages=history,
            slots=snapshot["slots"],
            quoted_prices=snapshot["quoted_prices"],
            unanswered=snapshot["unanswered"],
        )

        record = LeadRecord(
            session_id=session.id,
            contact=Contact(
                name=session.contact_name,
                email=session.contact_email,
                phone=session.contact_phone,
                company=session.contact_company,
            ),
            page_url=session.page_url,
            utm=dict(session.utm or {}),
            concern=summary.concern,
            use_case=summary.use_case,
            budget_stated=summary.budget,
            timeline=summary.timeline,
            services_of_interest=_services_of_interest(summary, snapshot),
            quoted_prices=[
                QuotedPrice.model_validate(quote)
                for quote in snapshot["quoted_prices"]
            ],
            unanswered_questions=list(snapshot["unanswered"]),
            chat_summary=summary.summary,
            suggested_next_step=summary.suggested_next_step,
            transcript_ref=session.id,
            message_count=len(history),
            handoff_reason=reason,
        )

        saved = await get_lead_service().capture(record)

        session.status = SessionStatus.CLOSED.value
        session.closed_at = _now()
        session.last_activity_at = _now()
        session.stage = Stage.HANDOFF.value
        session.handoff_reason = reason.value
        session.lead_id = saved.id
        session.lead_saved_at = _now()

        logger.info(
            "session closed",
            extra={
                "session_id": session.id,
                "reason": reason.value,
                "lead_id": saved.id,
                "lead_score": saved.lead_score,
            },
        )
        return saved


def _snapshot(session: ChatSession) -> dict[str, Any]:
    return {
        "slots": dict(session.slots or {}),
        "quoted_prices": list(session.quoted_prices or []),
        "unanswered": list(session.unanswered or []),
    }


def _services_of_interest(summary: Any, snapshot: dict[str, Any]) -> list[str]:
    """Prefer what we observed over what the summariser inferred.

    Quotes and the recorded slot are facts; the summariser's list is a guess
    that tends to re-list the same service once per tier. Only fall back to it
    when we observed nothing at all.
    """
    observed = {
        str(quote["service_name"])
        for quote in snapshot["quoted_prices"]
        if quote.get("service_name")
    }
    if snapshot["slots"].get("service_interest"):
        observed.add(snapshot["slots"]["service_interest"])

    if observed:
        return sorted(observed)
    return sorted({s for s in (summary.services_of_interest or []) if s})


def _dedupe_quotes(quotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same tier quoted twice is one quote, not two."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for quote in quotes:
        key = (str(quote.get("service_id")), str(quote.get("tier")))
        if key not in seen:
            seen.add(key)
            unique.append(quote)
    return unique


def _to_langchain(history: list[tuple[str, str]]) -> list[Any]:
    return [
        HumanMessage(content=content)
        if role == MessageRole.HUMAN
        else AIMessage(content=content)
        for role, content in history
        if content
    ]


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service

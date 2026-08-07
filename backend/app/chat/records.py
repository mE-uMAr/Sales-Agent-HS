"""Persisted conversation state.

These two tables are the single source of truth for a conversation. There is
deliberately no LangGraph checkpointer: the graph is stateless between turns and
rebuilds its message list from ``chat_messages`` each time. One store means one
thing to back up, one thing to reason about, and the transcript the lead record
needs is a by-product rather than a second copy.

Only human and assistant turns are persisted. Tool traffic from earlier turns is
not replayed to the model — the facts that matter were promoted into ``slots``,
and replaying stale tool output wastes tokens and invites the model to re-quote
figures out of context. A trace is kept on the assistant row for debugging.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class MessageRole(StrEnum):
    HUMAN = "human"
    ASSISTANT = "assistant"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    status: Mapped[str] = mapped_column(
        String(16), default=SessionStatus.ACTIVE, index=True
    )

    # Straight from the website form.
    contact_name: Mapped[str] = mapped_column(String(200))
    contact_email: Mapped[str | None] = mapped_column(String(320))
    contact_phone: Mapped[str | None] = mapped_column(String(64))
    contact_company: Mapped[str | None] = mapped_column(String(200))
    page_url: Mapped[str | None] = mapped_column(String(1000))
    utm: Mapped[dict] = mapped_column(JSON, default=dict)

    # Conversation working state.
    stage: Mapped[str] = mapped_column(String(24), default="greeting")
    slots: Mapped[dict] = mapped_column(JSON, default=dict)
    quoted_prices: Mapped[list] = mapped_column(JSON, default=list)
    unanswered: Mapped[list] = mapped_column(JSON, default=list)
    unanswered_streak: Mapped[int] = mapped_column(Integer, default=0)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)

    handoff_reason: Mapped[str | None] = mapped_column(String(32), default=None)
    lead_id: Mapped[str | None] = mapped_column(String(36), default=None)
    #: Set the moment a lead is written. Makes closing idempotent.
    lead_saved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    client_hash: Mapped[str | None] = mapped_column(String(32), default=None)

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="selectin",
    )

    @property
    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    #: Which tools ran to produce this reply. Debugging aid, never sent back.
    tool_trace: Mapped[list] = mapped_column(JSON, default=list)

    session: Mapped[ChatSession] = relationship(back_populates="messages")

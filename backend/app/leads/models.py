"""Lead payload (Pydantic) and its durable row (SQLAlchemy)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class HandoffReason(StrEnum):
    """Why the conversation ended. Drives how sales should triage the lead."""

    COMPLETED = "completed"  # bot gathered what it needed
    USER_REQUESTED = "user_requested"  # visitor asked for a human
    AGENT_ESCALATED = "agent_escalated"  # bot decided it was out of its depth
    KNOWLEDGE_GAP = "knowledge_gap"  # too many unanswerable questions
    MAX_TURNS = "max_turns"  # conversation length ceiling
    IDLE_TIMEOUT = "idle_timeout"  # visitor walked away
    ERROR = "error"  # unrecoverable failure


class DeliveryStatus(StrEnum):
    """State of forwarding this lead to an external system."""

    NOT_REQUIRED = "not_required"  # local database is the destination
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"  # attempts exhausted; needs a human


class Contact(BaseModel):
    """Identity, exactly as the website form supplied it."""

    name: str
    email: EmailStr | None = None
    phone: str | None = None
    company: str | None = None

    @property
    def first_name(self) -> str:
        """The only part of the contact that is ever sent to the LLM."""
        return self.name.strip().split(" ")[0] if self.name.strip() else "there"


class QuotedPrice(BaseModel):
    """A price the bot actually showed the visitor.

    Recorded verbatim from the pricing catalog so sales knows precisely what was
    promised, and so a quote can never be reconstructed from model output.
    """

    service_id: str
    service_name: str
    tier: str
    currency: str
    market_price: float
    our_price: float
    discount_pct: float
    quoted_at: datetime = Field(default_factory=_now)


class LeadRecord(BaseModel):
    """Everything sales needs to pick the conversation up warm."""

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    captured_at: datetime = Field(default_factory=_now)

    contact: Contact
    page_url: str | None = None
    utm: dict[str, str] = Field(default_factory=dict)

    concern: str = ""
    use_case: str | None = None
    budget_stated: str | None = None
    timeline: str | None = None
    services_of_interest: list[str] = Field(default_factory=list)
    quoted_prices: list[QuotedPrice] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)

    chat_summary: str = ""
    suggested_next_step: str | None = None

    #: Points at ``chat_messages.session_id`` — the full transcript lives there.
    transcript_ref: str = ""
    message_count: int = 0

    handoff_reason: HandoffReason = HandoffReason.COMPLETED
    lead_score: int = 0

    def model_post_init(self, _context: object) -> None:
        if not self.transcript_ref:
            self.transcript_ref = self.session_id


class Lead(Base):
    """Durable lead row. Doubles as the outbox when forwarding is enabled."""

    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )

    # Denormalised for querying and CSV export without unpacking the payload.
    contact_name: Mapped[str] = mapped_column(String(200))
    contact_email: Mapped[str | None] = mapped_column(String(320), index=True)
    contact_company: Mapped[str | None] = mapped_column(String(200))
    handoff_reason: Mapped[str] = mapped_column(String(32), index=True)
    lead_score: Mapped[int] = mapped_column(Integer, default=0, index=True)

    #: The full :class:`LeadRecord`, JSON-serialised. Source of truth.
    payload: Mapped[dict] = mapped_column(JSON)

    delivery_status: Mapped[str] = mapped_column(
        String(16), default=DeliveryStatus.NOT_REQUIRED, index=True
    )
    delivery_attempts: Mapped[int] = mapped_column(Integer, default=0)
    delivery_next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, index=True
    )
    delivery_last_error: Mapped[str | None] = mapped_column(Text, default=None)
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    def to_record(self) -> LeadRecord:
        return LeadRecord.model_validate(self.payload)

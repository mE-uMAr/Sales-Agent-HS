"""Request and response bodies.

These are the contract the website integrates against, so field names and
optionality here are a public interface — change them deliberately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class StartSessionIn(BaseModel):
    """The payload the old contact form used to submit."""

    name: str = Field(min_length=1, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(default=None, max_length=64)
    company: str | None = Field(default=None, max_length=200)
    page_url: str | None = Field(default=None, max_length=1000)
    utm: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("name cannot be blank")
        return cleaned

    @field_validator("utm")
    @classmethod
    def _bound_utm(cls, value: dict[str, str]) -> dict[str, str]:
        return {str(k)[:40]: str(v)[:200] for k, v in list(value.items())[:12]}


class StartSessionOut(BaseModel):
    session_id: str
    #: Send as `Authorization: Bearer <token>` on every subsequent call.
    token: str
    expires_in: int
    message: str
    stage: str


class MessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class MessageOut(BaseModel):
    session_id: str
    reply: str
    stage: str
    closed: bool
    handoff_reason: str | None = None


class CloseSessionOut(BaseModel):
    session_id: str
    closed: bool
    lead_captured: bool
    message: str


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None


# ── admin ────────────────────────────────────────────────────────────
class LeadOut(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    contact_name: str
    contact_email: str | None
    contact_company: str | None
    lead_score: int
    handoff_reason: str
    delivery_status: str
    concern: str
    budget_stated: str | None
    chat_summary: str
    unanswered_questions: list[str]
    quoted_prices: list[dict[str, Any]]


class HealthOut(BaseModel):
    status: str
    version: str
    knowledge_base_ready: bool
    llm_provider: str
    lead_sink: str
    pending_deliveries: int


class ReindexOut(BaseModel):
    files: int
    chunks: int
    skipped: list[str]

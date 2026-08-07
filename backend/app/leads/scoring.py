"""Deterministic lead scoring.

Kept out of the LLM's hands on purpose: sales needs a number that means the same
thing every time and can be explained line by line when someone asks why a lead
was ranked the way it was.
"""

from __future__ import annotations

from app.leads.models import HandoffReason, LeadRecord

#: Consumer mailbox providers — a business domain is a mild buying signal.
FREE_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "aol.com",
        "icloud.com",
        "me.com",
        "proton.me",
        "protonmail.com",
        "mail.com",
        "gmx.com",
        "yandex.com",
        "zoho.com",
    }
)

_SCORE_CAP = 100


def _has_business_email(record: LeadRecord) -> bool:
    email = record.contact.email
    if not email or "@" not in email:
        return False
    return email.rsplit("@", 1)[1].lower() not in FREE_EMAIL_DOMAINS


def explain_score(record: LeadRecord) -> dict[str, int]:
    """Per-signal contributions. The sum (clamped to 0..100) is the score."""
    breakdown: dict[str, int] = {}

    if record.budget_stated:
        breakdown["budget_stated"] = 30
    if record.use_case:
        breakdown["use_case_captured"] = 25
    if record.quoted_prices:
        breakdown["saw_a_quote"] = 15
    if record.timeline:
        breakdown["timeline_given"] = 10
    if record.services_of_interest:
        breakdown["named_a_service"] = 5
    if record.message_count >= 6:
        breakdown["engaged_conversation"] = 10
    if _has_business_email(record):
        breakdown["business_email"] = 10

    # A visitor who walked away mid-sentence is worth less than one who finished,
    # regardless of what they had already told us.
    if record.handoff_reason is HandoffReason.IDLE_TIMEOUT:
        breakdown["abandoned"] = -20
    elif record.handoff_reason is HandoffReason.ERROR:
        breakdown["ended_in_error"] = -10

    return breakdown


def score_lead(record: LeadRecord) -> int:
    total = sum(explain_score(record).values())
    return max(0, min(_SCORE_CAP, total))

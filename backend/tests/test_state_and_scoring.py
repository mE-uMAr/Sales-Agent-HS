"""The slot-driven stage machine and the deterministic lead score."""

from __future__ import annotations

from app.chat.state import Stage, compute_stage
from app.chat.tools import select_tools
from app.leads.models import Contact, HandoffReason, LeadRecord
from app.leads.scoring import explain_score, score_lead


# ── stage machine ────────────────────────────────────────────────────
def test_stage_follows_slots_not_the_model() -> None:
    assert compute_stage({}, turn_count=0) is Stage.GREETING
    assert compute_stage({}, turn_count=1) is Stage.DISCOVERY
    assert compute_stage({"use_case": "a portal"}, turn_count=2) is Stage.QUALIFY
    assert (
        compute_stage({"use_case": "a portal", "budget": "20k"}, turn_count=3)
        is Stage.WRAP_UP
    )


def test_budget_without_use_case_still_asks_discovery_first() -> None:
    """Order matters: understand the need before qualifying the spend."""
    assert compute_stage({"budget": "20k"}, turn_count=2) is Stage.DISCOVERY


def test_termination_overrides_every_other_stage() -> None:
    assert (
        compute_stage({"use_case": "x"}, turn_count=5, terminating=True)
        is Stage.HANDOFF
    )


# ── tool gating ──────────────────────────────────────────────────────
def test_pricing_tool_is_withheld_early() -> None:
    """Availability enforces what instruction only suggests."""
    names = [t.name for t in select_tools("discovery", "we need a client portal")]
    assert "lookup_pricing" not in names
    assert "search_company_knowledge" in names


def test_pricing_tool_appears_when_money_is_mentioned() -> None:
    names = [t.name for t in select_tools("discovery", "roughly what does that cost?")]
    assert "lookup_pricing" in names


def test_pricing_tool_always_available_after_discovery() -> None:
    names = [t.name for t in select_tools("qualify", "we need it by June")]
    assert "lookup_pricing" in names


# ── scoring ──────────────────────────────────────────────────────────
def _record(**kwargs) -> LeadRecord:
    payload = {
        "session_id": "s1",
        "contact": Contact(name="Dana Reyes", email="dana@brightpath.io"),
    }
    payload.update(kwargs)
    return LeadRecord(**payload)


def test_score_is_the_sum_of_its_explanation() -> None:
    record = _record(use_case="a portal", budget_stated="20k", message_count=8)
    assert score_lead(record) == sum(explain_score(record).values())


def test_budget_is_the_strongest_signal() -> None:
    breakdown = explain_score(_record(budget_stated="20k"))
    assert breakdown["budget_stated"] == 30


def test_business_email_scores_above_a_free_one() -> None:
    business = _record(contact=Contact(name="Dana", email="dana@brightpath.io"))
    personal = _record(contact=Contact(name="Dana", email="dana@gmail.com"))
    assert score_lead(business) > score_lead(personal)


def test_abandoned_conversation_is_penalised() -> None:
    engaged = _record(use_case="a portal", budget_stated="20k")
    abandoned = _record(
        use_case="a portal",
        budget_stated="20k",
        handoff_reason=HandoffReason.IDLE_TIMEOUT,
    )
    assert score_lead(abandoned) < score_lead(engaged)


def test_score_is_clamped_to_a_hundred() -> None:
    record = _record(
        use_case="a portal",
        budget_stated="20k",
        timeline="June",
        services_of_interest=["web_app"],
        message_count=20,
        quoted_prices=[],
    )
    assert 0 <= score_lead(record) <= 100


def test_empty_conversation_scores_low() -> None:
    assert score_lead(_record()) <= 10

"""Golden transcripts.

Whole conversations driven through the real graph, real tools and real
persistence — only the model is scripted. Each test asserts the two things that
actually matter: the visitor got the right words, and sales got the right lead.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.chat.copy import HANDOFF_LINE, UNKNOWN_ANSWER
from app.chat.llm import set_chat_model_override
from app.chat.service import ChatService, SessionClosedError, StartSessionRequest
from app.chat.summarizer import ConversationSummary
from app.leads import repository
from app.leads.models import HandoffReason
from app.persistence.db import session_scope
from tests.conftest import FlakyChatModel, ScriptedChatModel, tool_call_message

SUMMARY = ConversationSummary(
    concern="Needs a client portal",
    summary="Dana wants a customer portal and has a 20-30k budget.",
    use_case="client portal with order history",
    budget="20-30k",
    suggested_next_step="Book a scoping call.",
)


def script(*responses: AIMessage, summary: ConversationSummary = SUMMARY) -> ScriptedChatModel:
    """Install a scripted conversation model and summariser."""
    chat = ScriptedChatModel(responses=list(responses), calls=[])
    set_chat_model_override(chat)
    set_chat_model_override(
        ScriptedChatModel(
            responses=[AIMessage(content="")], calls=[], structured_response=summary
        ),
        summary=True,
    )
    return chat


async def start(service: ChatService) -> str:
    session_id, _ = await service.start_session(
        StartSessionRequest(
            name="Dana Reyes",
            email="dana@brightpath.io",
            company="Brightpath",
            page_url="https://example.com/contact",
        )
    )
    return session_id


async def stored_lead(session_id: str):
    async with session_scope() as db:
        return await repository.get_by_session_id(db, session_id)


# ── the happy path ───────────────────────────────────────────────────
async def test_full_qualification_produces_a_complete_lead(database, no_retrieval) -> None:
    script(
        tool_call_message("record_detail", {"field": "use_case", "value": "client portal"}),
        AIMessage(content="Understood. What budget range did you have in mind?"),
        tool_call_message("record_detail", {"field": "budget", "value": "20-30k"}),
        AIMessage(content="That works well. Someone will follow up."),
    )
    service = ChatService()
    session_id = await start(service)

    first = await service.send_message(session_id, "We need a customer portal")
    assert first.stage == "qualify", "asking for budget is the next objective"
    assert not first.closed

    second = await service.send_message(session_id, "Around 20 to 30 thousand")
    assert second.stage == "wrap_up"

    await service.close_session(session_id, HandoffReason.COMPLETED)

    lead = await stored_lead(session_id)
    assert lead is not None
    record = lead.to_record()
    assert record.use_case == "client portal"
    assert record.budget_stated == "20-30k"
    assert record.contact.email == "dana@brightpath.io"
    assert record.contact.company == "Brightpath"
    assert record.page_url == "https://example.com/contact"
    assert record.handoff_reason is HandoffReason.COMPLETED
    assert record.lead_score >= 55
    assert record.transcript_ref == session_id


# ── pricing ──────────────────────────────────────────────────────────
async def test_quoted_prices_are_recorded_verbatim(database, no_retrieval, catalog) -> None:
    script(
        tool_call_message("lookup_pricing", {"service": "web_app", "tier": "growth"}),
        AIMessage(content="The Growth tier is $26,500 against a market price of $38,000."),
    )
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "How much for a portal?")
    assert "26,500" in result.reply

    await service.close_session(session_id)
    record = (await stored_lead(session_id)).to_record()

    assert len(record.quoted_prices) == 1
    quote = record.quoted_prices[0]
    expected = catalog.quote("web_app", "growth")[0]
    assert quote.our_price == expected.our_price
    assert quote.market_price == expected.market_price
    assert record.services_of_interest == ["Custom Web Application"]


async def test_an_invented_price_never_reaches_the_visitor(database, no_retrieval) -> None:
    script(AIMessage(content="It'll be roughly $17,250 for that."))
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "Ballpark cost?")
    assert "17,250" not in result.reply
    assert "confirm the pricing" in result.reply


# ── not knowing ──────────────────────────────────────────────────────
async def test_unanswered_question_is_disclosed_and_recorded(
    database, no_retrieval
) -> None:
    script(
        tool_call_message("flag_unanswered", {"question": "Who founded the company?"}),
        AIMessage(content=UNKNOWN_ANSWER),
    )
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "Who founded the company?")
    assert "don't know" in result.reply.lower()

    await service.close_session(session_id)
    record = (await stored_lead(session_id)).to_record()
    assert record.unanswered_questions == ["Who founded the company?"]


async def test_honest_wording_is_appended_if_the_model_drifts(
    database, no_retrieval
) -> None:
    """The disclosure is ours, not the model's, so it cannot be paraphrased away."""
    script(
        tool_call_message("flag_unanswered", {"question": "What is your revenue?"}),
        AIMessage(content="Our revenue is about ten million."),
    )
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "What is your revenue?")
    assert UNKNOWN_ANSWER in result.reply


async def test_two_unanswered_in_a_row_hands_off(database, no_retrieval) -> None:
    script(
        tool_call_message("flag_unanswered", {"question": "Q1"}, "c1"),
        AIMessage(content=UNKNOWN_ANSWER),
        tool_call_message("flag_unanswered", {"question": "Q2"}, "c2"),
        AIMessage(content=UNKNOWN_ANSWER),
    )
    service = ChatService()
    session_id = await start(service)

    await service.send_message(session_id, "Q1")
    second = await service.send_message(session_id, "Q2")

    assert second.closed
    assert second.handoff_reason == HandoffReason.KNOWLEDGE_GAP.value
    record = (await stored_lead(session_id)).to_record()
    assert record.unanswered_questions == ["Q1", "Q2"]


async def test_the_streak_resets_after_a_good_answer(database, no_retrieval) -> None:
    """Two flags across a long useful chat must not end it."""
    script(
        tool_call_message("flag_unanswered", {"question": "Q1"}, "c1"),
        AIMessage(content=UNKNOWN_ANSWER),
        AIMessage(content="Yes, we work with in-house teams all the time."),
        tool_call_message("flag_unanswered", {"question": "Q2"}, "c2"),
        AIMessage(content=UNKNOWN_ANSWER),
    )
    service = ChatService()
    session_id = await start(service)

    await service.send_message(session_id, "Q1")
    middle = await service.send_message(session_id, "Do you work with our team?")
    assert not middle.closed

    third = await service.send_message(session_id, "Q2")
    assert not third.closed, "streak should have reset on the answered turn"


# ── endings ──────────────────────────────────────────────────────────
async def test_escalation_closes_and_is_attributed_to_the_visitor(
    database, no_retrieval
) -> None:
    script(
        tool_call_message(
            "escalate_to_human", {"reason": "visitor asked to speak to someone"}
        ),
        AIMessage(content="Of course, I'll pass you to the team."),
    )
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "Can I talk to a person?")

    assert result.closed
    assert result.handoff_reason == HandoffReason.USER_REQUESTED.value
    assert HANDOFF_LINE.lower() in result.reply.lower()


async def test_turn_ceiling_closes_the_conversation(
    database, no_retrieval, monkeypatch
) -> None:
    monkeypatch.setenv("MAX_TURNS", "2")
    from app.config import get_settings

    get_settings.cache_clear()

    script(AIMessage(content="Tell me more."))
    service = ChatService()
    session_id = await start(service)

    await service.send_message(session_id, "one")
    second = await service.send_message(session_id, "two")

    assert second.closed
    assert second.handoff_reason == HandoffReason.MAX_TURNS.value

    with pytest.raises(SessionClosedError):
        await service.send_message(session_id, "three")


async def test_rate_limited_model_falls_back_instead_of_ending(
    database, no_retrieval
) -> None:
    """A daily quota on the main model must not end someone's conversation."""
    flaky = FlakyChatModel(
        responses=[AIMessage(content="Happy to help with that.")],
        calls=[],
        failures_remaining=1,
    )
    set_chat_model_override(flaky)
    set_chat_model_override(
        ScriptedChatModel(
            responses=[AIMessage(content="")], calls=[], structured_response=SUMMARY
        ),
        summary=True,
    )

    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "Tell me about your work")

    assert not result.closed
    assert result.reply == "Happy to help with that."
    assert flaky.failures_remaining == 0, "the fallback should have been used"


async def test_model_failure_still_captures_the_lead(database, no_retrieval) -> None:
    """The worst case still has to produce a record someone can act on."""
    script()  # no scripted responses -> the model raises
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "Hello?")

    assert result.closed
    assert result.handoff_reason == HandoffReason.ERROR.value
    assert HANDOFF_LINE.lower() in result.reply.lower()

    lead = await stored_lead(session_id)
    assert lead is not None, "a failed conversation must still leave a lead"
    assert lead.to_record().contact.email == "dana@brightpath.io"


async def test_idle_sweeper_closes_abandoned_conversations(
    database, no_retrieval, monkeypatch
) -> None:
    monkeypatch.setenv("SESSION_IDLE_MINUTES", "0")
    from app.config import get_settings

    get_settings.cache_clear()

    script(AIMessage(content="Sure."))
    service = ChatService()
    session_id = await start(service)
    await service.send_message(session_id, "hi")

    closed = await service.sweep_idle_sessions()

    assert closed == 1
    lead = await stored_lead(session_id)
    assert lead.to_record().handoff_reason is HandoffReason.IDLE_TIMEOUT


async def test_closing_twice_produces_one_lead(database, no_retrieval) -> None:
    script(AIMessage(content="Sure."))
    service = ChatService()
    session_id = await start(service)
    await service.send_message(session_id, "hi")

    first = await service.close_session(session_id)
    second = await service.close_session(session_id)

    assert first is not None
    assert second is None, "the second close must be a no-op"

    async with session_scope() as db:
        assert len(await repository.list_leads(db, limit=50)) == 1


# ── output hygiene ───────────────────────────────────────────────────
async def test_leaked_tool_syntax_never_reaches_the_visitor(
    database, no_retrieval
) -> None:
    script(
        AIMessage(
            content='Sure thing. <function=lookup_pricing>{"service": "web_app"}</function>'
        )
    )
    service = ChatService()
    session_id = await start(service)

    result = await service.send_message(session_id, "What do you charge?")

    assert "<function" not in result.reply
    assert "lookup_pricing" not in result.reply

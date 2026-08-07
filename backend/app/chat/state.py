"""Conversation state and the slot-driven stage machine.

The bot has a job to do — understand the need, then ask about budget — and a
free-running agent drifts away from it. So the stage is **derived** from which
facts have actually been captured, never from the model's own opinion of where
the conversation is. Filling a slot is an explicit tool call, which makes every
transition loggable, replayable and testable without an LLM in the loop.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

#: Facts the conversation is trying to collect, in priority order.
SlotName = Literal["use_case", "budget", "timeline", "service_interest"]
SLOT_NAMES: tuple[str, ...] = ("use_case", "budget", "timeline", "service_interest")


class Stage(StrEnum):
    GREETING = "greeting"
    DISCOVERY = "discovery"
    QUALIFY = "qualify"
    WRAP_UP = "wrap_up"
    HANDOFF = "handoff"


#: What the model is told to aim for this turn. Injected into the system prompt.
STAGE_OBJECTIVES: dict[Stage, str] = {
    Stage.GREETING: (
        "Greet them by first name, say what you can help with in one sentence, "
        "and ask what they are looking to build or solve."
    ),
    Stage.DISCOVERY: (
        "Understand the project. Ask one focused follow-up at a time until you "
        "can describe what they need in a sentence or two, then call "
        "record_detail with field='use_case'. Do not ask about budget yet, and "
        "do not bring up pricing unless they ask about it first."
    ),
    Stage.QUALIFY: (
        "You understand the project. Now ask what budget range they have in "
        "mind, framed as helping you point them at the right option rather "
        "than as a gate. When they answer — even roughly, even 'not sure' — "
        "call record_detail with field='budget'."
    ),
    Stage.WRAP_UP: (
        "You have the need and the budget. Briefly confirm what you have "
        "understood, answer anything still open, and tell them a member of the "
        "team will follow up. Do not keep asking questions."
    ),
    Stage.HANDOFF: (
        "The conversation is ending. Hand over to a human warmly and stop "
        "asking questions."
    ),
}


class ConversationState(TypedDict, total=False):
    """LangGraph state for a single turn."""

    messages: Annotated[list[AnyMessage], add_messages]

    session_id: str
    first_name: str

    stage: str
    slots: dict[str, str]
    quoted_prices: list[dict[str, Any]]
    unanswered: list[str]
    unanswered_streak: int
    turn_count: int

    #: Tool calls made this turn, for the debugging trace on the reply.
    tool_trace: list[dict[str, Any]]

    terminate: bool
    handoff_reason: str | None
    tool_iterations: int


def compute_stage(
    slots: dict[str, str], *, turn_count: int, terminating: bool = False
) -> Stage:
    """Where the conversation is, purely as a function of what we know."""
    if terminating:
        return Stage.HANDOFF
    if turn_count == 0:
        return Stage.GREETING
    if not slots.get("use_case"):
        return Stage.DISCOVERY
    if not slots.get("budget"):
        return Stage.QUALIFY
    return Stage.WRAP_UP


def new_state(
    *,
    session_id: str,
    first_name: str,
    stage: str = Stage.GREETING,
    slots: dict[str, str] | None = None,
    quoted_prices: list[dict[str, Any]] | None = None,
    unanswered: list[str] | None = None,
    unanswered_streak: int = 0,
    turn_count: int = 0,
    messages: list[AnyMessage] | None = None,
) -> ConversationState:
    return ConversationState(
        messages=messages or [],
        session_id=session_id,
        first_name=first_name,
        stage=stage,
        slots=dict(slots or {}),
        quoted_prices=list(quoted_prices or []),
        unanswered=list(unanswered or []),
        unanswered_streak=unanswered_streak,
        turn_count=turn_count,
        tool_trace=[],
        terminate=False,
        handoff_reason=None,
        tool_iterations=0,
    )

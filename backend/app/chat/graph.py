"""The per-turn conversation graph.

One invocation of this graph turns one visitor message into one assistant reply:

    agent ─┬─(tool calls)─▶ tools ──▶ effects ──▶ agent
           └─(no calls)───▶ respond ──▶ END

State is not checkpointed here. Each turn is rebuilt from the database, which
keeps the graph a pure function of (history, slots, message) and makes a turn
reproducible in a test without any persistence at all.

Tools never mutate state. They return JSON carrying an ``effect``, and
:func:`apply_effects` is the single place where a tool call is allowed to change
the conversation — so every slot fill, quote and escalation is visible in one
readable function.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.chat.copy import (
    ERROR_CLOSING,
    ESCALATION_ACK,
    GLITCHED_TURN,
    HANDOFF_LINE,
    KNOWLEDGE_GAP_CLOSING,
    UNKNOWN_ANSWER,
)
from app.chat.guards import check_output, strip_tool_artifacts
from app.chat.llm import build_chat_model, chat_breaker
from app.chat.prompts import build_system_prompt
from app.chat.state import ConversationState, compute_stage
from app.chat.tools import ALL_TOOLS, select_tools
from app.config import get_settings
from app.leads.models import HandoffReason
from app.observability import get_logger

logger = get_logger(__name__)

#: Phrases in an escalation reason that mean the visitor asked, not the bot.
_USER_ASKED = ("asked", "requested", "wants to speak", "speak to", "talk to a")


def _is_tool_format_failure(exc: Exception) -> bool:
    """Did the model emit a malformed tool call rather than the provider fail?

    Smaller open models sometimes write ``<function=name {...}>`` into the
    content instead of producing a proper tool call, and Groq rejects the
    completion with ``400 tool_use_failed``. That is a generation quirk, not an
    outage: retrying the same request fails identically, the circuit breaker
    must not count it, and the conversation should survive it.
    """
    text = str(exc)
    return "tool_use_failed" in text or "Failed to call a function" in text


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text or "too many requests" in text


def _looks_answered(reply: str) -> bool:
    lowered = reply.lower()
    return any(
        phrase in lowered
        for phrase in ("don't know", "do not know", "not sure", "can't answer", "cannot answer")
    )


# ──────────────────────────────────────────────────────────────────────
# nodes
# ──────────────────────────────────────────────────────────────────────
def agent_node(state: ConversationState) -> dict[str, Any]:
    """Ask the model what to say or which tool to use."""
    if chat_breaker.is_open:
        logger.error("llm circuit open; handing off without calling the model")
        return {
            "terminate": True,
            "handoff_reason": HandoffReason.ERROR.value,
            "messages": [AIMessage(content=ERROR_CLOSING)],
        }

    base = build_chat_model()
    system = SystemMessage(content=build_system_prompt(state))
    conversation = [system, *state["messages"]]

    latest_visitor_message = next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ),
        "",
    )
    tools = select_tools(state.get("stage", ""), latest_visitor_message)

    try:
        reply = base.bind_tools(tools).invoke(conversation)
    except Exception as exc:
        if _is_tool_format_failure(exc):
            # Recoverable: ask again with no tools bound so the model is forced
            # to answer in prose. It loses tool access for this turn only, which
            # is far better than ending the conversation.
            logger.warning(
                "model produced a malformed tool call; retrying without tools",
                extra={"error": str(exc)[:300]},
            )
            try:
                reply = base.invoke(conversation)
            except Exception:
                chat_breaker.record_failure()
                logger.exception("retry without tools also failed")
                return _error_turn()
            chat_breaker.record_success()
            return {
                "messages": [reply],
                "tool_iterations": state.get("tool_iterations", 0) + 1,
            }

        fallback_model = get_settings().llm_fallback_model
        if _is_rate_limited(exc) and fallback_model:
            # Quotas are metered per model, so the smaller one usually still has
            # headroom. A slightly less capable answer beats ending someone's
            # conversation because the flagship model ran out of tokens today.
            logger.warning(
                "main model rate limited; falling back",
                extra={"fallback_model": fallback_model},
            )
            try:
                reply = build_chat_model(model_name=fallback_model).bind_tools(
                    tools
                ).invoke(conversation)
            except Exception:
                chat_breaker.record_failure()
                logger.exception("fallback model also failed")
                return _error_turn()
            chat_breaker.record_success()
            return {
                "messages": [reply],
                "tool_iterations": state.get("tool_iterations", 0) + 1,
            }

        chat_breaker.record_failure()
        logger.exception("model invocation failed")
        return _error_turn()

    chat_breaker.record_success()
    return {
        "messages": [reply],
        "tool_iterations": state.get("tool_iterations", 0) + 1,
    }


def _error_turn() -> dict[str, Any]:
    return {
        "terminate": True,
        "handoff_reason": HandoffReason.ERROR.value,
        "messages": [AIMessage(content=ERROR_CLOSING)],
    }


def apply_effects(state: ConversationState) -> dict[str, Any]:
    """Fold this turn's tool results into conversation state.

    The only place tool output is allowed to change anything.
    """
    settings = get_settings()

    slots = dict(state.get("slots", {}))
    quoted = list(state.get("quoted_prices", []))
    unanswered = list(state.get("unanswered", []))
    streak = state.get("unanswered_streak", 0)
    trace = list(state.get("tool_trace", []))
    terminate = state.get("terminate", False)
    handoff_reason = state.get("handoff_reason")

    # Walk back to the most recent AI turn; everything after it is this turn's
    # tool traffic.
    fresh: list[ToolMessage] = []
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            fresh.append(message)
        elif isinstance(message, AIMessage):
            break
    fresh.reverse()

    flagged_this_turn = False

    for message in fresh:
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("unparseable tool result", extra={"tool": message.name})
            continue

        trace.append({"tool": message.name, "ok": payload.get("ok", False)})
        effect = payload.get("effect")
        if not isinstance(effect, dict):
            continue

        match effect.get("type"):
            case "record_detail":
                slots[effect["field"]] = effect["value"]
                logger.info(
                    "slot recorded",
                    extra={"field": effect["field"], "value": effect["value"][:120]},
                )

            case "quoted_prices":
                quoted.extend(effect.get("quotes", []))

            case "unanswered":
                question = effect["question"]
                if question not in unanswered:
                    unanswered.append(question)
                flagged_this_turn = True

            case "escalate":
                reason = effect.get("reason", "")
                terminate = True
                handoff_reason = (
                    HandoffReason.USER_REQUESTED.value
                    if any(phrase in reason.lower() for phrase in _USER_ASKED)
                    else HandoffReason.AGENT_ESCALATED.value
                )
                logger.info("escalation recorded", extra={"reason": reason[:200]})

    # A streak, not a total: two flags in a row means the bot is out of its
    # depth here and should stop, while two across a long useful chat should not
    # end it.
    streak = streak + 1 if flagged_this_turn else 0
    if not terminate and streak >= settings.max_unanswered_streak:
        terminate = True
        handoff_reason = HandoffReason.KNOWLEDGE_GAP.value
        logger.info("terminating on knowledge gap", extra={"streak": streak})

    return {
        "slots": slots,
        "quoted_prices": quoted,
        "unanswered": unanswered,
        "unanswered_streak": streak,
        "tool_trace": trace,
        "terminate": terminate,
        "handoff_reason": handoff_reason,
        "stage": compute_stage(
            slots, turn_count=state.get("turn_count", 0), terminating=terminate
        ).value,
    }


def respond_node(state: ConversationState) -> dict[str, Any]:
    """Verify the reply, then attach any closing copy the ending calls for."""
    last = state["messages"][-1] if state["messages"] else None
    reply = last.content if isinstance(last, AIMessage) else ""
    reply = reply.strip() if isinstance(reply, str) else ""

    # The model sometimes writes a tool call into its own text. Strip it before
    # a visitor ever sees it; if that was most of the reply, the lookup never
    # ran and the honest move is to ask again rather than ship the remains.
    reply, had_artifact = strip_tool_artifacts(reply)
    if had_artifact:
        logger.warning(
            "stripped a malformed tool call from the reply",
            extra={"remaining_chars": len(reply)},
        )
        if len(reply) < 40:
            reply = GLITCHED_TURN

    # Figures the *visitor* used are legitimate for the bot to repeat back.
    # Only their turns count: including the assistant's own text here would let
    # every invented price whitelist itself.
    visitor_text = " ".join(
        message.content
        for message in state.get("messages", [])
        if isinstance(message, HumanMessage) and isinstance(message.content, str)
    )

    verdict = check_output(reply, conversation_text=visitor_text)
    text = verdict.text

    # The model was told to use the honest wording verbatim. If it drifted,
    # make sure the visitor is still told plainly that we do not know.
    if state.get("unanswered_streak", 0) > 0 and not _looks_answered(text):
        text = f"{text}\n\n{UNKNOWN_ANSWER}".strip() if text else UNKNOWN_ANSWER

    if state.get("terminate"):
        reason = state.get("handoff_reason")
        if reason == HandoffReason.ERROR.value:
            text = ERROR_CLOSING
        elif reason == HandoffReason.KNOWLEDGE_GAP.value:
            text = f"{text}\n\n{KNOWLEDGE_GAP_CLOSING}".strip()
        elif not text:
            text = ESCALATION_ACK
        elif HANDOFF_LINE.lower() not in text.lower():
            text = f"{text}\n\n{HANDOFF_LINE}".strip()

    if not text:
        text = UNKNOWN_ANSWER

    # The streak must be settled on every turn, not only on turns that used a
    # tool — otherwise a turn the bot answered well leaves the previous flag
    # standing, and two flags either side of a good answer look consecutive.
    flagged_this_turn = any(
        entry.get("tool") == "flag_unanswered"
        for entry in state.get("tool_trace", [])
    )

    return {
        "messages": [AIMessage(content=text)],
        "guard_reason": verdict.reason,
        "unanswered_streak": (
            state.get("unanswered_streak", 0) if flagged_this_turn else 0
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# routing
# ──────────────────────────────────────────────────────────────────────
def route_after_agent(state: ConversationState) -> str:
    if state.get("terminate") and state.get("handoff_reason") == HandoffReason.ERROR.value:
        return "respond"

    last = state["messages"][-1] if state["messages"] else None
    calls = getattr(last, "tool_calls", None) if isinstance(last, AIMessage) else None
    if not calls:
        return "respond"

    if state.get("tool_iterations", 0) >= get_settings().max_tool_iterations:
        logger.warning(
            "tool iteration ceiling reached; answering with what we have",
            extra={"iterations": state.get("tool_iterations")},
        )
        return "respond"

    return "tools"


def route_after_effects(state: ConversationState) -> str:
    # Even when terminating, go back to the model once so the handoff is worded
    # in context rather than pasted in cold.
    if state.get("tool_iterations", 0) >= get_settings().max_tool_iterations:
        return "respond"
    return "agent"


def build_graph() -> Any:
    graph = StateGraph(ConversationState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("effects", apply_effects)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "respond": "respond"}
    )
    graph.add_edge("tools", "effects")
    graph.add_conditional_edges(
        "effects", route_after_effects, {"agent": "agent", "respond": "respond"}
    )
    graph.add_edge("respond", END)

    return graph.compile()


_graph: Any | None = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

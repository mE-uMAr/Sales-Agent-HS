"""Turn a finished conversation into a handover note.

Run as a separate call on the cheaper model, not by the conversational agent.
That separation is the point: if the chat went sideways — the visitor was
hostile, the model rambled, the circuit breaker tripped — the summariser still
sees a clean transcript and produces a usable record.

If the model call fails outright, :func:`fallback_summary` builds the note from
captured slots alone. A lead is never lost because summarisation failed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.chat.llm import build_chat_model
from app.chat.prompts import build_summary_prompt, format_known_facts
from app.observability import get_logger

logger = get_logger(__name__)

MAX_TRANSCRIPT_CHARS = 12_000


class ConversationSummary(BaseModel):
    """Structured handover note. Every field is optional by design — a blank
    field is honest, an invented one wastes a salesperson's phone call."""

    concern: str = Field(
        default="",
        description="One sentence: the problem the visitor came here with.",
    )
    summary: str = Field(
        default="",
        description=(
            "Two to four sentences briefing a colleague about to phone this "
            "person. Third person, plain prose."
        ),
    )
    use_case: str | None = Field(
        default=None, description="What they want built, if stated."
    )
    budget: str | None = Field(
        default=None, description="Budget as they expressed it, if stated."
    )
    timeline: str | None = Field(
        default=None, description="When they need it, if stated."
    )
    services_of_interest: list[str] = Field(
        default_factory=list, description="Service names or ids discussed."
    )
    suggested_next_step: str | None = Field(
        default=None,
        description="The single most useful next action for the sales team.",
    )


def render_transcript(messages: list[tuple[str, str]]) -> str:
    """``[(role, content), ...]`` -> a plain, bounded transcript."""
    lines = [
        f"{'Visitor' if role == 'human' else 'Assistant'}: {content.strip()}"
        for role, content in messages
        if content and content.strip()
    ]
    transcript = "\n\n".join(lines)

    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        # Keep the ends: the opening states the need, the close states intent.
        head = transcript[: MAX_TRANSCRIPT_CHARS // 2]
        tail = transcript[-MAX_TRANSCRIPT_CHARS // 2 :]
        transcript = f"{head}\n\n[… middle of conversation omitted …]\n\n{tail}"

    return transcript


def fallback_summary(
    slots: dict[str, str], messages: list[tuple[str, str]]
) -> ConversationSummary:
    """Deterministic note used when the model is unavailable."""
    first_visitor_line = next(
        (content.strip() for role, content in messages if role == "human"), ""
    )
    concern = first_visitor_line[:280]

    parts: list[str] = []
    if slots.get("use_case"):
        parts.append(f"They are looking for: {slots['use_case']}.")
    if slots.get("budget"):
        parts.append(f"Budget mentioned: {slots['budget']}.")
    if slots.get("timeline"):
        parts.append(f"Timeline: {slots['timeline']}.")
    if not parts and concern:
        parts.append(f"Opened the conversation with: {concern}")

    return ConversationSummary(
        concern=concern,
        summary=(
            " ".join(parts)
            or "Conversation ended before any details were captured."
        )
        + " (Automatic summary — the summarisation model was unavailable.)",
        use_case=slots.get("use_case"),
        budget=slots.get("budget"),
        timeline=slots.get("timeline"),
        services_of_interest=(
            [slots["service_interest"]] if slots.get("service_interest") else []
        ),
    )


async def summarise_conversation(
    messages: list[tuple[str, str]],
    slots: dict[str, str],
    quoted_prices: list[dict] | None = None,
    unanswered: list[str] | None = None,
) -> ConversationSummary:
    if not messages:
        return ConversationSummary(summary="No conversation took place.")

    prompt = build_summary_prompt(
        transcript=render_transcript(messages),
        captured_facts=format_known_facts(slots, quoted_prices, unanswered),
    )

    try:
        model = build_chat_model(summary=True).with_structured_output(
            ConversationSummary
        )
        result = await model.ainvoke(prompt)
    except Exception:
        logger.exception("summarisation failed; using deterministic fallback")
        return fallback_summary(slots, messages)

    if not isinstance(result, ConversationSummary):
        logger.warning("summariser returned an unexpected shape; using fallback")
        return fallback_summary(slots, messages)

    # Captured slots are ground truth — they came from an explicit tool call.
    # The summariser may fill gaps but never overwrite.
    for field in ("use_case", "budget", "timeline"):
        if slots.get(field):
            setattr(result, field, slots[field])

    return result

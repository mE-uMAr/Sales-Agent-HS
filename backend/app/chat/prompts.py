"""Prompt assembly.

Templates live in ``prompts/*.md`` so the sales tone can be edited by someone
who does not want to open a Python file. Only two things vary per turn — the
current objective and the facts captured so far — which keeps the prompt stable
enough to be cached by providers that support it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from app.chat.state import STAGE_OBJECTIVES, ConversationState, Stage
from app.config import get_settings
from app.knowledge.pricing import get_catalog

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_SLOT_LABELS = {
    "use_case": "What they need",
    "budget": "Budget",
    "timeline": "Timeline",
    "service_interest": "Service of interest",
}


@lru_cache(maxsize=8)
def load_template(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def format_known_facts(
    slots: dict[str, str],
    quoted_prices: list[dict[str, Any]] | None = None,
    unanswered: list[str] | None = None,
) -> str:
    lines: list[str] = []

    for key, label in _SLOT_LABELS.items():
        value = (slots or {}).get(key)
        if value:
            lines.append(f"- {label}: {value}")

    # Carry the figures, not just the fact of a quote, so the model can refer
    # back to what it already said without calling the pricing tool again.
    for quote in quoted_prices or []:
        currency = quote.get("currency", "USD")
        lines.append(
            f"- Already quoted: {quote.get('service_name')} "
            f"({quote.get('tier')}) at {currency} {quote.get('our_price'):,.0f} "
            f"against a market price of {currency} {quote.get('market_price'):,.0f}. "
            "They have seen this — refer back to it rather than quoting it again."
        )

    for question in unanswered or []:
        lines.append(f"- Could not answer: {question}")

    return "\n".join(lines) if lines else "- Nothing captured yet."


def build_system_prompt(state: ConversationState) -> str:
    settings = get_settings()
    stage = Stage(state.get("stage", Stage.GREETING))

    return load_template("system").format(
        company_name=settings.company_name,
        first_name=state.get("first_name", "there"),
        stage_objective=STAGE_OBJECTIVES[stage],
        known_facts=format_known_facts(
            state.get("slots", {}),
            state.get("quoted_prices", []),
            state.get("unanswered", []),
        ),
        service_catalog=get_catalog().catalog_overview(),
    )


def build_summary_prompt(transcript: str, captured_facts: str) -> str:
    return load_template("summarize").format(
        company_name=get_settings().company_name,
        transcript=transcript,
        captured_facts=captured_facts,
    )

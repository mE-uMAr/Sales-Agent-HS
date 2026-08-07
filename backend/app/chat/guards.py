"""Input sanitisation and output verification.

The third layer of leak prevention, and the last line of defence against an
invented price. Both run on text that has already left the model, so neither
depends on the model behaving.

The price check is the interesting one: it extracts every currency figure from
the reply and verifies it against the pricing catalog. A number the catalog does
not contain cannot reach a visitor, no matter how confidently it was generated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.chat.copy import BLOCKED_OUTPUT
from app.knowledge.pricing import PricingCatalog, get_catalog
from app.observability import get_logger

logger = get_logger(__name__)

MAX_INPUT_CHARS = 2000

#: Phrases that only make sense if internal commercial detail has escaped.
#: Deliberately specific — a bare "margin" or "cost" is normal sales language.
INTERNAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(gross|net|profit|floor|target)\s+margins?\b",
        r"\bmargins?\s+(is|are|of)\s+\d",
        r"\bour\s+(internal\s+)?(cost|costs|margin|markup|rate\s+card)\b",
        r"\bcost\s+(price|basis|per\s+hour)\b",
        r"\binternal\s+(cost|rate|engineering\s+cost|note)\b",
        r"\b(salary|salaries|pay\s+band|compensation\s+band)\b",
        r"\bdiscount\s+authority\b",
        r"\bblended\s+(internal\s+)?(cost|rate)\b",
        r"\bpipeline\s+notes?\b",
        r"ZEPHYRINE-LEDGER",  # canary from content/internal
    )
)

_NUM = r"\d[\d,]*(?:\.\d+)?"
_UNIT = r"k|m|thousand|million"

#: Money written with a leading symbol, optionally as a range:
#: "$12,500", "$20-30,000", "$9.5k to $12k".
_MONEY_PREFIXED = re.compile(
    rf"[$£€]\s*(?P<a>{_NUM})\s*(?P<au>{_UNIT})?"
    rf"(?:\s*(?:-|–|—|to)\s*[$£€]?\s*(?P<b>{_NUM})\s*(?P<bu>{_UNIT})?)?",
    re.IGNORECASE,
)

#: Money written with a trailing currency word: "30 thousand dollars",
#: "20 to 30k USD".
_MONEY_SUFFIXED = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<au>{_UNIT})?"
    rf"(?:\s*(?:-|–|—|to)\s*(?P<b>{_NUM})\s*(?P<bu>{_UNIT})?)?"
    r"\s*(?:USD|usd|dollars|dollar)\b",
    re.IGNORECASE,
)

#: Money written with a leading currency word: "USD 64,000".
_MONEY_WORD_PREFIXED = re.compile(
    rf"\b(?:USD|usd)\s*(?P<a>{_NUM})\s*(?P<au>{_UNIT})?"
    rf"(?:\s*(?:-|–|—|to)\s*(?:USD\s*)?(?P<b>{_NUM})\s*(?P<bu>{_UNIT})?)?",
    re.IGNORECASE,
)

_UNIT_SCALE = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}

#: A malformed tool call the model wrote into its own text instead of emitting
#: properly. Groq sometimes rejects these with a 400 and sometimes hands them
#: back as content — this catches the second case before a visitor sees it.
_TOOL_ARTIFACT = re.compile(
    r"<\s*function\s*=?[^>]*>.*?(?:<\s*/\s*function\s*>|$)"
    r"|<\|?(?:tool_call|function_call)\|?>.*?(?:<\|?/?(?:tool_call|function_call)\|?>|$)",
    re.IGNORECASE | re.DOTALL,
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Logged, never blocked — a visitor typing this is curious, not an attacker,
#: and blocking real people to inconvenience a prompt is a poor trade.
_SUSPICIOUS_INPUT = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
    r"|disregard\s+(your|the)\s+(instructions|prompt|rules)"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+"
    r"|reveal\s+(your|the)\s+(prompt|instructions))",
    re.IGNORECASE,
)


def sanitise_input(text: str) -> str:
    """Normalise and bound a visitor message.

    Visitor text is never interpolated into the system prompt — it only ever
    arrives as a user turn — so this is hygiene rather than the injection
    defence. The defence is that there is nothing private to reach.
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_CHARS.sub("", cleaned).strip()

    if len(cleaned) > MAX_INPUT_CHARS:
        cleaned = cleaned[:MAX_INPUT_CHARS].rstrip() + "…"

    if _SUSPICIOUS_INPUT.search(cleaned):
        logger.warning(
            "prompt-injection-shaped input", extra={"excerpt": cleaned[:160]}
        )

    return cleaned


def _to_amount(raw: str, unit: str | None) -> float | None:
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    return value * _UNIT_SCALE.get((unit or "").lower(), 1)


def extract_money(text: str) -> set[float]:
    """Every monetary figure in a piece of text.

    Handles ranges and magnitude words, and infers a missing scale from the
    other end of a range — ``$20-30,000`` means twenty *thousand* to thirty
    thousand, not twenty dollars. Getting that wrong is what makes a naive
    version of this guard reject the bot for repeating the visitor's own budget.
    """
    figures: set[float] = set()

    for pattern in (_MONEY_PREFIXED, _MONEY_SUFFIXED, _MONEY_WORD_PREFIXED):
        for match in pattern.finditer(text):
            a_raw, a_unit = match.group("a"), match.group("au")
            b_raw, b_unit = match.group("b"), match.group("bu")

            if b_raw is None:
                amount = _to_amount(a_raw, a_unit)
                if amount is not None:
                    figures.add(amount)
                continue

            # Range: a bare low end inherits the high end's scale.
            a_amount = _to_amount(a_raw, a_unit or b_unit)
            b_amount = _to_amount(b_raw, b_unit)
            if (
                a_amount is not None
                and b_amount is not None
                and not a_unit
                and not b_unit
                and a_amount < b_amount / 100
            ):
                # "$20-30,000" — scale the low end to the high end's magnitude.
                while a_amount < b_amount / 100:
                    a_amount *= 1_000
            for amount in (a_amount, b_amount):
                if amount is not None:
                    figures.add(amount)

    return figures


def strip_tool_artifacts(text: str) -> tuple[str, bool]:
    """Remove malformed tool-call text the model wrote into its reply.

    Returns the cleaned text and whether anything was removed. A stripped reply
    is usually incomplete — the tool never ran — so the caller should treat
    ``True`` as a signal to ask the visitor to rephrase rather than to ship a
    truncated answer.
    """
    if "<function" not in text.lower() and "tool_call" not in text.lower():
        return text, False

    cleaned = _TOOL_ARTIFACT.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, cleaned != text.strip()


def allowed_price_figures(catalog: PricingCatalog) -> set[float]:
    """Every number the bot is permitted to say with a currency attached."""
    allowed: set[float] = set()
    for service_id in catalog.service_ids:
        for quote in catalog.quote(service_id):
            allowed.add(quote.market_price)
            allowed.add(quote.our_price)
            allowed.add(quote.savings)
    return allowed


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    text: str
    reason: str | None = None


def check_output(
    reply: str,
    catalog: PricingCatalog | None = None,
    conversation_text: str = "",
) -> GuardVerdict:
    """Verify an assistant reply before it reaches the visitor.

    ``conversation_text`` is everything already said in this conversation. Any
    figure the visitor themselves mentioned is fair to repeat — echoing "your
    budget of $20-30k" back at them is normal and must not be blocked. What is
    not allowed is a figure that appears neither in the catalog nor anywhere in
    the conversation, because the only place left for it to have come from is
    the model's imagination.
    """
    if not reply or not reply.strip():
        return GuardVerdict(allowed=True, text=reply)

    for pattern in INTERNAL_PATTERNS:
        if pattern.search(reply):
            logger.error(
                "blocked reply containing internal-sounding content",
                extra={"pattern": pattern.pattern, "excerpt": reply[:200]},
            )
            return GuardVerdict(
                allowed=False, text=BLOCKED_OUTPUT, reason="internal_content"
            )

    catalog = catalog or get_catalog()
    quoted = extract_money(reply)
    if quoted:
        permitted = allowed_price_figures(catalog) | extract_money(conversation_text)
        invented = {figure for figure in quoted if figure not in permitted}
        if invented:
            logger.error(
                "blocked reply containing a price from no known source",
                extra={"invented": sorted(invented), "excerpt": reply[:200]},
            )
            return GuardVerdict(
                allowed=False,
                text=(
                    "I want to make sure I give you an accurate figure rather "
                    "than an approximate one, so let me have someone from the "
                    "team confirm the pricing for you and follow up shortly."
                ),
                reason="unverified_price",
            )

    return GuardVerdict(allowed=True, text=reply)

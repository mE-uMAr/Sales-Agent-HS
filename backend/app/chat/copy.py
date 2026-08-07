"""Canned visitor-facing text.

Every sentence the application puts in front of a visitor without the model
choosing it lives here, so the tone can be changed in one place and asserted on
in tests.
"""

from __future__ import annotations

#: The honest "I don't know" — returned by our code, not generated, so it cannot
#: drift into a guess.
UNKNOWN_ANSWER = (
    "I'm an AI assistant and I don't know that one, so I'd rather not guess. "
    "I've noted it down and a member of the team will get back to you on it."
)

HANDOFF_LINE = (
    "A member of our team will reach out to you shortly to pick this up "
    "properly."
)

ESCALATION_ACK = (
    "Of course — I'll pass this to the team now. " + HANDOFF_LINE
)

MAX_TURNS_CLOSING = (
    "We've covered a good amount here, and I think a person can help you "
    "better from this point. " + HANDOFF_LINE
)

KNOWLEDGE_GAP_CLOSING = (
    "There have been a couple of things I couldn't answer well, and I don't "
    "want to guess at them. " + HANDOFF_LINE
)

ERROR_CLOSING = (
    "Sorry — I'm having a technical problem on my side and don't want to leave "
    "you waiting. " + HANDOFF_LINE
)

BLOCKED_OUTPUT = (
    "That's not something I can go into, I'm afraid. " + HANDOFF_LINE
)

#: The model wrote a malformed tool call into its reply and we stripped it, so
#: whatever it meant to look up never happened. Ask rather than half-answer.
GLITCHED_TURN = (
    "Sorry — something went wrong on my side just then and I didn't finish "
    "that thought. Could you ask me that again?"
)

SESSION_CLOSED = "This conversation has ended. Refresh the page to start a new one."

#: Sent when the conversation opens, before the model is invoked at all, so the
#: first paint is instant and costs nothing.
def greeting(first_name: str, company_name: str) -> str:
    return (
        f"Hi {first_name}, thanks for getting in touch with {company_name}. "
        "I can answer questions about what we do, our past work and our "
        "pricing, and make sure the right person follows up with you.\n\n"
        "To start — what are you looking to build or solve?"
    )

"""Chat model factory and a circuit breaker.

Provider choice is one setting. Groq today, OpenAI later, and a ``fake``
provider so the whole conversation graph can be tested without a network call
or an API key.

The circuit breaker matters more than it usually would: Groq's free tier has
real per-minute ceilings, and the correct behaviour when we hit one is to hand
the visitor to a human immediately rather than to make them watch retries fail.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings
from app.observability import get_logger

logger = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """The model could not be reached. Callers should hand off, not retry."""


class CircuitBreaker:
    """Stop calling a failing provider for a cooling-off period.

    Three consecutive failures opens the circuit for 60 seconds; the next call
    after that is allowed through and closes the circuit if it succeeds.
    """

    def __init__(self, threshold: int = 3, reset_after_seconds: float = 60.0) -> None:
        self._threshold = threshold
        self._reset_after = reset_after_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self._reset_after:
            # Half-open: let one request through to probe.
            self._opened_at = None
            self._failures = self._threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.error(
                "llm circuit opened",
                extra={"failures": self._failures, "cooldown_s": self._reset_after},
            )

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


chat_breaker = CircuitBreaker()

_overrides: dict[str, BaseChatModel] = {}


def set_chat_model_override(model: BaseChatModel, *, summary: bool = False) -> None:
    """Inject a model. Used by tests to drive the graph deterministically."""
    _overrides["summary" if summary else "chat"] = model


def reset_chat_model_overrides() -> None:
    _overrides.clear()


def build_chat_model(
    *,
    summary: bool = False,
    model_name: str | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """The conversational model, or the cheaper one used for summarising.

    ``model_name`` overrides the configured model — used for the rate-limit
    fallback.
    """
    key = "summary" if summary else "chat"
    if key in _overrides:
        return _overrides[key]

    settings = settings or get_settings()
    model_name = model_name or (
        settings.llm_summary_model if summary else settings.llm_model
    )

    common: dict[str, Any] = {
        "model": model_name,
        # Summarising is an extraction task; creativity is a liability there.
        "temperature": 0.0 if summary else settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "timeout": settings.llm_timeout_seconds,
        **kwargs,
    }

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        return ChatGroq(
            api_key=settings.groq_api_key,
            max_tokens=settings.llm_max_tokens,
            **common,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            max_completion_tokens=settings.llm_max_tokens,
            **common,
        )

    raise ValueError(
        f"LLM_PROVIDER={settings.llm_provider!r} needs a model override. "
        "Call set_chat_model_override() (tests) or set a real provider."
    )

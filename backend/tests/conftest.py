"""Shared fixtures.

The important one is :class:`ScriptedChatModel`: it lets a whole conversation —
tool calls, slot fills, escalations, terminations — run through the real graph
with no network, no API key and no nondeterminism. Every behavioural test in
this suite is a scripted transcript rather than a mock of an internal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import get_settings
from app.knowledge.pricing import PricingCatalog


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of assistant messages, one per invocation.

    ``bind_tools`` and ``with_structured_output`` are accepted and ignored so
    the model can stand in anywhere the real one is used.
    """

    responses: list[AIMessage] = []
    structured_response: Any = None
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        if not self.responses:
            raise AssertionError("ScriptedChatModel has no responses configured")
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[index])]
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        response = self.structured_response

        class _Structured:
            async def ainvoke(self, _prompt: Any, **_kw: Any) -> Any:
                if response is None:
                    raise RuntimeError("no structured response configured")
                return response

            def invoke(self, _prompt: Any, **_kw: Any) -> Any:
                if response is None:
                    raise RuntimeError("no structured response configured")
                return response

        return _Structured()


class FlakyChatModel(ScriptedChatModel):
    """Fails a set number of times, then replays the script.

    Used to exercise the rate-limit fallback: the first call raises as the main
    model would when its quota is exhausted, and the retry succeeds.
    """

    failures_remaining: int = 0
    failure_message: str = "Error code: 429 - rate_limit_exceeded on tokens per day"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError(self.failure_message)
        return super()._generate(messages, stop, run_manager, **kwargs)


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """An assistant turn that asks for exactly one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch) -> Iterator[None]:
    """Point every test at its own database and a keyless LLM."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LEAD_SINK", "sqlite")
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.setenv("MAX_TURNS", "25")
    monkeypatch.setenv("MAX_UNANSWERED_STREAK", "2")

    get_settings.cache_clear()
    _reset_singletons()
    yield
    get_settings.cache_clear()
    _reset_singletons()


def _reset_singletons() -> None:
    """Clear module-level caches so tests cannot leak state into each other."""
    import app.chat.graph as graph_module
    import app.chat.service as chat_service
    import app.knowledge.retriever as retriever_module
    import app.leads.service as lead_service
    from app.chat.llm import chat_breaker, reset_chat_model_overrides

    reset_chat_model_overrides()
    chat_breaker.reset()
    graph_module._graph = None
    chat_service._service = None
    lead_service._service = None
    retriever_module._retriever = None


@pytest.fixture
async def database() -> AsyncIterator[None]:
    from app.persistence.db import dispose_engine, init_db

    await dispose_engine()
    await init_db()
    yield
    await dispose_engine()


@pytest.fixture
def catalog() -> PricingCatalog:
    return PricingCatalog.from_file(
        get_settings().public_content_dir / "pricing.yaml"
    )


@pytest.fixture
def no_retrieval(monkeypatch):
    """Make the knowledge base return nothing, without touching Chroma."""
    from app.knowledge import retriever as retriever_module

    class _Empty:
        def search(self, *_args, **_kwargs):
            return []

        def is_ready(self):
            return True

    monkeypatch.setattr(retriever_module, "get_retriever", lambda: _Empty())
    import app.chat.tools as tools_module

    monkeypatch.setattr(tools_module, "get_retriever", lambda: _Empty())

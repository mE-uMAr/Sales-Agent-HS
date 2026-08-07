"""Application settings.

Everything tunable lives here and is populated from environment variables or a
local `.env` file. Nothing else in the codebase reads `os.environ` directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Chroma phones home on import unless told not to, and its failure path is a
# stream of ERROR logs. Set before chromadb is ever imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.NoopTelemetry")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── application ───────────────────────────────────────────────────
    app_name: str = "Hashed Systems Assistant"
    company_name: str = "Hashed Systems"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai", "fake"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_summary_model: str = "llama-3.1-8b-instant"
    #: Used when the main model is rate-limited. Providers meter per model, so a
    #: smaller one usually still has quota — a slightly worse answer beats
    #: ending the conversation. Set empty to disable.
    llm_fallback_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 700
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # ── embeddings (local by default: Groq serves no embeddings API) ──
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── knowledge base ────────────────────────────────────────────────
    content_dir: Path = BASE_DIR / "content"
    chroma_dir: Path = BASE_DIR / "var" / "chroma"
    chroma_collection: str = "public_kb"
    #: Build the index at startup when it is missing or empty. Makes a fresh
    #: deployment work without a separate ingest step; a warm one skips it.
    auto_index_on_startup: bool = True
    retrieval_k: int = 4
    # Below this relevance the passage is noise. Dropping it is what makes the
    # bot say "I don't know" instead of answering from a loosely-related chunk.
    # Calibrated against bge-small: on-topic hits score ~0.5-0.7, off-topic ~0.3.
    retrieval_score_threshold: float = 0.45
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ── conversation limits ───────────────────────────────────────────
    max_turns: int = 25
    max_unanswered_streak: int = 2
    max_tool_iterations: int = 6
    session_idle_minutes: int = 30

    # ── database ──────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./var/app.db"

    # ── lead capture ──────────────────────────────────────────────────
    # The local `leads` table is ALWAYS written — it is the durable record and
    # the outbox. This setting only chooses where leads are *forwarded*:
    #   sqlite -> nowhere (local database is the destination)
    #   http   -> POST to CRM_WEBHOOK_URL, with retry
    lead_sink: Literal["sqlite", "http"] = "sqlite"
    crm_webhook_url: str | None = None
    crm_webhook_secret: SecretStr | None = None
    outbox_enabled: bool = True
    outbox_poll_seconds: int = 30
    outbox_max_attempts: int = 8
    outbox_batch_size: int = 20

    # ── security ──────────────────────────────────────────────────────
    session_token_secret: SecretStr = SecretStr("change_me_in_production")
    admin_api_key: SecretStr = SecretStr("change_me_in_production")
    widget_public_key: str = "pub_dev"
    session_token_ttl_minutes: int = 120
    # NoDecode: without it pydantic-settings tries to JSON-parse the env value
    # before the validator below gets to split the comma-separated form.
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )
    rate_limit_messages_per_minute: int = 20
    rate_limit_sessions_per_hour: int = 10

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string so `.env` stays readable."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def public_content_dir(self) -> Path:
        """Only this subtree is ever indexed."""
        return self.content_dir / "public"

    @property
    def internal_content_dir(self) -> Path:
        """Exists so internal material has an obvious home that is never indexed."""
        return self.content_dir / "internal"

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    def ensure_runtime_dirs(self) -> None:
        (BASE_DIR / "var").mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

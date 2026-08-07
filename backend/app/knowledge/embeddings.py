"""Embedding models.

Groq serves chat completions but **no embeddings endpoint**, so retrieval runs
on a local ONNX model. That is a feature rather than a workaround: embeddings
cost nothing, work offline, keep document text out of a third party, and — most
usefully — the vector index does not need rebuilding when the chat provider
changes from Groq to OpenAI.

The FastEmbed adapter is written directly against ``langchain_core.embeddings``
so the project does not need the whole ``langchain-community`` dependency tree
for one class.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.observability import get_logger

logger = get_logger(__name__)


class FastEmbedEmbeddings(Embeddings):
    """LangChain adapter over `fastembed <https://github.com/qdrant/fastembed>`_.

    Model weights are downloaded once to the HuggingFace cache and then run
    through ONNX Runtime — no torch, roughly 130 MB on disk.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    logger.info(
                        "loading embedding model", extra={"model": self.model_name}
                    )
                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [vector.tolist() for vector in model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        return next(iter(model.query_embed([text]))).tolist()


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    return FastEmbedEmbeddings(model_name=settings.embedding_model)

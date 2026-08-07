"""Audience-filtered retrieval over the public knowledge base.

Second layer of leak prevention: every query is filtered to
``audience == "public"``. Ingestion already refuses to index anything else, so
this is redundant by design — it catches a mis-tagged document that slipped
through, and it means a future change to ingestion cannot silently widen what
the bot can see.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.knowledge.embeddings import build_embeddings
from app.observability import get_logger

logger = get_logger(__name__)

PUBLIC_FILTER: dict[str, Any] = {"audience": "public"}


@dataclass(frozen=True)
class KnowledgeSnippet:
    """One retrieved passage, with enough provenance to cite it."""

    text: str
    title: str
    source: str
    doc_type: str
    score: float

    def to_display(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "doc_type": self.doc_type,
            "source": self.source,
            "content": self.text,
        }


class KnowledgeRetriever:
    """Thin wrapper over Chroma that can never be asked for private material."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._store: Any | None = None
        self._lock = threading.Lock()

    def _get_store(self) -> Any:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    from langchain_chroma import Chroma

                    self._store = Chroma(
                        collection_name=self._settings.chroma_collection,
                        embedding_function=build_embeddings(self._settings),
                        persist_directory=str(self._settings.chroma_dir),
                    )
        return self._store

    def is_ready(self) -> bool:
        """True when the collection exists and holds at least one chunk."""
        try:
            return self._get_store()._collection.count() > 0
        except Exception:
            logger.warning("knowledge index unavailable", exc_info=True)
            return False

    def search(self, query: str, k: int | None = None) -> list[KnowledgeSnippet]:
        query = (query or "").strip()
        if not query:
            return []

        k = k or self._settings.retrieval_k
        try:
            results = self._get_store().similarity_search_with_relevance_scores(
                query, k=k, filter=PUBLIC_FILTER
            )
        except Exception:
            # Retrieval failing must degrade to "I don't know", never to a 500.
            logger.exception("retrieval failed", extra={"k": k})
            return []

        threshold = self._settings.retrieval_score_threshold
        snippets = [
            KnowledgeSnippet(
                text=document.page_content.strip(),
                title=str(document.metadata.get("title", "Untitled")),
                source=str(document.metadata.get("source", "unknown")),
                doc_type=str(document.metadata.get("doc_type", "general")),
                score=round(float(score), 4),
            )
            for document, score in results
            if score >= threshold
        ]

        logger.info(
            "knowledge search",
            extra={
                "hits": len(snippets),
                "candidates": len(results),
                "threshold": threshold,
                "top_score": snippets[0].score if snippets else None,
            },
        )
        return snippets


_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached store so a re-index is picked up without a restart."""
    global _retriever
    _retriever = None

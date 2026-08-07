"""Internal material must not be reachable.

Three independent layers, tested independently, because the whole point is that
no single one of them has to hold on its own:

1. ingestion never reads outside ``content/public``
2. retrieval always filters to public documents
3. the output guard catches internal-sounding text on the way out
"""

from __future__ import annotations

import pytest

from app.chat.guards import check_output
from app.config import get_settings
from app.knowledge.ingest import chunk_documents, load_documents
from app.knowledge.retriever import PUBLIC_FILTER

#: Appears only in content/internal. If it ever surfaces, layer 1 has failed.
CANARY = "ZEPHYRINE-LEDGER-9931"


def test_the_canary_actually_exists_in_the_internal_tree() -> None:
    """Guards the guard: a rotted fixture would make the tests below vacuous."""
    internal = get_settings().internal_content_dir
    assert internal.exists(), "content/internal is missing"
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in internal.rglob("*.md")
    )
    assert CANARY in corpus


# ── layer 1: ingestion ───────────────────────────────────────────────
def test_ingestion_reads_only_the_public_tree() -> None:
    documents, _ = load_documents()
    public_dir = get_settings().public_content_dir

    assert documents, "expected some public content to index"
    for document in documents:
        resolved = (public_dir / document.metadata["source"]).resolve()
        assert resolved.is_relative_to(public_dir.resolve())
        assert "internal" not in document.metadata["source"].split("/")


def test_no_indexed_chunk_contains_internal_material() -> None:
    documents, _ = load_documents()
    corpus = "\n".join(chunk.page_content for chunk in chunk_documents(documents))

    assert CANARY not in corpus
    for phrase in ("gross margin", "discount authority", "blended internal cost"):
        assert phrase.lower() not in corpus.lower()


def test_every_document_is_tagged_public() -> None:
    documents, _ = load_documents()
    assert all(doc.metadata["audience"] == "public" for doc in documents)


def test_pricing_yaml_is_not_vectorised() -> None:
    """Numbers come from the catalog, never from a retrieved chunk."""
    _, report = load_documents()
    assert any("pricing.yaml" in skipped for skipped in report.skipped)


def test_a_public_file_marked_internal_is_skipped(tmp_path, monkeypatch) -> None:
    content = tmp_path / "content"
    (content / "public" / "about").mkdir(parents=True)
    (content / "public" / "about" / "ok.md").write_text(
        "---\ntitle: Fine\naudience: public\n---\n\nPublic body.",
        encoding="utf-8",
    )
    (content / "public" / "about" / "oops.md").write_text(
        f"---\ntitle: Oops\naudience: internal\n---\n\n{CANARY}",
        encoding="utf-8",
    )

    monkeypatch.setenv("CONTENT_DIR", str(content))
    get_settings.cache_clear()

    documents, report = load_documents()

    assert [d.metadata["title"] for d in documents] == ["Fine"]
    assert any("oops.md" in skipped for skipped in report.skipped)
    assert CANARY not in "\n".join(d.page_content for d in documents)


# ── layer 2: retrieval ───────────────────────────────────────────────
def test_retrieval_filter_is_pinned_to_public() -> None:
    assert PUBLIC_FILTER == {"audience": "public"}


def test_retriever_always_passes_the_public_filter(monkeypatch) -> None:
    from app.knowledge.retriever import KnowledgeRetriever

    seen: dict[str, object] = {}

    class _Store:
        def similarity_search_with_relevance_scores(self, query, k, filter):
            seen["filter"] = filter
            return []

    retriever = KnowledgeRetriever()
    monkeypatch.setattr(retriever, "_get_store", lambda: _Store())
    retriever.search("anything at all")

    assert seen["filter"] == PUBLIC_FILTER


def test_retrieval_failure_degrades_to_no_results(monkeypatch) -> None:
    """A broken index must mean 'I don't know', never a 500."""
    from app.knowledge.retriever import KnowledgeRetriever

    class _Broken:
        def similarity_search_with_relevance_scores(self, *_a, **_kw):
            raise RuntimeError("index corrupt")

    retriever = KnowledgeRetriever()
    monkeypatch.setattr(retriever, "_get_store", lambda: _Broken())

    assert retriever.search("anything") == []


# ── layer 3: the output guard ────────────────────────────────────────
@pytest.mark.parametrize(
    "reply",
    [
        f"The internal note says {CANARY}.",
        "Our gross margin on this work is 58%.",
        "Our internal cost is 46 an hour.",
        "Delivery leads have discount authority up to 12%.",
        "Here are the pipeline notes you asked for.",
    ],
)
def test_internal_phrasing_is_blocked_on_the_way_out(reply: str) -> None:
    verdict = check_output(reply)
    assert not verdict.allowed
    assert CANARY not in verdict.text
    assert verdict.reason == "internal_content"

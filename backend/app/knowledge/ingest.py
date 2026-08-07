"""Build the vector index from ``content/public``.

The first and strongest layer of leak prevention lives here: **only**
``content/public`` is ever read. Anything under ``content/internal`` never
enters the index, so it cannot be retrieved — not by a clever prompt, not by a
mis-scoped filter, not by a bug in the retriever. What is not indexed cannot
leak.

Run with::

    python -m app.knowledge.ingest --reset
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.knowledge.embeddings import build_embeddings
from app.observability import configure_logging, get_logger

logger = get_logger(__name__)

#: Only prose is vectorised. `pricing.yaml` is served by the pricing catalog.
INDEXABLE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

#: Directory name under content/public -> doc_type tag.
KNOWN_DOC_TYPES = frozenset({"about", "services", "projects", "faq", "process"})


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    doc_types: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "doc_types": self.doc_types,
        }


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split optional ``---`` YAML front matter from the body."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        logger.warning("unparseable front matter; ignoring it")
        return {}, text

    return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")


def _derive_doc_type(path: Path, public_dir: Path) -> str:
    relative = path.relative_to(public_dir)
    if len(relative.parts) > 1 and relative.parts[0] in KNOWN_DOC_TYPES:
        return relative.parts[0]
    return "general"


def _title_from(path: Path, meta: dict[str, Any]) -> str:
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def load_documents(settings: Settings | None = None) -> tuple[list[Document], IngestReport]:
    settings = settings or get_settings()
    public_dir = settings.public_content_dir.resolve()
    report = IngestReport()

    if not public_dir.exists():
        raise FileNotFoundError(
            f"no public content directory at {public_dir}. "
            "Create it and add markdown files before indexing."
        )

    documents: list[Document] = []

    for path in sorted(public_dir.rglob("*")):
        if not path.is_file():
            continue

        resolved = path.resolve()
        # Belt and braces: a symlink must not walk us out of the public tree.
        if not resolved.is_relative_to(public_dir):
            report.skipped.append(f"{path.name} (resolves outside content/public)")
            logger.error(
                "refusing file that escapes the public content directory",
                extra={"path": str(path), "resolved": str(resolved)},
            )
            continue

        if resolved.suffix.lower() not in INDEXABLE_SUFFIXES:
            report.skipped.append(f"{resolved.name} (not an indexable file type)")
            continue

        raw = resolved.read_text(encoding="utf-8").strip()
        if not raw:
            report.skipped.append(f"{resolved.name} (empty)")
            continue

        meta, body = _parse_front_matter(raw)
        if not body.strip():
            report.skipped.append(f"{resolved.name} (no body after front matter)")
            continue

        # An author can mark a public-tree file as internal; honour it loudly.
        if str(meta.get("audience", "public")).lower() != "public":
            report.skipped.append(f"{resolved.name} (front matter marks it non-public)")
            logger.warning(
                "skipping non-public document inside content/public",
                extra={"path": str(resolved.relative_to(public_dir))},
            )
            continue

        doc_type = str(meta.get("doc_type") or _derive_doc_type(resolved, public_dir))
        title = _title_from(resolved, meta)

        documents.append(
            Document(
                page_content=body,
                metadata={
                    "source": str(resolved.relative_to(public_dir)),
                    "title": title,
                    "doc_type": doc_type,
                    "audience": "public",
                },
            )
        )
        report.files += 1
        report.doc_types[doc_type] = report.doc_types.get(doc_type, 0) + 1

    return documents, report


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        # Prefixing the title keeps the heading in the embedded text, which
        # noticeably helps short, keyword-ish visitor questions.
        chunk.page_content = f"{chunk.metadata.get('title', '')}\n\n{chunk.page_content}".strip()

    return chunks


def build_index(*, reset: bool = True, settings: Settings | None = None) -> IngestReport:
    settings = settings or get_settings()
    settings.ensure_runtime_dirs()

    documents, report = load_documents(settings)
    if not documents:
        raise RuntimeError(
            f"no indexable documents found under {settings.public_content_dir}"
        )

    chunks = chunk_documents(documents, settings)
    report.chunks = len(chunks)

    if reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=build_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )
    store.add_documents(chunks)

    logger.info("index built", extra=report.as_dict())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index content/public into Chroma.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="add to the existing collection instead of rebuilding it",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    report = build_index(reset=not args.no_reset, settings=settings)

    print(f"\nIndexed {report.files} files into {report.chunks} chunks.")
    for doc_type, count in sorted(report.doc_types.items()):
        print(f"  {doc_type:<12} {count} file(s)")
    if report.skipped:
        print("\nSkipped:")
        for item in report.skipped:
            print(f"  - {item}")
    print(f"\nVector store: {settings.chroma_dir}")


if __name__ == "__main__":
    main()

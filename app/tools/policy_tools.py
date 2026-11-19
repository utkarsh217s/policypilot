"""Policy tools: the four operations the agent uses to gather evidence.

    search_policies       - hybrid search for candidate chunks
    get_clause            - full text of one exact clause (policy + version + section)
    list_policy_versions  - every indexed revision of a policy, with dates
    get_related_sections  - clauses this clause cites, and clauses that cite it

These are plain Python functions returning JSON-friendly dicts. The MCP server
(app/tools/mcp_server.py) exposes them to MCP clients; tests and scripts can
call them directly.

Only search_policies in 'hybrid' or 'semantic' mode calls Gemini. The other
tools read stored chunks from Qdrant and make no API calls.
"""

import threading
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.retrieval.hybrid_search import hybrid_search, load_indexed_chunks
from app.retrieval.vector_store import (
    VectorStoreError,
    build_client,
    build_embeddings,
)


class PolicyToolError(ValueError):
    """Readable error returned to the tool caller."""


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------
class ToolContext:
    """Holds one Qdrant client and one embedding client for the process.

    Local Qdrant storage allows only one client per folder, so the server must
    share a single client rather than open one per tool call. The lock keeps
    tool calls from using it concurrently.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: QdrantClient | None = None
        self._embeddings: Embeddings | None = None
        self.lock = threading.RLock()

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = build_client(self.settings)
        return self._client

    @property
    def embeddings(self) -> Embeddings:
        # Created only when a semantic search needs it, so the other tools
        # work without a Gemini key.
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    def set_embeddings(self, embeddings: Embeddings) -> None:
        """Override the embedding client (used by tests)."""
        self._embeddings = embeddings

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require(value: str | None, name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise PolicyToolError(f"{name} is required")
    return value


def _citation(meta: dict[str, Any]) -> dict[str, Any]:
    """Fields needed to cite a clause precisely."""
    return {
        "policy_id": meta.get("policy_id"),
        "title": meta.get("title"),
        "version": meta.get("version"),
        "section_number": meta.get("section_number"),
        "heading": meta.get("heading"),
        "heading_path": meta.get("heading_path") or [],
        "source_file": meta.get("source_file"),
        "page_start": meta.get("parent_page_start"),
        "page_end": meta.get("parent_page_end"),
        "effective_from": meta.get("effective_from"),
        "effective_to": meta.get("effective_to"),
        "clause_id": meta.get("clause_id"),
    }


def _reassemble(chunks: list[Document]) -> str:
    """Rebuild a clause's text from its overlapping chunks.

    Each chunk stores 'start_index' (its position in the original clause),
    so overlapping text is skipped rather than repeated. Whitespace that the
    splitter dropped at a chunk boundary is restored as a single space.
    """
    ordered = sorted(chunks, key=lambda d: d.metadata.get("chunk_index", 0))
    text = ""
    for doc in ordered:
        start = doc.metadata.get("start_index")
        content = doc.page_content
        if not text:
            text = content
        elif isinstance(start, int) and start < len(text):
            text += content[len(text) - start :]  # skip the overlapping part
        else:
            text += " " + content
    return text


def _clause_chunks(
    ctx: ToolContext, policy_id: str, version: str
) -> dict[str, list[Document]]:
    """All chunks of one policy revision, grouped by section number."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in load_indexed_chunks(ctx.client, ctx.settings, policy_id, version):
        grouped[str(doc.metadata.get("section_number"))].append(doc)
    return grouped


def _clause_payload(chunks: list[Document]) -> dict[str, Any]:
    meta = chunks[0].metadata
    return {
        **_citation(meta),
        "text": _reassemble(chunks),
        "chunk_count": len(chunks),
        "related_sections": meta.get("related_sections") or [],
    }


# ---------------------------------------------------------------------------
# Tool 1: search_policies
# ---------------------------------------------------------------------------
def search_policies(
    ctx: ToolContext,
    query: str,
    k: int = 5,
    policy_id: str | None = None,
    version: str | None = None,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Find candidate policy chunks for a question.

    Results are CANDIDATES ranked by relevance, not confirmed answers.
    Use get_clause to read a full clause before relying on it.
    """
    query = _require(query, "query")
    with ctx.lock:
        try:
            embeddings = None if mode == "keyword" else ctx.embeddings
            results = hybrid_search(
                query,
                ctx.client,
                ctx.settings,
                embeddings,
                k=k,
                policy_id=policy_id or None,
                version=version or None,
                mode=mode,
            )
        except VectorStoreError as exc:
            raise PolicyToolError(str(exc)) from exc

    return {
        "query": query,
        "mode": mode,
        "result_count": len(results),
        "results": [
            {
                **_citation(r.document.metadata),
                "text": r.document.page_content,
                "chunk_index": r.document.metadata.get("chunk_index"),
                "chunk_count": r.document.metadata.get("chunk_count"),
                "rrf_score": round(r.rrf_score, 6),
                "found_by": r.sources,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Tool 2: get_clause
# ---------------------------------------------------------------------------
def get_clause(
    ctx: ToolContext, policy_id: str, version: str, section_number: str
) -> dict[str, Any]:
    """Return the complete text of one clause, rebuilt from all its chunks.

    Version is required: the same section number can say different things
    in different revisions.
    """
    policy_id = _require(policy_id, "policy_id")
    version = _require(version, "version")
    section_number = _require(section_number, "section_number")

    with ctx.lock:
        try:
            grouped = _clause_chunks(ctx, policy_id, version)
        except VectorStoreError as exc:
            raise PolicyToolError(str(exc)) from exc

    if not grouped:
        raise PolicyToolError(f"Policy {policy_id} v{version} is not indexed")
    chunks = grouped.get(section_number)
    if not chunks:
        available = ", ".join(sorted(grouped))
        raise PolicyToolError(
            f"Section {section_number} not found in {policy_id} v{version}. "
            f"Available sections: {available}"
        )
    return _clause_payload(chunks)


# ---------------------------------------------------------------------------
# Tool 3: list_policy_versions
# ---------------------------------------------------------------------------
def list_policy_versions(ctx: ToolContext, policy_id: str) -> dict[str, Any]:
    """List every indexed revision of a policy with its effective dates.

    Use this to pick the revision that applies on a given date:
    effective_from is inclusive, effective_to is exclusive, null = open-ended.
    """
    policy_id = _require(policy_id, "policy_id")
    with ctx.lock:
        try:
            chunks = load_indexed_chunks(ctx.client, ctx.settings)
        except VectorStoreError as exc:
            raise PolicyToolError(str(exc)) from exc

    versions: dict[str, dict[str, Any]] = {}
    for doc in chunks:
        meta = doc.metadata
        if meta.get("policy_id") != policy_id:
            continue
        entry = versions.setdefault(
            str(meta.get("version")),
            {
                "version": meta.get("version"),
                "title": meta.get("title"),
                "effective_from": meta.get("effective_from"),
                "effective_to": meta.get("effective_to"),
                "applies_to": meta.get("applies_to") or [],
                "departments": meta.get("departments") or [],
                "source_file": meta.get("source_file"),
                "sections": set(),
            },
        )
        entry["sections"].add(str(meta.get("section_number")))

    if not versions:
        raise PolicyToolError(f"No indexed versions found for policy {policy_id}")

    ordered = sorted(
        versions.values(),
        # Undated revisions sort first; dates are ISO strings, so text order works.
        key=lambda v: (v["effective_from"] or "", str(v["version"])),
    )
    for entry in ordered:
        entry["section_count"] = len(entry.pop("sections"))
    return {"policy_id": policy_id, "version_count": len(ordered), "versions": ordered}


# ---------------------------------------------------------------------------
# Tool 4: get_related_sections
# ---------------------------------------------------------------------------
def get_related_sections(
    ctx: ToolContext, policy_id: str, version: str, section_number: str
) -> dict[str, Any]:
    """Return clauses linked to a clause, in both directions.

    - cites:    sections this clause refers to ("see Section 4")
    - cited_by: sections that refer to this clause

    'cited_by' matters for exceptions: an exception often points back to the
    general rule, while the rule itself does not mention the exception.
    Only links found in the same policy revision are followed.
    """
    policy_id = _require(policy_id, "policy_id")
    version = _require(version, "version")
    section_number = _require(section_number, "section_number")

    with ctx.lock:
        try:
            grouped = _clause_chunks(ctx, policy_id, version)
        except VectorStoreError as exc:
            raise PolicyToolError(str(exc)) from exc

    if section_number not in grouped:
        raise PolicyToolError(
            f"Section {section_number} not found in {policy_id} v{version}"
        )

    def targets(chunks: list[Document]) -> set[str]:
        return {
            str(ref.get("section_number"))
            for ref in chunks[0].metadata.get("related_sections") or []
            if ref.get("policy_id") == policy_id and ref.get("version") == version
        }

    cites = sorted(targets(grouped[section_number]) & grouped.keys())
    cited_by = sorted(
        number
        for number, chunks in grouped.items()
        if number != section_number and section_number in targets(chunks)
    )
    missing = sorted(targets(grouped[section_number]) - grouped.keys())

    return {
        "policy_id": policy_id,
        "version": version,
        "section_number": section_number,
        "cites": [_clause_payload(grouped[n]) for n in cites],
        "cited_by": [_clause_payload(grouped[n]) for n in cited_by],
        "missing_references": missing,
    }

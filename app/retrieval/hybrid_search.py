import argparse
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.core.config import Settings, get_settings
from app.retrieval.vector_store import (
    VectorStoreError,
    build_client,
    build_embeddings,
    revision_filter,
    search_policies,
)

RRF_K = 60

CANDIDATES_PER_RETRIEVER = 20

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "by",
    "with",
    "as",
    "at",
    "it",
    "this",
    "that",
    "i",
    "my",
    "me",
    "can",
    "do",
    "does",
    "what",
    "how",
    "who",
    "when",
    "which",
}

TOKEN_PATTERN = re.compile(r"selfsection_\d+(?:\.\d+)*|\d+(?:\.\d+)+|\w+")


@dataclass
class HybridResult:
    """One fused search result with an explanation of where it came from."""

    document: Document
    rrf_score: float
    semantic_rank: int | None = None  # 1-based rank in semantic results
    keyword_rank: int | None = None  # 1-based rank in keyword results
    semantic_score: float | None = None
    keyword_score: float | None = None
    sources: list[str] = field(default_factory=list)


SELF_SECTION_PREFIX = "selfsection_"
SECTION_NUMBER = re.compile(r"\d+(?:\.\d+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into tokens, drop stopwords."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


def query_tokens(query: str) -> list[str]:
    """Tokenize a query, adding a self-section token for each section number.
    """
    tokens = tokenize(query)
    extra = [
        SELF_SECTION_PREFIX + token
        for token in tokens
        if SECTION_NUMBER.fullmatch(token)
    ]
    return tokens + extra


def keyword_text(doc: Document) -> str:
    """Text used for BM25: section number + heading path + chunk content.
    """
    meta = doc.metadata
    number = str(meta.get("section_number", ""))
    parts = [
        f"section {number} {SELF_SECTION_PREFIX}{number}",
        " ".join(meta.get("heading_path") or [meta.get("heading", "")]),
        doc.page_content,
    ]
    return "\n".join(parts)


def load_indexed_chunks(
    client: QdrantClient,
    settings: Settings,
    policy_id: str | None = None,
    version: str | None = None,
) -> list[Document]:
    """Read stored chunks (text + metadata, no vectors) back from Qdrant.
    """
    if not client.collection_exists(settings.qdrant_collection):
        raise VectorStoreError("Nothing indexed yet. Run the index command first.")

    if bool(policy_id) != bool(version):
        raise VectorStoreError("Provide both policy_id and version, or neither")
    # Checking BOTH values lets the type checker narrow them from "str | None"
    # to "str", which revision_filter() requires.
    query_filter = (
        revision_filter(policy_id, version) if policy_id and version else None
    )
    documents: list[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            # LangChain's QdrantVectorStore stores text under "page_content"
            # and our metadata under "metadata".
            documents.append(
                Document(
                    id=str(point.id),
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )
        if offset is None:
            return documents


def keyword_search(
    query: str, chunks: list[Document], k: int
) -> list[tuple[Document, float]]:
    """Rank chunks with BM25. Returns only chunks with a positive score."""
    tokens = query_tokens(query)
    if not chunks or not tokens:
        return []
    bm25 = BM25Okapi([tokenize(keyword_text(doc)) for doc in chunks])
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    # A score of 0 means no query term occurs in the chunk: not a match.
    return [(doc, float(score)) for doc, score in ranked[:k] if score > 0]


def _chunk_key(doc: Document) -> str:
    """Stable identity for a chunk, used to merge the two result lists."""
    return doc.metadata.get("chunk_id") or str(doc.id)


def reciprocal_rank_fusion(
    semantic: list[tuple[Document, float]],
    keyword: list[tuple[Document, float]],
    k: int,
) -> list[HybridResult]:
    """Merge two ranked lists: score = sum of 1 / (RRF_K + rank).
    """
    merged: dict[str, HybridResult] = {}

    for rank, (doc, score) in enumerate(semantic, start=1):
        result = merged.setdefault(_chunk_key(doc), HybridResult(doc, 0.0))
        result.rrf_score += 1.0 / (RRF_K + rank)
        result.semantic_rank, result.semantic_score = rank, score
        result.sources.append("semantic")

    for rank, (doc, score) in enumerate(keyword, start=1):
        result = merged.setdefault(_chunk_key(doc), HybridResult(doc, 0.0))
        result.rrf_score += 1.0 / (RRF_K + rank)
        result.keyword_rank, result.keyword_score = rank, score
        result.sources.append("keyword")

    ranked = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)
    return ranked[:k]


def hybrid_search(
    query: str,
    client: QdrantClient,
    settings: Settings,
    embeddings: Embeddings | None,
    k: int = 5,
    policy_id: str | None = None,
    version: str | None = None,
    mode: str = "hybrid",
) -> list[HybridResult]:
    """Search policies using 'hybrid', 'semantic' or 'keyword' mode."""
    query = query.strip()
    if not query:
        raise VectorStoreError("Query cannot be empty")
    if not 1 <= k <= 20:
        raise VectorStoreError("k must be between 1 and 20")
    if bool(policy_id) != bool(version):
        raise VectorStoreError("Provide both policy_id and version, or neither")
    if mode not in {"hybrid", "semantic", "keyword"}:
        raise VectorStoreError(f"Unknown mode: {mode}")

    semantic: list[tuple[Document, float]] = []
    keyword: list[tuple[Document, float]] = []

    if mode in {"hybrid", "semantic"}:
        if embeddings is None:
            raise VectorStoreError(f"Mode '{mode}' needs an embedding client")
        semantic = search_policies(
            query,
            client,
            settings,
            embeddings,
            k=CANDIDATES_PER_RETRIEVER,
            policy_id=policy_id,
            version=version,
        )
    if mode in {"hybrid", "keyword"}:
        chunks = load_indexed_chunks(client, settings, policy_id, version)
        keyword = keyword_search(query, chunks, CANDIDATES_PER_RETRIEVER)

    return reciprocal_rank_fusion(semantic, keyword, k)

def _print_results(results: list[HybridResult], mode: str) -> None:
    if not results:
        print("No matching chunks found.")
        return
    for position, result in enumerate(results, start=1):
        meta = result.document.metadata
        start, end = meta.get("parent_page_start"), meta.get("parent_page_end")
        pages = (
            "no page info"
            if start is None
            else (f"page {start}" if start == end else f"pages {start}-{end}")
        )
        found_by = []
        if result.semantic_rank is not None:
            found_by.append(
                f"semantic #{result.semantic_rank} ({result.semantic_score:.3f})"
            )
        if result.keyword_rank is not None:
            found_by.append(
                f"keyword #{result.keyword_rank} ({result.keyword_score:.3f})"
            )

        print(
            f"\nResult {position} | RRF {result.rrf_score:.4f} | found by: {', '.join(found_by)}"
        )
        print(f"  {meta.get('title')} | {meta.get('policy_id')} v{meta.get('version')}")
        print(
            f"  Section {meta.get('section_number')}: {meta.get('heading')} ({pages})"
        )
        print(f"  {result.document.page_content}")
    print(f"\nMode: {mode}. Scores rank candidates; they do not confirm an answer.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid policy search (semantic + BM25)"
    )
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--policy-id")
    parser.add_argument("--version")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="'keyword' makes no Gemini call",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = None
    try:
        # Keyword-only search must work without any Gemini request.
        embeddings = None if args.mode == "keyword" else build_embeddings(settings)
        client = build_client(settings)
        results = hybrid_search(
            args.query,
            client,
            settings,
            embeddings,
            k=args.k,
            policy_id=args.policy_id,
            version=args.version,
            mode=args.mode,
        )
        _print_results(results, args.mode)
        return 0
    except Exception as exc:  # Short message; never print the API key.
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())

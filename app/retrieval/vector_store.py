import argparse
import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.core.config import Settings, get_settings

MAX_CHUNK_CHARS = 4000


class VectorStoreError(RuntimeError):
    """Readable error for configuration, indexing and search problems."""

def build_embeddings(settings: Settings) -> Embeddings:
    """Create the LangChain Gemini embedding client.
    """
    if settings.gemini_api_key is None:
        raise VectorStoreError("GEMINI_API_KEY is not set in .env")
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embedding_dimensions,
    )


def build_client(settings: Settings) -> QdrantClient:
    """Connect to a Qdrant server if QDRANT_URL is set, else local storage.
    """
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, timeout=60)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Create the collection once, sized for the configured embedding dimension.
    """
    name = settings.qdrant_collection
    if client.collection_exists(name):
        info = client.get_collection(name)
        size = info.config.params.vectors.size
        if size != settings.embedding_dimensions:
            raise VectorStoreError(
                f"Collection {name} stores {size}-dimension vectors, but settings "
                f"request {settings.embedding_dimensions}."
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=settings.embedding_dimensions,
            distance=models.Distance.COSINE,
        ),
    )
    if settings.qdrant_url:
        for field in ("metadata.policy_id", "metadata.version"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


def open_store(
    client: QdrantClient, settings: Settings, embeddings: Embeddings
) -> QdrantVectorStore:
    """Wrap the collection in LangChain's QdrantVectorStore."""
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
        distance=models.Distance.COSINE,
    )


def revision_filter(policy_id: str, version: str) -> models.Filter:
    """Filter that matches one specific policy revision."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.policy_id", match=models.MatchValue(value=policy_id)
            ),
            models.FieldCondition(
                key="metadata.version", match=models.MatchValue(value=version)
            ),
        ]
    )


def _existing_signatures(
    client: QdrantClient, collection: str, policy_id: str, version: str
) -> set[str]:
    """Collect ingestion signatures already stored for one policy revision."""
    signatures: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=revision_filter(policy_id, version),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            signatures.add(metadata.get("ingestion_signature", "<missing>"))
        if offset is None:
            return signatures

def index_policy(
    source: Path,
    client: QdrantClient,
    settings: Settings,
    embeddings: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """Load, chunk, embed and store one policy file. Returns the chunk count.
    """
    # Imported here so 'search' does not need the PDF/DOCX loader packages.
    from app.ingestion.document_loader import load_policy_document
    from app.ingestion.policy_chunker import chunk_policy

    if chunk_size > MAX_CHUNK_CHARS:
        raise VectorStoreError(
            f"chunk_size {chunk_size} exceeds the safe limit of {MAX_CHUNK_CHARS} characters"
        )

    result = load_policy_document(source)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    policy = result.document
    chunks: list[Document] = chunk_policy(policy, chunk_size, chunk_overlap)
    if not chunks:
        raise VectorStoreError("The policy produced no chunks")

    # Fingerprint of content + chunking + embedding settings for this revision.
    signature = hashlib.sha256(
        json.dumps(
            {
                "policy": policy.model_dump(mode="json", round_trip=True),
                "chunk_ids": [c.metadata["chunk_id"] for c in chunks],
                "embedding_model": settings.gemini_embedding_model,
                "dimensions": settings.embedding_dimensions,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    ensure_collection(client, settings)
    existing = _existing_signatures(
        client,
        settings.qdrant_collection,
        policy.metadata.policy_id,
        policy.metadata.version,
    )
    if existing and existing != {signature}:
        raise VectorStoreError(
            f"{policy.metadata.policy_id} v{policy.metadata.version} is already indexed "
            "with different content or chunk settings. Publish it as a new version; "
            "nothing was changed."
        )

    for chunk in chunks:
        chunk.metadata.update(
            embedding_model=settings.gemini_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            ingestion_signature=signature,
        )

    # The check happens BEFORE embedding, so rejected files cost no API calls.
    store = open_store(client, settings, embeddings)
    store.add_documents(
        documents=chunks,
        ids=[c.metadata["chunk_id"] for c in chunks],
        batch_size=64,
    )
    return len(chunks)


def search_policies(
    query: str,
    client: QdrantClient,
    settings: Settings,
    embeddings: Embeddings,
    k: int = 3,
    policy_id: str | None = None,
    version: str | None = None,
) -> list[tuple[Document, float]]:
    """Return the k most similar chunks with their cosine similarity scores.
    """
    query = query.strip()
    if not query:
        raise VectorStoreError("Query cannot be empty")
    if not 1 <= k <= 20:
        raise VectorStoreError("k must be between 1 and 20")
    if bool(policy_id) != bool(version):
        raise VectorStoreError("Provide both policy_id and version, or neither")
    if not client.collection_exists(settings.qdrant_collection):
        raise VectorStoreError("Nothing indexed yet. Run the index command first.")

    store = open_store(client, settings, embeddings)
    # Checking BOTH values narrows them from "str | None" to "str" for the type checker.
    query_filter = (
        revision_filter(policy_id, version) if policy_id and version else None
    )
    return store.similarity_search_with_score(query, k=k, filter=query_filter)


def _print_results(results: list[tuple[Document, float]]) -> None:
    if not results:
        print("No matching chunks found.")
        return
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        start, end = meta.get("parent_page_start"), meta.get("parent_page_end")
        pages = (
            "no page info"
            if start is None
            else (f"page {start}" if start == end else f"pages {start}-{end}")
        )
        print(f"\nResult {rank} | similarity {score:.4f}")
        print(f"  {meta.get('title')} | {meta.get('policy_id')} v{meta.get('version')}")
        print(
            f"  Section {meta.get('section_number')}: {meta.get('heading')} ({pages})"
        )
        print(f"  Source: {meta.get('source_file')}")
        print(f"  {doc.page_content}")
    print("\nNote: similarity is not confidence. No answer has been generated.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index and search policies with Gemini embeddings"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="Embed and store one policy file")
    index_cmd.add_argument("source", type=Path)
    index_cmd.add_argument("--chunk-size", type=int, default=1000)
    index_cmd.add_argument("--chunk-overlap", type=int, default=150)

    search_cmd = sub.add_parser("search", help="Semantic search over stored chunks")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--k", type=int, default=3)
    search_cmd.add_argument("--policy-id")
    search_cmd.add_argument("--version")

    args = parser.parse_args()
    settings = get_settings()
    client = None
    try:
        embeddings = build_embeddings(settings)
        client = build_client(settings)
        print(
            f"Embedding model : {settings.gemini_embedding_model} "
            f"({settings.embedding_dimensions} dimensions)"
        )
        print(f"Collection      : {settings.qdrant_collection}")

        if args.command == "index":
            count = index_policy(
                args.source,
                client,
                settings,
                embeddings,
                args.chunk_size,
                args.chunk_overlap,
            )
            print(f"Indexed {count} chunks from {args.source.name}")
        else:
            results = search_policies(
                args.query,
                client,
                settings,
                embeddings,
                args.k,
                args.policy_id,
                args.version,
            )
            _print_results(results)
        return 0
    except Exception as exc:  # Show a short message; never print the API key.
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())

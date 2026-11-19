import argparse
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document_loader import load_policy_document
from app.ingestion.policy_loader import PolicyLoadError
from app.schemas.policy import PolicyDocument


def chunk_policy(
    policy: PolicyDocument,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split each clause independently, preserving its parent metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    # Revalidate in case a caller modified a nested list after construction.
    policy = PolicyDocument.model_validate_json(policy.model_dump_json(round_trip=True))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )
    results: list[Document] = []
    policy_metadata = policy.metadata.model_dump(mode="json")

    for clause in policy.clauses:
        metadata = {
            **policy_metadata,
            "clause_id": str(clause.clause_id),
            "section_number": clause.section_number,
            "heading": clause.heading,
            "heading_path": list(clause.heading_path),
            "parent_page_start": clause.page_number,
            "parent_page_end": clause.page_end or clause.page_number,
            "related_sections": [
                reference.model_dump(mode="json")
                for reference in clause.related_sections
            ],
        }
        # Split only this clause's text; never merge unrelated sections.
        pieces = splitter.split_documents(
            [Document(page_content=clause.text, metadata=metadata)]
        )
        for index, piece in enumerate(pieces):
            identity = json.dumps(
                [
                    str(clause.clause_id),
                    chunk_size,
                    chunk_overlap,
                    index,
                    hashlib.sha256(piece.page_content.encode("utf-8")).hexdigest(),
                ],
                separators=(",", ":"),
            )
            chunk_id = str(uuid5(NAMESPACE_URL, "policypilot:chunk:" + identity))
            piece.id = chunk_id
            piece.metadata.update(
                chunk_id=chunk_id,
                chunk_index=index,
                chunk_count=len(pieces),
                chunk_size_chars=chunk_size,
                chunk_overlap_chars=chunk_overlap,
            )
            results.append(piece)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Create clause-aware LangChain chunks")
    parser.add_argument("source", type=Path, help="PDF, DOCX, TXT, MD or JSON policy")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--output", type=Path, help="Optional new .json chunk export")
    args = parser.parse_args()

    try:
        result = load_policy_document(args.source)
        policy = result.document
        chunks = chunk_policy(policy, args.chunk_size, args.chunk_overlap)
        if args.output:
            if args.output.suffix.lower() != ".json":
                raise ValueError("Output must use the .json extension")
            # A chunk dataset, not a PolicyDocument input file.
            payload = {
                "format": "policypilot.chunks.v1",
                "chunks": [
                    {"id": c.id, "page_content": c.page_content, "metadata": c.metadata}
                    for c in chunks
                ],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:  # never overwrite
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(f"Saved chunks: {args.output}")
    except (PolicyLoadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Policy: {policy.metadata.policy_id} v{policy.metadata.version}")
    print(f"Clauses: {len(policy.clauses)}")
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        print(
            f"Section {chunk.metadata['section_number']} | "
            f"Chunk {chunk.metadata['chunk_index'] + 1}/{chunk.metadata['chunk_count']} | "
            f"{len(chunk.page_content)} characters"
        )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

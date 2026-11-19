import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*langchain-community.*sunset.*",
        category=DeprecationWarning,
    )
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )
from langchain_core.documents import Document
from pydantic import ValidationError

from app.ingestion.metadata_extractor import MetadataError, extract_metadata
from app.ingestion.policy_loader import PolicyLoadError, load_policy
from app.schemas.policy import (
    ClauseReference,
    PolicyClause,
    PolicyDocument,
    PolicyMetadata,
)

HEADING_PATTERN = re.compile(
    r"^(?:section\s+)?(?P<number>\d+(?:\.\d+)*)[.):]?\s+(?P<title>[A-Za-z][^\n]{0,119})$",
    re.IGNORECASE,
)
DATE_LIKE = re.compile(
    r"^\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}$",
    re.IGNORECASE,
)

REFERENCE_PATTERN = re.compile(r"\bsections?\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".json"}


@dataclass
class LoadResult:
    """Validated policy plus notes on how it was parsed."""

    document: PolicyDocument
    warnings: list[str] = field(default_factory=list)
    metadata_sources: dict[str, str] = field(default_factory=dict)  # field -> origin


def load_source_pages(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8-sig")
    else:
        raise PolicyLoadError(f"Unsupported file type: {suffix}")

    try:
        pages = loader.load()
    except Exception as exc:  # loader libraries raise many different types
        raise PolicyLoadError(
            f"Could not read {path.name}: {type(exc).__name__}"
        ) from exc

    if not any(page.page_content.strip() for page in pages):
        raise PolicyLoadError(
            f"No extractable text found in {path.name}. "
            "Scanned documents need OCR, which is not supported yet."
        )
    return pages

def load_overrides(path: Path) -> dict[str, Any]:
    """Read '<name>.meta.json' if present. Its values override the document's."""
    sidecar = path.with_name(path.stem + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyLoadError(f"Cannot read {sidecar.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyLoadError(f"{sidecar.name} must contain a JSON object")
    if "source_file" in raw:
        raise PolicyLoadError(
            f"{sidecar.name} must not set source_file; it is taken from the file name"
        )
    unknown = set(raw) - set(PolicyMetadata.model_fields)
    if unknown:
        raise PolicyLoadError(
            f"{sidecar.name} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return raw

@dataclass
class _Section:
    number: str
    title: str
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def _iter_lines(pages: list[Document]):
    """Yield (line, one-based page number or None) across all pages."""
    for page in pages:
        raw_page = page.metadata.get("page")
        page_number = raw_page + 1 if isinstance(raw_page, int) else None
        for line in page.page_content.splitlines():
            yield line.strip(), page_number


def _is_heading(line: str) -> re.Match | None:
    if DATE_LIKE.match(line) or line.rstrip().endswith((".", ";", ",")):
        return None
    return HEADING_PATTERN.match(line)


def extract_sections(pages: list[Document]) -> tuple[list[str], list[_Section]]:
    """Return (header lines before the first numbered heading, sections)."""
    header: list[str] = []
    sections: list[_Section] = []
    current: _Section | None = None

    for line, page_number in _iter_lines(pages):
        if not line:
            continue
        match = _is_heading(line)
        if match:
            current = _Section(
                number=match.group("number"), title=match.group("title").strip()
            )
            if page_number is not None:
                current.pages.append(page_number)
            sections.append(current)
        elif current is None:
            header.append(line)  # title, metadata fields, introduction
        else:
            current.lines.append(line)
            if page_number is not None:
                current.pages.append(page_number)
    return header, sections


def _join_lines(lines: list[str]) -> str:
    """Join extracted lines, repairing words hyphenated across a line break."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", "\n".join(lines)).strip()


def build_policy_document(
    metadata: PolicyMetadata, sections: list[_Section]
) -> tuple[PolicyDocument, list[str]]:
    notes: list[str] = []
    known_numbers = {s.number for s in sections}
    titles = {s.number: s.title for s in sections}
    clauses: list[PolicyClause] = []

    for section in sections:
        text = _join_lines(section.lines)
        if not text:
            # A parent heading such as "3 Working Arrangements"; its title still
            # appears in the heading path of its children (3.1, 3.2).
            notes.append(
                f"Section {section.number} has no body text and was not stored as a clause"
            )
            continue

        parts = section.number.split(".")
        parents = [
            titles[".".join(parts[:i])]
            for i in range(1, len(parts))
            if ".".join(parts[:i]) in titles
        ]

        # Link only to sections in this document, so phrases like
        # "section 80C of the Income Tax Act" do not create false links.
        referenced: list[str] = []
        for target in REFERENCE_PATTERN.findall(text):
            if (
                target in known_numbers
                and target != section.number
                and target not in referenced
            ):
                referenced.append(target)

        clauses.append(
            PolicyClause(
                policy_id=metadata.policy_id,
                version=metadata.version,
                section_number=section.number,
                heading=section.title,
                heading_path=[metadata.title, *parents, section.title],
                text=text,
                page_number=min(section.pages) if section.pages else None,
                page_end=(
                    max(section.pages)
                    if section.pages and max(section.pages) != min(section.pages)
                    else None
                ),
                related_sections=[
                    ClauseReference(
                        policy_id=metadata.policy_id,
                        version=metadata.version,
                        section_number=n,
                    )
                    for n in referenced
                ],
            )
        )

    if not clauses:
        raise PolicyLoadError(
            "No numbered sections with text were found. "
            "Headings must look like '1 Scope', '2. Leave' or 'Section 3: Approval'."
        )
    try:
        return PolicyDocument(metadata=metadata, clauses=clauses), notes
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors(include_input=False, include_url=False)
        )
        raise PolicyLoadError(f"Validation failed: {details}") from exc


def load_policy_document(path: str | Path) -> LoadResult:
    """Load any supported policy file and return a validated PolicyDocument."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise PolicyLoadError(
            f"Unsupported file type '{suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )
    if not source.is_file():
        raise PolicyLoadError(f"File not found: {source}")
    if suffix == ".json":
        return LoadResult(
            document=load_policy(source), metadata_sources={"all": "structured JSON"}
        )

    overrides = load_overrides(source)
    pages = load_source_pages(source)
    header, sections = extract_sections(pages)
    try:
        extracted = extract_metadata(
            header_lines=header,
            source_file=source.name,
            pdf_properties=pages[0].metadata if suffix == ".pdf" else None,
            overrides=overrides,
        )
    except MetadataError as exc:
        raise PolicyLoadError(f"{source.name}: {exc}") from exc

    document, notes = build_policy_document(extracted.metadata, sections)
    return LoadResult(
        document=document,
        warnings=extracted.warnings + notes,
        metadata_sources=extracted.sources,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a policy file with LangChain")
    parser.add_argument("source", type=Path, help="PDF, DOCX, TXT, MD or JSON policy")
    args = parser.parse_args()
    try:
        result = load_policy_document(args.source)
    except PolicyLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    meta = result.document.metadata
    print("Policy loaded successfully\n\nMetadata (value <- source):")
    for name in (
        "policy_id",
        "title",
        "version",
        "effective_from",
        "effective_to",
        "applies_to",
        "departments",
        "jurisdiction",
        "is_synthetic",
    ):
        value = getattr(meta, name)
        source = result.metadata_sources.get(name, "not found")
        print(f"  {name:15} {str(value):40} <- {source}")

    print(f"\nClauses: {len(result.document.clauses)}")
    for clause in result.document.clauses:
        if clause.page_number is None:
            pages = "no page info"
        elif clause.page_end:
            pages = f"pages {clause.page_number}-{clause.page_end}"
        else:
            pages = f"page {clause.page_number}"
        refs = ", ".join(r.section_number for r in clause.related_sections)
        print(
            f"  Section {clause.section_number}: {clause.heading} ({pages})"
            + (f"  -> references section {refs}" if refs else "")
        )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

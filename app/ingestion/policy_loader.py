import argparse
from pathlib import Path

from pydantic import ValidationError

from app.schemas.policy import PolicyDocument


class PolicyLoadError(ValueError):
    """Readable error for an unreadable or invalid policy input."""


def load_policy(path: str | Path) -> PolicyDocument:
    """Read one structured JSON policy and validate it against our schemas.
    """
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise PolicyLoadError("Expected a .json policy file.")
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PolicyLoadError(f"Cannot read policy file: {source}") from exc

    try:
        return PolicyDocument.model_validate_json(text)
    except ValidationError as exc:
        # Do not include source policy text in diagnostic messages.
        details = []
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "document"
            details.append(f"{location}: {error['msg']}")
        raise PolicyLoadError(
            "Policy validation failed:\n" + "\n".join(details)
        ) from exc


def save_policy(document: PolicyDocument, path: str | Path) -> Path:
    """Export reloadable JSON without overwriting an existing file.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("Output must be a .json file.")

    # Revalidate to catch in-place list edits, which assignment validation
    # alone cannot intercept. This does not enforce cross-file version rules.
    validated = PolicyDocument.model_validate_json(
        document.model_dump_json(round_trip=True)
    )
    payload = validated.model_dump_json(indent=2, round_trip=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a policy revision.
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a structured policy JSON file."
    )
    parser.add_argument("source", type=Path, help="Path to a policy JSON file")
    parser.add_argument("--output", type=Path, help="Optional new JSON output path")
    args = parser.parse_args()

    try:
        document = load_policy(args.source)
        if args.output:
            saved = save_policy(document, args.output)
            print(f"Saved reloadable policy: {saved}")
    except (PolicyLoadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Policy loaded successfully")
    print(f"Policy: {document.metadata.policy_id}")
    print(f"Version: {document.metadata.version}")
    print(f"Clauses: {len(document.clauses)}")
    for clause in document.clauses:
        print(f"  Section {clause.section_number}: {clause.heading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

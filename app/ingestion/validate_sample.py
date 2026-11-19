import sys
from pathlib import Path

from pydantic import ValidationError

from app.schemas.policy import PolicyDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    source = PROJECT_ROOT / "data" / "policies" / "synthetic_leave_v1.json"
    try:
        # Read the original fixture, not a serialized model with computed fields.
        document = PolicyDocument.model_validate_json(
            source.read_text(encoding="utf-8-sig")
        )
    except OSError as exc:
        print(f"Cannot read policy fixture: {exc}")
        return 1
    except ValidationError as exc:
        print("Policy validation failed:")
        print(exc)
        return 1

    print("Policy validation successful")
    print(f"Policy: {document.metadata.policy_id}")
    print(f"Version: {document.metadata.version}")
    print(f"Synthetic: {document.metadata.is_synthetic}")
    print(f"Clauses: {len(document.clauses)}")
    for clause in document.clauses:
        print(f"  Section {clause.section_number}: {clause.heading}")
        print(f"    Clause ID: {clause.clause_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

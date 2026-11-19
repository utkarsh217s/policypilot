import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.schemas.policy import PolicyMetadata


class MetadataError(ValueError):
    """Required metadata is missing or invalid."""

FIELD_LABELS: dict[str, list[str]] = {
    "policy_id": [
        "policy id",
        "policy no",
        "policy number",
        "policy code",
        "document id",
        "document no",
        "document number",
        "doc id",
    ],
    "title": ["policy title", "policy name", "document title", "title"],
    "version": ["version number", "version no", "version", "revision"],
    "effective_from": [
        "effective date",
        "effective from",
        "valid from",
        "in force from",
        "date of effect",
    ],
    "effective_to": [
        "effective to",
        "effective until",
        "valid until",
        "valid to",
        "expiry date",
        "superseded on",
        "end date",
    ],
    "applies_to": ["applies to", "applicable to", "applicability"],
    "departments": ["departments", "department", "business units", "business unit"],
    "jurisdiction": ["jurisdiction", "country"],
    "classification": ["classification", "document type"],
}

_LABEL_TO_FIELD = {
    label: name for name, labels in FIELD_LABELS.items() for label in labels
}
_LABEL_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(l) for l in sorted(_LABEL_TO_FIELD, key=len, reverse=True))
    + r")\s*:\s*",
    re.IGNORECASE,
)

MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}

SYNTHETIC_WORDS = re.compile(
    r"\b(synthetic|fictional|sample data|test data)\b", re.IGNORECASE
)

# PDF titles that are tool defaults, not real titles.
_PLACEHOLDER_TITLE = re.compile(
    r"^(untitled|document\d*|microsoft word - .*|.*\.(docx?|pdf))$", re.IGNORECASE
)


@dataclass
class ExtractedMetadata:
    metadata: PolicyMetadata
    sources: dict[str, str] = field(default_factory=dict)  # field -> where it came from
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def find_labelled_fields(lines: list[str]) -> dict[str, str]:
    """Return the first value found for each field in the header lines."""
    found: dict[str, str] = {}
    for index, line in enumerate(lines):
        matches = list(_LABEL_PATTERN.finditer(line))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            value = line[match.end() : end].strip().strip(";,|").strip()
            # PDF text extraction sometimes puts the value on the next line:
            # "Effective Date:" / "1 April 2026".
            is_last = i + 1 == len(matches)
            if (
                not value
                and is_last
                and index + 1 < len(lines)
                and not _LABEL_PATTERN.search(lines[index + 1])
            ):
                value = lines[index + 1].strip()
            name = _LABEL_TO_FIELD[match.group(1).lower()]
            if value and name not in found:
                found[name] = value
    return found


def parse_date(text: str) -> tuple[date | None, str | None]:
    """Parse common date formats. Returns (date, warning).
    """
    value = re.sub(
        r"(\d)(st|nd|rd|th)\b", r"\1", text.strip().rstrip("."), flags=re.IGNORECASE
    )
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value):
            return date(int(m[1]), int(m[2]), int(m[3])), None

        if m := re.fullmatch(r"(\d{1,2})[\s-]+([A-Za-z]+)[\s,-]+(\d{4})", value):
            if m[2].lower() in MONTHS:
                return date(int(m[3]), MONTHS[m[2].lower()], int(m[1])), None

        if m := re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", value):
            if m[1].lower() in MONTHS:
                return date(int(m[3]), MONTHS[m[1].lower()], int(m[2])), None

        if m := re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", value):
            first, second, year = int(m[1]), int(m[2]), int(m[3])
            if first > 12:
                return date(year, second, first), None
            if second > 12:

                return date(year, first, second), None
            warning = None
            if first != second:
                warning = (
                    f"Date '{text}' is ambiguous; read as day/month/year "
                    f"({date(year, second, first).isoformat()}). Use YYYY-MM-DD to avoid this."
                )
            return date(year, second, first), warning
    except ValueError:
        pass  # e.g. 31 February
    return None, f"Could not read date '{text}'; it was left unknown. Use YYYY-MM-DD."


def split_list(text: str) -> list[str]:
    """'Engineering, Finance; HR' -> ['Engineering', 'Finance', 'HR'].

    Splits only on commas and semicolons. Splitting on 'and' would break
    values like 'Permanent and contract employees' into wrong pieces.
    """
    return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


def normalise_version(text: str) -> str:
    """'v1.0' / 'Version 1.0' -> '1.0'. Other text is kept as written."""
    return re.sub(r"^(?:v|ver\.?)\s*(?=\d)", "", text.strip(), flags=re.IGNORECASE)


def slugify_policy_id(title: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", title.upper()).strip("-")[:60]


def _is_label_line(line: str) -> bool:
    return bool(_LABEL_PATTERN.search(line))

def extract_metadata(
    header_lines: list[str],
    source_file: str,
    pdf_properties: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExtractedMetadata:
    """Build validated PolicyMetadata from the document header.
    """
    pdf_properties = pdf_properties or {}
    overrides = dict(overrides or {})
    labelled = find_labelled_fields(header_lines)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []

    def take(name: str, value: Any, source: str) -> None:
        if name not in values and value not in (None, "", []):
            values[name] = value
            sources[name] = source

    for name, value in overrides.items():
        take(name, value, "metadata file")

    for name in ("policy_id", "title", "jurisdiction"):
        take(name, labelled.get(name), "document header")
    if "version" in labelled:
        take("version", normalise_version(labelled["version"]), "document header")
    for name in ("applies_to", "departments"):
        if name in labelled:
            take(name, split_list(labelled[name]), "document header")
    for name in ("effective_from", "effective_to"):
        if name in labelled and name not in values:
            parsed, warning = parse_date(labelled[name])
            if warning:
                warnings.append(f"{name}: {warning}")
            take(name, parsed, "document header")

    pdf_title = str(pdf_properties.get("title") or "").strip()
    if pdf_title and not _PLACEHOLDER_TITLE.match(pdf_title):
        take("title", pdf_title, "PDF properties")

    # Fallbacks
    if "title" not in values:
        first_line = next(
            (l.strip() for l in header_lines if l.strip() and not _is_label_line(l)),
            None,
        )
        take("title", first_line, "first line of document")
    if "policy_id" not in values and "title" in values:
        take("policy_id", slugify_policy_id(values["title"]), "derived from title")
        warnings.append(
            f"No 'Policy ID:' found; using '{values['policy_id']}' derived from the title. "
            "Changing the title in a later revision would change this ID."
        )

    if "is_synthetic" not in values:
        header_text = "\n".join(header_lines)
        if SYNTHETIC_WORDS.search(
            labelled.get("classification", "")
        ) or SYNTHETIC_WORDS.search(header_text):
            take("is_synthetic", True, "document header mentions synthetic/fictional")

    # Required fields: report everything missing at once.
    missing = [name for name in ("title", "version") if name not in values]
    if missing:
        labels = {"title": "Title:", "version": "Version:"}
        raise MetadataError(
            "Could not find "
            + " and ".join(missing)
            + " in the document. Add a header line such as "
            + ", ".join(f"'{labels[m]} ...'" for m in missing)
            + " before the first numbered section."
        )

    try:
        metadata = PolicyMetadata(**values, source_file=source_file)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'value'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
        raise MetadataError(f"Invalid metadata: {details}") from exc

    for name in ("effective_from", "applies_to"):
        if name not in values:
            warnings.append(f"No {name} found; it is recorded as unknown.")
    return ExtractedMetadata(metadata=metadata, sources=sources, warnings=warnings)

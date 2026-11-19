from datetime import date
from typing import Annotated, Self
from uuid import NAMESPACE_URL, UUID, uuid5
import json

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PolicyBase(BaseModel):
    """Reject unknown fields and validate assignments as well as initial input."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PolicyMetadata(PolicyBase):
    """Metadata for one specific revision of a policy.
    """

    policy_id: NonEmptyText
    title: NonEmptyText
    version: NonEmptyText
    source_file: NonEmptyText
    effective_from: date | None = None
    effective_to: date | None = None
    departments: list[NonEmptyText] = Field(default_factory=list)
    applies_to: list[NonEmptyText] = Field(default_factory=list)
    jurisdiction: NonEmptyText | None = None
    is_synthetic: bool = False

    @model_validator(mode="after")
    def validate_effective_dates(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to <= self.effective_from
        ):
            raise ValueError("effective_to must be later than effective_from")
        return self


class ClauseReference(PolicyBase):
    """Version-aware reference, including targets not ingested yet."""

    policy_id: NonEmptyText
    version: NonEmptyText
    section_number: NonEmptyText


class PolicyClause(PolicyBase):
    """One logical source clause, before retrieval-specific chunking.
    """

    policy_id: NonEmptyText
    version: NonEmptyText
    section_number: NonEmptyText
    heading: NonEmptyText
    heading_path: list[NonEmptyText] = Field(default_factory=list)
    text: NonEmptyText
    page_number: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    related_sections: list[ClauseReference] = Field(default_factory=list)

    @computed_field
    @property
    def clause_id(self) -> UUID:
        """Deterministic ID for a policy/version/section, independent of text.
        """
        identity = json.dumps(
            [self.policy_id, self.version, self.section_number],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return uuid5(NAMESPACE_URL, "policypilot:clause:" + identity)

    @model_validator(mode="after")
    def validate_page_range(self) -> Self:
        if self.page_end is not None:
            if self.page_number is None:
                raise ValueError("page_number is required when page_end is provided")
            if self.page_end < self.page_number:
                raise ValueError("page_end cannot be earlier than page_number")
        return self


class PolicyDocument(PolicyBase):
    """Validated ingestion result for one policy revision.
    """

    metadata: PolicyMetadata
    clauses: list[PolicyClause] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_clauses(self) -> Self:
        sections: set[str] = set()
        for clause in self.clauses:
            if clause.policy_id != self.metadata.policy_id:
                raise ValueError("Clause policy_id must match document metadata")
            if clause.version != self.metadata.version:
                raise ValueError("Clause version must match document metadata")
            if clause.section_number in sections:
                raise ValueError(f"Duplicate section_number: {clause.section_number}")
            sections.add(clause.section_number)
        return self

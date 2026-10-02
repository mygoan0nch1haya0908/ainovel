"""Local memory data contracts; memory text is never an agent instruction."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceRef(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    project_id: str
    source_type: str
    source_id: str
    source_version: str
    content_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    field_path: str
    excerpt_start: int | None = Field(default=None, ge=0)
    excerpt_end: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def validate_range(self):
        if (self.excerpt_start is None) != (self.excerpt_end is None):
            raise ValueError('excerpt bounds must be paired')
        if self.excerpt_start is not None and self.excerpt_end <= self.excerpt_start:
            raise ValueError('excerpt range must be nonempty')
        return self


class MemoryEntryInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entry_id: str | None = None
    origin: Literal['manual', 'ai'] = 'manual'
    author_locked: bool = False
    kind: Literal['rule', 'character', 'fact', 'foreshadowing', 'summary']
    text: str = Field(min_length=1)
    source_refs: list[SourceRef] = Field(min_length=1)
    entity_ids: list[str] = Field(default_factory=list)
    point_ids: list[str] = Field(default_factory=list)
    effective_from: int = Field(default=1, ge=1)
    effective_until: int | None = Field(default=None, ge=1)
    reveal_from: int | None = Field(default=None, ge=1)
    audience: Literal['author_only', 'narratable'] = 'author_only'

    @model_validator(mode='after')
    def validate_range(self):
        if not self.text.strip():
            raise ValueError('memory text must not be blank')
        if self.effective_until is not None and self.effective_until < self.effective_from:
            raise ValueError('effective range is reversed')
        return self


class MemorySelection(BaseModel):
    required: list[dict[str, Any]] = Field(default_factory=list)
    optional: list[dict[str, Any]] = Field(default_factory=list)
    excluded: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    source_fingerprint: str


class ContextPreview(BaseModel):
    workflow_id: str
    step_id: str
    policy_version: int
    source_fingerprint: str
    preview_fingerprint: str = ''
    legacy_estimated_tokens: int
    estimated_tokens: int
    capacity: int
    deficit: int
    items: list[dict[str, Any]]
    blockers: list[str]

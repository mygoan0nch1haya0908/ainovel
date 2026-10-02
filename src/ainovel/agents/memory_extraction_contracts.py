from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5
from pydantic import BaseModel, ConfigDict, Field
from ainovel.agents.memory_contracts import MemoryEntryInput


def identified_entries(card_id: str, entries: list[dict]) -> list[dict]:
    result = []
    for index, raw in enumerate(entries):
        entry = MemoryEntryInput.model_validate(deepcopy(raw)).model_dump()
        entry['entry_id'] = entry['entry_id'] or str(uuid5(NAMESPACE_URL, f'ainovel:memory:{card_id}:{index}'))
        result.append(entry)
    return result


class ExtractionReference(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ExtractionEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: str
    text: str = Field(min_length=1)
    references: list[ExtractionReference] = Field(min_length=1)
    entity_ids: list[str] = Field(default_factory=list)
    point_ids: list[str] = Field(default_factory=list)
    effective_from: int = Field(default=1, ge=1)
    effective_until: int | None = None
    reveal_from: int | None = None
    audience: str = 'author_only'
    replaces_entry_id: str | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entries: list[ExtractionEntry]
    unresolved: list[str]

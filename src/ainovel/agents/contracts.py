from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ainovel.services.counting import count_visible_characters


class AgentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChapterPlan(AgentSchema):
    ordinal: int = Field(ge=1)
    title: str
    goal: str
    ending_hook: str

    @field_validator("title", "goal", "ending_hook")
    @classmethod
    def text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be nonblank")
        return value


class BatchPlanDraft(AgentSchema):
    chapters: list[ChapterPlan] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def ordinals_are_contiguous(self) -> "BatchPlanDraft":
        if [chapter.ordinal for chapter in self.chapters] != list(range(1, len(self.chapters) + 1)):
            raise ValueError("chapter ordinals must be contiguous from 1")
        return self


class ChapterDraft(AgentSchema):
    title: str
    body: str

    @field_validator("title", "body")
    @classmethod
    def text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be nonblank")
        return value

    @field_validator("body")
    @classmethod
    def body_has_valid_visible_character_count(cls, value: str) -> str:
        visible_count = count_visible_characters(value)
        if not 4500 <= visible_count <= 6000:
            raise ValueError("body must contain between 4500 and 6000 visible characters")
        return value

class ChapterSummaryDelta(AgentSchema):
    summary: str
    state_delta: dict[str, object]

    @field_validator("summary")
    @classmethod
    def summary_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must be nonblank")
        return value


class BatchReview(AgentSchema):
    passed: bool
    issues: list[str]
    evidence_queries: list[str]

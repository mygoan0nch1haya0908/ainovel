from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator, model_validator
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


class ChapterScenePlan(AgentSchema):
    ordinal: int = Field(ge=1)
    description: str
    target_characters: int = Field(gt=0)

    @field_validator("description")
    @classmethod
    def description_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene description must be nonblank")
        return value


class ChapterPlanV2(ChapterPlan):
    scenes: list[ChapterScenePlan] = Field(min_length=1)

    @model_validator(mode="after")
    def scenes_are_ordered_and_budgeted(self) -> "ChapterPlanV2":
        if [scene.ordinal for scene in self.scenes] != list(
            range(1, len(self.scenes) + 1)
        ):
            raise ValueError("scene ordinals must be contiguous from 1")
        total = sum(scene.target_characters for scene in self.scenes)
        if not 4500 <= total <= 6000:
            raise ValueError("scene target characters must total between 4500 and 6000")
        return self


class BatchPlanDraft(AgentSchema):
    chapters: list[ChapterPlan] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def ordinals_are_contiguous(self) -> "BatchPlanDraft":
        if [chapter.ordinal for chapter in self.chapters] != list(range(1, len(self.chapters) + 1)):
            raise ValueError("chapter ordinals must be contiguous from 1")
        return self


class BatchPlanDraftV2(AgentSchema):
    chapters: list[ChapterPlanV2] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def ordinals_are_contiguous(self) -> "BatchPlanDraftV2":
        if [chapter.ordinal for chapter in self.chapters] != list(
            range(1, len(self.chapters) + 1)
        ):
            raise ValueError("chapter ordinals must be contiguous from 1")
        return self


class WorkChapterDraft(AgentSchema):
    title: str
    body: str

    @field_validator("title", "body")
    @classmethod
    def text_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be nonblank")
        return value


class CoverageVerdict(AgentSchema):
    passed: bool
    excerpt: str


class ChapterCoverage(AgentSchema):
    goal: CoverageVerdict
    ending_hook: CoverageVerdict


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
    # The wire contract is a JSON string so strict providers can retain arbitrary
    # nested keys without an open object schema. Persistence remains a domain dict.
    state_delta: Annotated[
        dict[str, object],
        WithJsonSchema({
            "type": "string",
            "description": "A JSON-encoded object containing all state changes, including arbitrary nested keys and values. Use \"{}\" when empty.",
        }, mode="validation"),
    ]

    @field_validator("state_delta", mode="before")
    @classmethod
    def decode_wire_state_delta(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError("state delta must encode a JSON object") from None
        return value

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

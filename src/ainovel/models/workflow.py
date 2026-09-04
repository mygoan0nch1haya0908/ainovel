from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from ainovel.models.base import Base, TimestampMixin


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTCDateTime requires a timezone-aware datetime")
        normalized = value.astimezone(timezone.utc)
        if dialect.name == "sqlite":
            return normalized.replace(tzinfo=None)
        return normalized

    def process_result_value(self, value: datetime | None, _dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class GenerationWorkflow(TimestampMixin, Base):
    __tablename__ = "generation_workflows"
    __table_args__ = (Index("ix_generation_workflows_project_status", "project_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    base_outline_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("outline_versions.id"), nullable=False
    )
    provider_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_chapters: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    current_position: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    candidate_batch_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("writing_batches.id"), nullable=True
    )
    planner_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    planner_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    writer_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    writer_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    summarizer_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    summarizer_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewer_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewer_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    actual_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkflowStep(Base):
    __tablename__ = "workflow_steps"
    __table_args__ = (
        UniqueConstraint("workflow_id", "position"),
        UniqueConstraint("workflow_id", "kind", "ordinal"),
        CheckConstraint(
            "(lease_owner IS NULL AND lease_expires_at IS NULL) OR "
            "(lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_workflow_steps_lease_pair",
        ),
        Index(
            "uq_workflow_steps_one_null_ordinal_kind",
            "workflow_id",
            "kind",
            unique=True,
            sqlite_where=text("ordinal IS NULL"),
        ),
        Index("ix_workflow_steps_workflow_status", "workflow_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_workflows.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    active_artifact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )


class ModelAttempt(TimestampMixin, Base):
    __tablename__ = "model_attempts"
    __table_args__ = (UniqueConstraint("step_id", "attempt_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    step_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflow_steps.id"), nullable=False
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_response_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkflowArtifact(TimestampMixin, Base):
    __tablename__ = "workflow_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_workflows.id"), nullable=False
    )
    step_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflow_steps.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    visible_char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class PlanDecision(TimestampMixin, Base):
    __tablename__ = "plan_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_workflows.id"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)

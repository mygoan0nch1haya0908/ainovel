from sqlalchemy import ForeignKey, Integer, JSON, String, Text, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from ainovel.models.base import Base, TimestampMixin


class StoryStage(TimestampMixin, Base):
    __tablename__ = "story_stages"
    __table_args__ = (CheckConstraint("confirmed_chapters >= 0"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("novel_projects.id"))
    base_outline_version_id: Mapped[str] = mapped_column(ForeignKey("outline_versions.id"))
    architecture: Mapped[str] = mapped_column(Text)
    approved_roadmap_id: Mapped[str | None] = mapped_column(String(36))
    confirmed_chapters: Mapped[int] = mapped_column(Integer, default=0)
    revision: Mapped[int] = mapped_column(Integer, default=1)


class StageRoadmapVersion(TimestampMixin, Base):
    __tablename__ = "stage_roadmap_versions"
    __table_args__ = (UniqueConstraint("stage_id", "version_number"),
                      CheckConstraint("attempts_used >= 0 AND attempts_used <= attempt_limit"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    stage_id: Mapped[str] = mapped_column(ForeignKey("story_stages.id"))
    version_number: Mapped[int] = mapped_column(Integer)
    input_revision: Mapped[int] = mapped_column(Integer)
    constitution_version_id: Mapped[str] = mapped_column(ForeignKey("constitution_versions.id"))
    status: Mapped[str] = mapped_column(String(40), default="PENDING")
    provider_name: Mapped[str] = mapped_column(String(40))
    model_name: Mapped[str] = mapped_column(String(255))
    architecture: Mapped[str] = mapped_column(Text)
    prompt_snapshot: Mapped[dict] = mapped_column(JSON)
    input_snapshot: Mapped[dict] = mapped_column(JSON)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts_used: Mapped[int] = mapped_column(Integer, default=0)
    attempt_limit: Mapped[int] = mapped_column(Integer, default=2)
    input_token_limit: Mapped[int] = mapped_column(Integer)
    output_token_limit: Mapped[int] = mapped_column(Integer)
    total_input_token_limit: Mapped[int] = mapped_column(Integer)
    total_output_token_limit: Mapped[int] = mapped_column(Integer)
    actual_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    actual_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    approved_by: Mapped[str | None] = mapped_column(String(255))

    @property
    def estimated_chapters(self):
        return len(self.payload["nodes"]) if self.payload else None


class StageModelAttempt(TimestampMixin, Base):
    __tablename__ = "stage_model_attempts"
    __table_args__ = (UniqueConstraint("roadmap_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    roadmap_id: Mapped[str] = mapped_column(ForeignKey("stage_roadmap_versions.id"))
    number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(80))


class StageWorkflow(TimestampMixin, Base):
    __tablename__ = "stage_workflows"
    workflow_id: Mapped[str] = mapped_column(ForeignKey("generation_workflows.id"), primary_key=True)
    stage_id: Mapped[str] = mapped_column(ForeignKey("story_stages.id"))
    roadmap_id: Mapped[str] = mapped_column(ForeignKey("stage_roadmap_versions.id"))
    confirmed_start: Mapped[int] = mapped_column(Integer)
    committed_batch_id: Mapped[str | None] = mapped_column(ForeignKey("writing_batches.id"), unique=True)


class StageWorkflowNode(Base):
    __tablename__ = "stage_workflow_nodes"
    __table_args__ = (UniqueConstraint("workflow_id", "node_id"),
                      UniqueConstraint("workflow_id", "stage_ordinal"),
                      CheckConstraint("ordinal >= 1 AND ordinal <= 5 AND stage_ordinal >= 1 AND book_ordinal >= 1"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("stage_workflows.workflow_id"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64))
    stage_ordinal: Mapped[int] = mapped_column(Integer)
    book_ordinal: Mapped[int] = mapped_column(Integer)

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class PromptVersion(TimestampMixin, Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("role", "version_number"),
        Index(
            "uq_prompt_versions_one_active_role",
            "role",
            unique=True,
            sqlite_where=text("active = 1"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)


class WorkflowPromptSnapshot(TimestampMixin, Base):
    __tablename__ = "workflow_prompt_snapshots"
    __table_args__ = (UniqueConstraint("workflow_id", "role"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_workflows.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompt_versions.id"), nullable=False
    )
    prompt_body: Mapped[str] = mapped_column(Text, nullable=False)
    output_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

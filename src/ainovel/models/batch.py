from __future__ import annotations

from typing import Any

from sqlalchemy import ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class WritingBatch(TimestampMixin, Base):
    __tablename__ = "writing_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    base_outline_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("outline_versions.id"), nullable=False
    )
    planned_chapters: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class Chapter(TimestampMixin, Base):
    __tablename__ = "chapters"
    __table_args__ = (
        UniqueConstraint("batch_id", "ordinal"),
        UniqueConstraint("project_id", "official_chapter_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("writing_batches.id"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    visible_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state_delta: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    official_chapter_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class NovelProject(TimestampMixin, Base):
    __tablename__ = "novel_projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    target_chars_min: Mapped[int] = mapped_column(Integer, nullable=False)
    target_chars_max: Mapped[int] = mapped_column(Integer, nullable=False)
    official_outline_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    current_constitution_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    active_batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    next_batch_sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    next_official_chapter_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )


class ConstitutionVersion(TimestampMixin, Base):
    __tablename__ = "constitution_versions"
    __table_args__ = (UniqueConstraint("project_id", "version_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    author_approved: Mapped[bool] = mapped_column(Boolean, nullable=False)

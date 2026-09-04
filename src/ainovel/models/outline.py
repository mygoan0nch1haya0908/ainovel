from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class OutlineVersion(TimestampMixin, Base):
    __tablename__ = "outline_versions"
    __table_args__ = (UniqueConstraint("project_id", "version_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    base_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)


class OutlineNode(TimestampMixin, Base):
    __tablename__ = "outline_nodes"
    __table_args__ = (UniqueConstraint("outline_version_id", "stable_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    outline_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("outline_versions.id"), nullable=False
    )
    stable_key: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    author_locked: Mapped[bool] = mapped_column(Boolean, nullable=False)

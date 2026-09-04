from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class ContextSource(TimestampMixin, Base):
    __tablename__ = "context_sources"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            "source_version",
            "state_scope",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("novel_projects.id"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_version: Mapped[int] = mapped_column(Integer, nullable=False)
    state_scope: Mapped[str] = mapped_column(String(64), nullable=False)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class ContextPacket(TimestampMixin, Base):
    __tablename__ = "context_packets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_workflows.id"), nullable=False
    )
    step_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workflow_steps.id"), nullable=False
    )
    max_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    used_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    fixed_overhead_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class ContextPacketItem(TimestampMixin, Base):
    __tablename__ = "context_packet_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    packet_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("context_packets.id"), nullable=False
    )
    source_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("context_sources.id"), nullable=True
    )
    stable_source_key: Mapped[str] = mapped_column(String(255), nullable=False)
    layer: Mapped[int] = mapped_column(Integer, nullable=False)
    text_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    selected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    relevance: Mapped[int] = mapped_column(Integer, nullable=False)
    temporal_distance: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    excerpt_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excerpt_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trim_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

"""Durable local extraction state; credentials never belong in these rows."""
from datetime import datetime
from sqlalchemy import CheckConstraint, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from ainovel.models.base import Base, TimestampMixin
from ainovel.models.workflow import UTCDateTime


class MemoryExtractionJob(TimestampMixin, Base):
    __tablename__ = 'memory_extraction_jobs'
    __table_args__ = (CheckConstraint("status IN ('DRAFT','READY','RUNNING','PAUSED','NEEDS_REVIEW','STALE','MERGED','CANCELLED')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey('novel_projects.id'))
    model_profile_version_id: Mapped[str] = mapped_column(ForeignKey('model_profile_versions.id'))
    base_card_id: Mapped[str | None] = mapped_column(ForeignKey('memory_card_versions.id'))
    merged_card_id: Mapped[str | None] = mapped_column(ForeignKey('memory_card_versions.id'))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), default='DRAFT')
    snapshot: Mapped[dict] = mapped_column(JSON)
    source_fingerprint: Mapped[str] = mapped_column(String(64))
    rule_version: Mapped[str] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))


class MemoryExtractionChunk(TimestampMixin, Base):
    __tablename__ = 'memory_extraction_chunks'
    __table_args__ = (UniqueConstraint('job_id', 'ordinal'), CheckConstraint("status IN ('PENDING','RUNNING','SUCCEEDED','FAILED','UNKNOWN','REUSED','CANCELLED')"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('memory_extraction_jobs.id'))
    ordinal: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default='PENDING')
    snapshot: Mapped[dict] = mapped_column(JSON)
    cache_key: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict | None] = mapped_column(JSON)


class MemoryExtractionAuthorization(TimestampMixin, Base):
    __tablename__ = 'memory_extraction_authorizations'
    __table_args__ = (CheckConstraint('calls_used >= 0 AND calls_used <= 8'),
                      CheckConstraint('reserved_output_tokens >= 0 AND reserved_output_tokens <= 512000'))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('memory_extraction_jobs.id'))
    job_revision: Mapped[int] = mapped_column(Integer)
    calls_used: Mapped[int] = mapped_column(Integer, default=0)
    reserved_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default='ACTIVE')


class MemoryExtractionAttempt(TimestampMixin, Base):
    __tablename__ = 'memory_extraction_attempts'
    __table_args__ = (UniqueConstraint('chunk_id', 'authorization_id'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey('memory_extraction_chunks.id'))
    authorization_id: Mapped[str] = mapped_column(ForeignKey('memory_extraction_authorizations.id'))
    status: Mapped[str] = mapped_column(String(24))
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    diagnostic: Mapped[dict | None] = mapped_column(JSON)


class ProjectLLMClaim(TimestampMixin, Base):
    __tablename__ = 'project_llm_claims'
    project_id: Mapped[str] = mapped_column(ForeignKey('novel_projects.id'), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), unique=True)

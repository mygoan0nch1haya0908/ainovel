from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin
from ainovel.models.workflow import UTCDateTime


class MemoryCardVersion(TimestampMixin, Base):
    __tablename__ = 'memory_card_versions'
    __table_args__ = (UniqueConstraint('project_id', 'version_number'),
                      CheckConstraint("status IN ('DRAFT','APPROVED')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey('novel_projects.id'))
    version_number: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey('memory_card_versions.id'))
    status: Mapped[str] = mapped_column(String(16), default='DRAFT', server_default='DRAFT')
    entries: Mapped[list] = mapped_column(JSON)
    source_fingerprint: Mapped[str] = mapped_column(String(64))
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class StoryMemoryEntry(TimestampMixin, Base):
    __tablename__ = 'story_memory_entries'
    __table_args__ = (UniqueConstraint('project_id', 'source_key', 'state_scope'),
                      CheckConstraint('effective_from >= 1 AND (effective_until IS NULL OR effective_until >= effective_from)'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey('novel_projects.id'))
    source_key: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    source_refs: Mapped[list] = mapped_column(JSON)
    entity_ids: Mapped[list] = mapped_column(JSON, default=list)
    point_ids: Mapped[list] = mapped_column(JSON, default=list)
    effective_from: Mapped[int] = mapped_column(Integer, default=1)
    effective_until: Mapped[int | None] = mapped_column(Integer)
    reveal_from: Mapped[int | None] = mapped_column(Integer)
    audience: Mapped[str] = mapped_column(String(32), default='author_only')
    state_scope: Mapped[str] = mapped_column(String(64))
    supersedes_id: Mapped[str | None] = mapped_column(ForeignKey('story_memory_entries.id'))


class WorkflowContextPolicy(TimestampMixin, Base):
    __tablename__ = 'workflow_context_policies'
    __table_args__ = (UniqueConstraint('workflow_id', 'version_number'),
                      CheckConstraint("strategy IN ('legacy','scoped_story_v1')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(ForeignKey('generation_workflows.id'))
    version_number: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(32))
    card_id: Mapped[str | None] = mapped_column(ForeignKey('memory_card_versions.id'))
    source_versions: Mapped[dict] = mapped_column(JSON)
    previous_policy_id: Mapped[str | None] = mapped_column(ForeignKey('workflow_context_policies.id'))
    preview_fingerprint: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text('1'))


@event.listens_for(MemoryCardVersion, 'before_update')
def _immutable_card(mapper, connection, target):
    state = inspect(target)
    if any(state.attrs[name].history.has_changes() for name in
           ('project_id', 'version_number', 'parent_id', 'entries', 'source_fingerprint')):
        raise ValueError('memory card version is immutable; create a new version')
    history = state.attrs.status.history
    if history.has_changes() and (list(history.deleted) != ['DRAFT'] or list(history.added) != ['APPROVED']):
        raise ValueError('memory approval is immutable')


@event.listens_for(WorkflowContextPolicy, 'before_update')
def _immutable_policy(mapper, connection, target):
    state = inspect(target)
    if any(state.attrs[name].history.has_changes() for name in
           ('id', 'workflow_id', 'version_number', 'strategy', 'card_id', 'source_versions',
            'previous_policy_id', 'preview_fingerprint')):
        raise ValueError('context policy version is immutable; create a new version')

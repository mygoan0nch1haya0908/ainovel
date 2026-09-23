"""Non-secret profile metadata; authorization flags are the only mutable version fields."""

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String, UniqueConstraint, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from ainovel.models.base import Base, TimestampMixin


class ModelProfile(TimestampMixin, Base):
    __tablename__ = "model_profiles"
    __table_args__ = (CheckConstraint("revision >= 1"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelProfileVersion(TimestampMixin, Base):
    __tablename__ = "model_profile_versions"
    __table_args__ = (
        UniqueConstraint("profile_id", "version_number"),
        CheckConstraint("version_number >= 1"),
        CheckConstraint("context_limit >= 1 AND context_limit <= 32000"),
        CheckConstraint("output_limit >= 1 AND output_limit <= 12000 AND output_limit <= context_limit"),
        CheckConstraint("connection_kind IN ('remote', 'loopback')"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("model_profiles.id"))
    version_number: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(120))
    base_url: Mapped[str] = mapped_column(String(2048))
    connection_kind: Mapped[str] = mapped_column(String(16))
    protocol: Mapped[str] = mapped_column(String(40), default="chat_completions_json_object")
    model_name: Mapped[str] = mapped_column(String(255))
    context_limit: Mapped[int] = mapped_column(Integer)
    output_limit: Mapped[int] = mapped_column(Integer)
    credential_ref: Mapped[str | None] = mapped_column(String(36), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


@event.listens_for(ModelProfileVersion, "before_update")
def _prevent_metadata_mutation(_mapper, _connection, version):
    state = inspect(version)
    immutable = ("id", "profile_id", "version_number", "name", "base_url", "connection_kind",
                 "protocol", "model_name", "context_limit", "output_limit", "credential_ref")
    if any(state.attrs[name].history.has_changes() for name in immutable):
        raise ValueError("model profile versions are immutable")

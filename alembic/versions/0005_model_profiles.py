"""Store immutable non-secret model profiles and mutable authorization flags."""

from alembic import op
import sqlalchemy as sa

revision = "0005_model_profiles"
down_revision = "0004_stage_roadmaps"
branch_labels = None
depends_on = None


def timestamps():
    return [sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    op.create_table("model_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        *timestamps(), sa.CheckConstraint("revision >= 1"))
    op.create_table("model_profile_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), sa.ForeignKey("model_profiles.id"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("base_url", sa.String(2048), nullable=False),
        sa.Column("connection_kind", sa.String(16), nullable=False),
        sa.Column("protocol", sa.String(40), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("context_limit", sa.Integer(), nullable=False),
        sa.Column("output_limit", sa.Integer(), nullable=False),
        sa.Column("credential_ref", sa.String(36)),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        *timestamps(), sa.UniqueConstraint("profile_id", "version_number"),
        sa.CheckConstraint("version_number >= 1"),
        sa.CheckConstraint("context_limit >= 1 AND context_limit <= 32000"),
        sa.CheckConstraint("output_limit >= 1 AND output_limit <= 12000 AND output_limit <= context_limit"),
        sa.CheckConstraint("connection_kind IN ('remote', 'loopback')"))


def downgrade():
    op.drop_table("model_profile_versions")
    op.drop_table("model_profiles")

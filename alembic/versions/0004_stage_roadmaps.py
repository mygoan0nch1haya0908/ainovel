"""Persist versioned stage roadmaps, bounded generation and chapter mappings."""

import sqlalchemy as sa
from alembic import op

revision = "0004_stage_roadmaps"
down_revision = "0003_draft_repair"
branch_labels = None
depends_on = None


def timestamps():
    return [sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    op.create_table(
        "story_stages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), sa.ForeignKey("novel_projects.id"), nullable=False),
        sa.Column("base_outline_version_id", sa.String(36), sa.ForeignKey("outline_versions.id"), nullable=False),
        sa.Column("architecture", sa.Text(), nullable=False),
        sa.Column("approved_roadmap_id", sa.String(36)),
        sa.Column("confirmed_chapters", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        *timestamps(), sa.CheckConstraint("confirmed_chapters >= 0"),
    )
    op.create_table(
        "stage_roadmap_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("stage_id", sa.String(36), sa.ForeignKey("story_stages.id"), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.Integer(), nullable=False),
        sa.Column("constitution_version_id", sa.String(36), sa.ForeignKey("constitution_versions.id"), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("provider_name", sa.String(40), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("architecture", sa.Text(), nullable=False),
        sa.Column("prompt_snapshot", sa.JSON(), nullable=False),
        sa.Column("input_snapshot", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("attempts_used", sa.Integer(), nullable=False),
        sa.Column("attempt_limit", sa.Integer(), nullable=False),
        sa.Column("input_token_limit", sa.Integer(), nullable=False),
        sa.Column("output_token_limit", sa.Integer(), nullable=False),
        sa.Column("total_input_token_limit", sa.Integer(), nullable=False),
        sa.Column("total_output_token_limit", sa.Integer(), nullable=False),
        sa.Column("actual_input_tokens", sa.Integer()),
        sa.Column("actual_output_tokens", sa.Integer()),
        sa.Column("approved_by", sa.String(255)),
        *timestamps(), sa.UniqueConstraint("stage_id", "version_number"),
        sa.CheckConstraint("attempts_used >= 0 AND attempts_used <= attempt_limit"),
    )
    op.create_table(
        "stage_model_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("roadmap_id", sa.String(36), sa.ForeignKey("stage_roadmap_versions.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("error_code", sa.String(80)),
        *timestamps(), sa.UniqueConstraint("roadmap_id", "number"),
    )
    op.create_table(
        "stage_workflows",
        sa.Column("workflow_id", sa.String(36), sa.ForeignKey("generation_workflows.id"), primary_key=True),
        sa.Column("stage_id", sa.String(36), sa.ForeignKey("story_stages.id"), nullable=False),
        sa.Column("roadmap_id", sa.String(36), sa.ForeignKey("stage_roadmap_versions.id"), nullable=False),
        sa.Column("confirmed_start", sa.Integer(), nullable=False),
        sa.Column("committed_batch_id", sa.String(36), sa.ForeignKey("writing_batches.id"), unique=True),
        *timestamps(),
    )
    op.create_table(
        "stage_workflow_nodes",
        sa.Column("workflow_id", sa.String(36), sa.ForeignKey("stage_workflows.workflow_id"), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
        sa.Column("node_id", sa.String(64), nullable=False),
        sa.Column("stage_ordinal", sa.Integer(), nullable=False),
        sa.Column("book_ordinal", sa.Integer(), nullable=False),
        sa.UniqueConstraint("workflow_id", "node_id"),
        sa.UniqueConstraint("workflow_id", "stage_ordinal"),
        sa.CheckConstraint("ordinal >= 1 AND ordinal <= 5 AND stage_ordinal >= 1 AND book_ordinal >= 1"),
    )


def downgrade():
    for name in ("stage_workflow_nodes", "stage_workflows", "stage_model_attempts", "stage_roadmap_versions", "story_stages"):
        op.drop_table(name)

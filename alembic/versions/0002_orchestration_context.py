"""Add orchestration and context persistence.

Revision ID: 0002_orchestration_context
Revises: 0001_foundation
Create Date: 2026-09-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0002_orchestration_context"
down_revision: str | None = "0001_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def upgrade() -> None:
    op.add_column(
        "novel_projects",
        sa.Column("active_workflow_id", sa.String(length=36), nullable=True),
    )
    with op.batch_alter_table("writing_batches") as batch_op:
        batch_op.add_column(
            sa.Column("source_workflow_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_unique_constraint(
            "uq_writing_batches_source_workflow_id", ["source_workflow_id"]
        )

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "active", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("source", sa.String(length=64), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("role", "version_number"),
    )
    op.create_index(
        "uq_prompt_versions_one_active_role",
        "prompt_versions",
        ["role"],
        unique=True,
        sqlite_where=sa.text("active = 1"),
    )

    op.create_table(
        "generation_workflows",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("base_outline_version_id", sa.String(length=36), nullable=False),
        sa.Column("provider_name", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("requested_chapters", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "current_position",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("candidate_batch_id", sa.String(length=36), nullable=True),
        sa.Column("planner_input_tokens", sa.Integer(), nullable=False),
        sa.Column("planner_output_tokens", sa.Integer(), nullable=False),
        sa.Column("writer_input_tokens", sa.Integer(), nullable=False),
        sa.Column("writer_output_tokens", sa.Integer(), nullable=False),
        sa.Column("summarizer_input_tokens", sa.Integer(), nullable=False),
        sa.Column("summarizer_output_tokens", sa.Integer(), nullable=False),
        sa.Column("reviewer_input_tokens", sa.Integer(), nullable=False),
        sa.Column("reviewer_output_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "actual_input_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "actual_output_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column("last_error_detail", sa.Text(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["base_outline_version_id"], ["outline_versions.id"]
        ),
        sa.ForeignKeyConstraint(["candidate_batch_id"], ["writing_batches.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["novel_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_generation_workflows_project_status",
        "generation_workflows",
        ["project_id", "status"],
        unique=False,
    )

    op.create_table(
        "workflow_prompt_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("prompt_version_id", sa.String(length=36), nullable=False),
        sa.Column("prompt_body", sa.Text(), nullable=False),
        sa.Column("output_schema", sa.JSON(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["prompt_version_id"], ["prompt_versions.id"]),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workflow_id", "role"),
    )

    op.create_table(
        "workflow_steps",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("active_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("lease_owner", sa.String(length=255), nullable=True),
        sa.Column(
            "lease_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "(lease_owner IS NULL AND lease_expires_at IS NULL) OR "
            "(lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_workflow_steps_lease_pair",
        ),
        sa.UniqueConstraint("workflow_id", "kind", "ordinal"),
        sa.UniqueConstraint("workflow_id", "position"),
    )
    op.create_index(
        "ix_workflow_steps_workflow_status",
        "workflow_steps",
        ["workflow_id", "status"],
        unique=False,
    )
    op.create_index(
        "uq_workflow_steps_one_null_ordinal_kind",
        "workflow_steps",
        ["workflow_id", "kind"],
        unique=True,
        sqlite_where=sa.text("ordinal IS NULL"),
    )

    op.create_table(
        "model_attempts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_digest", sa.String(length=64), nullable=False),
        sa.Column("provider_response_id", sa.String(length=255), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["step_id"], ["workflow_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("step_id", "attempt_number"),
    )

    op.create_table(
        "workflow_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=True),
        sa.Column("text_content", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("visible_char_count", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["step_id"], ["workflow_steps.id"]),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "plan_decisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "context_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("source_version", sa.Integer(), nullable=False),
        sa.Column("state_scope", sa.String(length=64), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["novel_projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            "source_version",
            "state_scope",
        ),
    )

    op.create_table(
        "context_packets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("step_id", sa.String(length=36), nullable=False),
        sa.Column("max_input_tokens", sa.Integer(), nullable=False),
        sa.Column("used_input_tokens", sa.Integer(), nullable=False),
        sa.Column("fixed_overhead_tokens", sa.Integer(), nullable=False),
        sa.Column("reserved_output_tokens", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["step_id"], ["workflow_steps.id"]),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "context_packet_items",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("packet_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=True),
        sa.Column("stable_source_key", sa.String(length=255), nullable=False),
        sa.Column("layer", sa.Integer(), nullable=False),
        sa.Column("text_snapshot", sa.Text(), nullable=False),
        sa.Column(
            "selected", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "required", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("relevance", sa.Integer(), nullable=False),
        sa.Column("temporal_distance", sa.Integer(), nullable=False),
        sa.Column("estimated_tokens", sa.Integer(), nullable=False),
        sa.Column("excerpt_start", sa.Integer(), nullable=True),
        sa.Column("excerpt_end", sa.Integer(), nullable=True),
        sa.Column("trim_reason", sa.String(length=64), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["packet_id"], ["context_packets.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["context_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.execute(
        "CREATE VIRTUAL TABLE context_source_fts "
        "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE context_source_fts")
    op.drop_table("context_packet_items")
    op.drop_table("context_packets")
    op.drop_table("context_sources")
    op.drop_table("plan_decisions")
    op.drop_table("workflow_artifacts")
    op.drop_table("model_attempts")
    op.drop_index(
        "uq_workflow_steps_one_null_ordinal_kind", table_name="workflow_steps"
    )
    op.drop_index(
        "ix_workflow_steps_workflow_status", table_name="workflow_steps"
    )
    op.drop_table("workflow_steps")
    op.drop_table("workflow_prompt_snapshots")
    op.drop_index(
        "ix_generation_workflows_project_status",
        table_name="generation_workflows",
    )
    op.drop_table("generation_workflows")
    op.drop_index(
        "uq_prompt_versions_one_active_role", table_name="prompt_versions"
    )
    op.drop_table("prompt_versions")

    with op.batch_alter_table("writing_batches") as batch_op:
        batch_op.drop_constraint(
            "uq_writing_batches_source_workflow_id", type_="unique"
        )
        batch_op.drop_column("source_workflow_id")
    op.drop_column("novel_projects", "active_workflow_id")

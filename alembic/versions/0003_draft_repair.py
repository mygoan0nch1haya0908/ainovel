"""Add versioned durable chapter draft repair.

Revision ID: 0003_draft_repair
Revises: 0002_orchestration_context
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0003_draft_repair"
down_revision: str | None = "0002_orchestration_context"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("generation_workflows") as batch_op:
        batch_op.add_column(
            sa.Column(
                "generation_version",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("1"),
            )
        )
        batch_op.add_column(sa.Column("model_call_limit", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("total_input_token_limit", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("total_output_token_limit", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "model_calls_used",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
    with op.batch_alter_table("workflow_steps") as batch_op:
        batch_op.add_column(
            sa.Column(
                "protocol_failure_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )

    op.create_table(
        "chapter_draft_repairs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=36), nullable=False),
        sa.Column("writing_step_id", sa.String(length=36), nullable=False),
        sa.Column("latest_attempt_id", sa.String(length=36), nullable=False),
        sa.Column("latest_payload", sa.JSON(), nullable=False),
        sa.Column("visible_count", sa.Integer(), nullable=False),
        sa.Column(
            "repair_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "repair_pending", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "draft_revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["latest_attempt_id"], ["model_attempts.id"]),
        sa.ForeignKeyConstraint(["workflow_id"], ["generation_workflows.id"]),
        sa.ForeignKeyConstraint(["writing_step_id"], ["workflow_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("writing_step_id"),
    )
    op.create_index(
        "ix_chapter_draft_repairs_workflow",
        "chapter_draft_repairs",
        ["workflow_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chapter_draft_repairs_workflow", table_name="chapter_draft_repairs"
    )
    op.drop_table("chapter_draft_repairs")
    with op.batch_alter_table("workflow_steps") as batch_op:
        batch_op.drop_column("protocol_failure_count")
    with op.batch_alter_table("generation_workflows") as batch_op:
        batch_op.drop_column("model_calls_used")
        batch_op.drop_column("total_output_token_limit")
        batch_op.drop_column("total_input_token_limit")
        batch_op.drop_column("model_call_limit")
        batch_op.drop_column("generation_version")

"""Versioned, author-approved local story memory. No legacy data is converted."""
from alembic import op
import sqlalchemy as sa

revision = '0009_scoped_story_memory'
down_revision = '0008_model_output_capacity'
branch_labels = None
depends_on = None


def timestamps():
    return [sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    op.create_table('memory_card_versions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('novel_projects.id'), nullable=False),
        sa.Column('version_number', sa.Integer, nullable=False),
        sa.Column('parent_id', sa.String(36), sa.ForeignKey('memory_card_versions.id')),
        sa.Column('status', sa.String(16), nullable=False, server_default='DRAFT'),
        sa.Column('entries', sa.JSON, nullable=False),
        sa.Column('source_fingerprint', sa.String(64), nullable=False),
        sa.Column('approved_by', sa.String(255)), sa.Column('approved_at', sa.DateTime(timezone=True)),
        *timestamps(), sa.UniqueConstraint('project_id', 'version_number'),
        sa.CheckConstraint("status IN ('DRAFT','APPROVED')"))
    op.create_table('story_memory_entries',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('novel_projects.id'), nullable=False),
        sa.Column('source_key', sa.String(255), nullable=False),
        sa.Column('kind', sa.String(32), nullable=False), sa.Column('text', sa.Text, nullable=False),
        sa.Column('source_refs', sa.JSON, nullable=False), sa.Column('entity_ids', sa.JSON, nullable=False),
        sa.Column('point_ids', sa.JSON, nullable=False), sa.Column('effective_from', sa.Integer, nullable=False),
        sa.Column('effective_until', sa.Integer), sa.Column('reveal_from', sa.Integer),
        sa.Column('audience', sa.String(32), nullable=False), sa.Column('state_scope', sa.String(64), nullable=False),
        sa.Column('supersedes_id', sa.String(36), sa.ForeignKey('story_memory_entries.id')), *timestamps(),
        sa.UniqueConstraint('project_id', 'source_key', 'state_scope'),
        sa.CheckConstraint('effective_from >= 1 AND (effective_until IS NULL OR effective_until >= effective_from)'))
    op.create_table('workflow_context_policies',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('workflow_id', sa.String(36), sa.ForeignKey('generation_workflows.id'), nullable=False),
        sa.Column('version_number', sa.Integer, nullable=False), sa.Column('strategy', sa.String(32), nullable=False),
        sa.Column('card_id', sa.String(36), sa.ForeignKey('memory_card_versions.id')),
        sa.Column('source_versions', sa.JSON, nullable=False),
        sa.Column('previous_policy_id', sa.String(36), sa.ForeignKey('workflow_context_policies.id')),
        sa.Column('preview_fingerprint', sa.String(64), nullable=False),
        sa.Column('active', sa.Boolean, nullable=False, server_default=sa.text('1')), *timestamps(),
        sa.UniqueConstraint('workflow_id', 'version_number'),
        sa.CheckConstraint("strategy IN ('legacy','scoped_story_v1')"))


def downgrade():
    tables = ('workflow_context_policies', 'story_memory_entries', 'memory_card_versions')
    if any(op.get_bind().execute(sa.text(f'SELECT COUNT(*) FROM {table}')).scalar() for table in tables):
        raise ValueError('Cannot discard story memory data; restore a pre-migration backup instead.')
    for table in tables:
        op.drop_table(table)

"""Durable AI extraction jobs, authorizations and project dispatch claims."""
from alembic import op
import sqlalchemy as sa

revision = '0010_memory_extraction'
down_revision = '0009_scoped_story_memory'
branch_labels = None
depends_on = None


def stamps():
    return [sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    _output_limit(32000, 64000)
    op.create_table('memory_extraction_jobs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('project_id', sa.String(36), sa.ForeignKey('novel_projects.id'), nullable=False),
        sa.Column('model_profile_version_id', sa.String(36), sa.ForeignKey('model_profile_versions.id'), nullable=False),
        sa.Column('base_card_id', sa.String(36), sa.ForeignKey('memory_card_versions.id')),
        sa.Column('merged_card_id', sa.String(36), sa.ForeignKey('memory_card_versions.id')),
        sa.Column('revision', sa.Integer, nullable=False), sa.Column('status', sa.String(24), nullable=False),
        sa.Column('snapshot', sa.JSON, nullable=False), sa.Column('source_fingerprint', sa.String(64), nullable=False),
        sa.Column('rule_version', sa.String(64), nullable=False), sa.Column('error_code', sa.String(64)), *stamps(),
        sa.CheckConstraint("status IN ('DRAFT','READY','RUNNING','PAUSED','NEEDS_REVIEW','STALE','MERGED','CANCELLED')"))
    op.create_table('memory_extraction_chunks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('memory_extraction_jobs.id'), nullable=False),
        sa.Column('ordinal', sa.Integer, nullable=False), sa.Column('status', sa.String(24), nullable=False),
        sa.Column('snapshot', sa.JSON, nullable=False), sa.Column('cache_key', sa.String(64), nullable=False),
        sa.Column('result', sa.JSON), *stamps(), sa.UniqueConstraint('job_id', 'ordinal'),
        sa.CheckConstraint("status IN ('PENDING','RUNNING','SUCCEEDED','FAILED','UNKNOWN','REUSED','CANCELLED')"))
    op.create_table('memory_extraction_authorizations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('job_id', sa.String(36), sa.ForeignKey('memory_extraction_jobs.id'), nullable=False),
        sa.Column('job_revision', sa.Integer, nullable=False), sa.Column('calls_used', sa.Integer, nullable=False),
        sa.Column('reserved_output_tokens', sa.Integer, nullable=False), sa.Column('status', sa.String(24), nullable=False),
        *stamps(), sa.CheckConstraint('calls_used >= 0 AND calls_used <= 8'),
        sa.CheckConstraint('reserved_output_tokens >= 0 AND reserved_output_tokens <= 512000'))
    op.create_table('memory_extraction_attempts',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('chunk_id', sa.String(36), sa.ForeignKey('memory_extraction_chunks.id'), nullable=False),
        sa.Column('authorization_id', sa.String(36), sa.ForeignKey('memory_extraction_authorizations.id'), nullable=False),
        sa.Column('status', sa.String(24), nullable=False), sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('ended_at', sa.DateTime(timezone=True)), sa.Column('input_tokens', sa.Integer),
        sa.Column('output_tokens', sa.Integer), sa.Column('error_code', sa.String(64)), sa.Column('diagnostic', sa.JSON),
        *stamps(), sa.UniqueConstraint('chunk_id', 'authorization_id'))
    op.create_table('project_llm_claims',
        sa.Column('project_id', sa.String(36), sa.ForeignKey('novel_projects.id'), primary_key=True),
        sa.Column('owner_id', sa.String(128), nullable=False, unique=True), *stamps())


def downgrade():
    tables = ('project_llm_claims', 'memory_extraction_attempts', 'memory_extraction_authorizations',
              'memory_extraction_chunks', 'memory_extraction_jobs')
    if any(op.get_bind().execute(sa.text(f'SELECT COUNT(*) FROM {table}')).scalar() for table in tables):
        raise ValueError('Cannot discard memory extraction history; restore a backup.')
    if op.get_bind().execute(sa.text('SELECT COUNT(*) FROM model_profile_versions WHERE output_limit > 32000')).scalar():
        raise ValueError('Cannot downgrade profiles above 32000; restore a backup.')
    for table in tables:
        op.drop_table(table)
    _output_limit(64000, 32000)


def _output_limit(old, new):
    table = sa.Table('model_profile_versions', sa.MetaData(), autoload_with=op.get_bind())
    for constraint in list(table.constraints):
        if isinstance(constraint, sa.CheckConstraint) and str(constraint.sqltext) == f'output_limit >= 1 AND output_limit <= {old} AND output_limit <= context_limit':
            table.constraints.remove(constraint)
    table.append_constraint(sa.CheckConstraint(f'output_limit >= 1 AND output_limit <= {new} AND output_limit <= context_limit'))
    with op.batch_alter_table(table.name, copy_from=table, recreate='always'):
        pass

"""Allow 32K output budgets without rewriting immutable profile versions."""
from alembic import op
import sqlalchemy as sa

revision = "0008_model_output_capacity"
down_revision = "0007_model_context_capacity"
branch_labels = None
depends_on = None


def _replace_output(old, new):
    table = sa.Table('model_profile_versions', sa.MetaData(), autoload_with=op.get_bind())
    old_check = f'output_limit >= 1 AND output_limit <= {old} AND output_limit <= context_limit'
    for constraint in list(table.constraints):
        if isinstance(constraint, sa.CheckConstraint) and str(constraint.sqltext) == old_check:
            table.constraints.remove(constraint)
    table.append_constraint(sa.CheckConstraint(f'output_limit >= 1 AND output_limit <= {new} AND output_limit <= context_limit'))
    with op.batch_alter_table(table.name, copy_from=table, recreate='always'):
        pass


def upgrade():
    _replace_output(12000, 32000)


def downgrade():
    if op.get_bind().execute(sa.text('SELECT COUNT(*) FROM model_profile_versions WHERE output_limit > 12000')).scalar():
        raise ValueError('Cannot downgrade while profiles require output above 12000; restore a pre-upgrade backup instead.')
    _replace_output(32000, 12000)

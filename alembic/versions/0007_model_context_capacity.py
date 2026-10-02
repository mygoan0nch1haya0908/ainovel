"""Allow author-configured model context capacity without the old 32K cap."""
from alembic import op
import sqlalchemy as sa

revision = "0007_model_context_capacity"
down_revision = "0006_model_profile_bindings"
branch_labels = None
depends_on = None


def _replace_capacity(old, new):
    # Explicit reflection preserves all unnamed CHECKs during SQLite recreation.
    table = sa.Table("model_profile_versions", sa.MetaData(), autoload_with=op.get_bind())
    for constraint in list(table.constraints):
        if isinstance(constraint, sa.CheckConstraint) and str(constraint.sqltext) == f"context_limit >= 1 AND context_limit <= {old}":
            table.constraints.remove(constraint)
    table.append_constraint(sa.CheckConstraint(f"context_limit >= 1 AND context_limit <= {new}"))
    with op.batch_alter_table(table.name, copy_from=table, recreate="always"):
        pass


def upgrade():
    _replace_capacity(32000, 2147483647)


def downgrade():
    if op.get_bind().execute(sa.text("SELECT COUNT(*) FROM model_profile_versions WHERE context_limit > 32000")).scalar():
        raise ValueError("Cannot downgrade while model profiles require contexts above 32000; restore a pre-upgrade backup instead.")
    _replace_capacity(2147483647, 32000)

"""Pin workflow and stage generation to immutable model profile versions."""

from alembic import op
import sqlalchemy as sa

revision = "0006_model_profile_bindings"
down_revision = "0005_model_profiles"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("generation_workflows") as batch:
        batch.add_column(sa.Column("model_profile_version_id", sa.String(36), nullable=True))
        batch.create_foreign_key("fk_generation_workflows_profile_version",
                                 "model_profile_versions", ["model_profile_version_id"], ["id"])
    with op.batch_alter_table("stage_roadmap_versions") as batch:
        batch.add_column(sa.Column("model_profile_version_id", sa.String(36), nullable=True))
        batch.create_foreign_key("fk_stage_roadmap_versions_profile_version",
                                 "model_profile_versions", ["model_profile_version_id"], ["id"])


def downgrade():
    with op.batch_alter_table("stage_roadmap_versions") as batch:
        batch.drop_constraint("fk_stage_roadmap_versions_profile_version", type_="foreignkey")
        batch.drop_column("model_profile_version_id")
    with op.batch_alter_table("generation_workflows") as batch:
        batch.drop_constraint("fk_generation_workflows_profile_version", type_="foreignkey")
        batch.drop_column("model_profile_version_id")

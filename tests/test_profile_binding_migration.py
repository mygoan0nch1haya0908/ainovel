from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_binding_columns_upgrade_and_downgrade(tmp_path: Path):
    database = tmp_path / "binding.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    command.upgrade(config, "0005_model_profiles")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    try:
        assert "model_profile_version_id" in {c["name"] for c in inspect(engine).get_columns("generation_workflows")}
        assert "model_profile_version_id" in {c["name"] for c in inspect(engine).get_columns("stage_roadmap_versions")}
        command.downgrade(config, "0005_model_profiles")
        assert "model_profile_version_id" not in {c["name"] for c in inspect(engine).get_columns("generation_workflows")}
    finally:
        engine.dispose()

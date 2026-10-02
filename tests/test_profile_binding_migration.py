from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


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


def test_stage_attempt_check_and_existing_data_survive_upgrade_and_downgrade(tmp_path: Path):
    database = tmp_path / "attempts.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database.as_posix()}")
    command.upgrade(config, "0005_model_profiles")
    engine = create_engine(f"sqlite:///{database.as_posix()}")
    try:
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO stage_roadmap_versions
                (id, stage_id, version_number, input_revision, constitution_version_id,
                 status, provider_name, model_name, architecture, prompt_snapshot,
                 input_snapshot, attempts_used, attempt_limit, input_token_limit,
                 output_token_limit, total_input_token_limit, total_output_token_limit,
                 created_at, updated_at)
                VALUES ('roadmap-existing', 'stage-existing', 1, 1, 'constitution-existing',
                        'PENDING', 'fake', 'demo', 'architecture', '{}', '{}',
                        1, 2, 100, 100, 200, 200, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """))
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT attempts_used, attempt_limit FROM stage_roadmap_versions WHERE id='roadmap-existing'"
            )).one() == (1, 2)
            with pytest.raises(IntegrityError):
                connection.execute(text(
                    "UPDATE stage_roadmap_versions SET attempts_used=3 WHERE id='roadmap-existing'"
                ))
        command.downgrade(config, "0005_model_profiles")
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT attempts_used, attempt_limit FROM stage_roadmap_versions WHERE id='roadmap-existing'"
            )).one() == (1, 2)
            with pytest.raises(IntegrityError):
                connection.execute(text(
                    "UPDATE stage_roadmap_versions SET attempts_used=3 WHERE id='roadmap-existing'"
                ))
    finally:
        engine.dispose()

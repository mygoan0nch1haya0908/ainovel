from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from ainovel.db import create_engine_for_url, database_readiness


def test_profile_migration_preserves_0004_data_and_downgrades(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'profile-migration.db'}"
    monkeypatch.setenv("AINOVEL_DATABASE_URL", url)
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "0004_stage_roadmaps")
    engine = create_engine_for_url(url)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO novel_projects (id,title,target_chars_min,target_chars_max,next_batch_sequence,next_official_chapter_number,created_at,updated_at) VALUES ('legacy','preserved',2000000,5000000,1,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
    command.upgrade(config, "head")
    assert {"model_profiles", "model_profile_versions"} <= set(inspect(engine).get_table_names())
    assert database_readiness(engine) == (True, None)
    command.check(config)
    command.downgrade(config, "0004_stage_roadmaps")
    assert "model_profiles" not in inspect(engine).get_table_names()
    assert database_readiness(engine)[0] is False
    with engine.connect() as connection:
        assert connection.execute(text("SELECT title FROM novel_projects WHERE id='legacy'")).scalar_one() == "preserved"
    engine.dispose()

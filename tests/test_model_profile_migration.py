from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from ainovel.db import create_engine_for_url, database_readiness


def test_context_migration_preserves_profiles_and_other_checks(tmp_path, monkeypatch):
    import pytest
    from sqlalchemy.exc import IntegrityError
    url = f"sqlite+pysqlite:///{tmp_path / 'context.db'}"
    monkeypatch.setenv("AINOVEL_DATABASE_URL", url)
    config = Config("alembic.ini")
    command.upgrade(config, "0006_model_profile_bindings")
    engine = create_engine_for_url(url)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO model_profiles (id,revision,revoked,created_at,updated_at) VALUES ('p',1,0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        conn.execute(text("INSERT INTO model_profile_versions (id,profile_id,version_number,name,base_url,connection_kind,protocol,model_name,context_limit,output_limit,enabled,revoked,created_at,updated_at) VALUES ('v','p',1,'test','https://example.com','remote','chat_completions_json_object','test',32000,8000,1,0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
    command.upgrade(config, "head")
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_profile_versions SET context_limit=131072 WHERE id='v'"))
        conn.execute(text("UPDATE model_profile_versions SET output_limit=32000 WHERE id='v'"))
        assert conn.execute(text("SELECT profile_id FROM model_profile_versions WHERE id='v'")).scalar_one() == 'p'
        with pytest.raises(IntegrityError):
            conn.execute(text("UPDATE model_profile_versions SET output_limit=64001 WHERE id='v'"))
    with pytest.raises(ValueError):
        command.downgrade(config, "0006_model_profile_bindings")
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_profile_versions SET output_limit=8000 WHERE id='v'"))
        conn.execute(text("UPDATE model_profile_versions SET context_limit=32000 WHERE id='v'"))
    command.downgrade(config, "0006_model_profile_bindings")
    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(text("UPDATE model_profile_versions SET context_limit=131072 WHERE id='v'"))
    engine.dispose()


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

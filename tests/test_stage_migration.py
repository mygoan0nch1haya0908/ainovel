from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ainovel.db import create_engine_for_url, database_readiness
from ainovel.models import StoryStage


def test_stage_migration_preserves_legacy_rows_and_downgrades_cleanly(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'stage-upgrade.db'}"
    monkeypatch.setenv("AINOVEL_DATABASE_URL", url)
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    command.upgrade(config, "0003_draft_repair")
    engine = create_engine_for_url(url)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO novel_projects (id,title,target_chars_min,target_chars_max,next_batch_sequence,next_official_chapter_number,created_at,updated_at) VALUES ('old-project','legacy',2000000,5000000,1,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        connection.execute(text("INSERT INTO outline_versions (id,project_id,version_number,status,reason,created_at,updated_at) VALUES ('old-outline','old-project',1,'official','legacy',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        connection.execute(text("UPDATE novel_projects SET official_outline_version_id='old-outline' WHERE id='old-project'"))
    command.upgrade(config, "head")
    assert "stage_roadmap_versions" in inspect(engine).get_table_names()
    assert database_readiness(engine) == (True, None)
    command.check(config)
    with Session(engine) as session:
        session.add(StoryStage(id="new-stage", project_id="old-project", base_outline_version_id="old-outline", architecture="new architecture"))
        session.commit()
        assert session.get(StoryStage, "new-stage").confirmed_chapters == 0
    command.downgrade(config, "0003_draft_repair")
    assert "story_stages" not in inspect(engine).get_table_names()
    with engine.connect() as connection:
        assert connection.execute(text("SELECT title FROM novel_projects WHERE id='old-project'")).scalar_one() == "legacy"
        assert connection.execute(text("SELECT reason FROM outline_versions WHERE id='old-outline'")).scalar_one() == "legacy"
    engine.dispose()

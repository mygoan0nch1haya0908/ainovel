from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from ainovel.services.batches import BatchService
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_TABLES = {
    "alembic_version",
    "audit_events",
    "chapters",
    "constitution_versions",
    "novel_projects",
    "outline_nodes",
    "outline_versions",
    "writing_batches",
}


def _table_names(database_url: str) -> set[str]:
    engine = create_engine(database_url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _revision_number(database_url: str) -> str:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
    finally:
        engine.dispose()


def test_initial_migration_round_trip_creates_foundation_schema(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "migration-round-trip.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("AINOVEL_DATABASE_URL", database_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))

    command.upgrade(config, "head")
    assert _table_names(database_url) == FOUNDATION_TABLES
    assert _revision_number(database_url) == "0001_foundation"

    command.downgrade(config, "base")
    assert _table_names(database_url) == {"alembic_version"}

    command.upgrade(config, "head")
    assert _table_names(database_url) == FOUNDATION_TABLES
    assert _revision_number(database_url) == "0001_foundation"


def test_author_can_approve_one_batch_without_candidate_leakage(session) -> None:
    project = ProjectService(session).create("山海铸天", 2_000_000, 5_000_000)
    outline_service = OutlineService(session)
    outline = outline_service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="author approved initial outline",
    )
    outline_service.approve(outline.id)

    batch_service = BatchService(session)
    batch = batch_service.create(project.id, outline.id, 1)
    candidate = batch_service.save_candidate_chapter(
        batch.id, 1, "山门之外", "山" * 4500, {"timeline_days": 1}
    )

    session.expire_all()
    isolated_candidate = batch_service.list_chapters(batch.id)[0]
    assert candidate.status == "candidate"
    assert isolated_candidate.status == "candidate"
    assert isolated_candidate.official_chapter_number is None
    candidate_statistics = batch_service.official_chapter_statistics(project.id)
    assert candidate_statistics.chapter_count == 0
    assert candidate_statistics.visible_character_count == 0

    batch_service.mark_ready(batch.id)
    batch_service.approve(batch.id, outline.id)

    session.expire_all()
    official = batch_service.list_chapters(batch.id)[0]
    audit_events = batch_service.list_audit_events(project.id)
    assert official.status == "official"
    assert official.official_chapter_number == 1
    official_statistics = batch_service.official_chapter_statistics(project.id)
    assert batch_service.get(batch.id).status == "approved"
    assert ProjectService(session).get(project.id).next_official_chapter_number == 2
    assert official_statistics.chapter_count == 1
    assert official_statistics.visible_character_count == 4500
    assert audit_events[0].action == "batch_approved"
    assert audit_events[0].entity_id == batch.id

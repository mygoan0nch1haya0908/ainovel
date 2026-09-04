from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

from ainovel.models import Base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_TWO_TABLES = {
    "context_packet_items",
    "context_packets",
    "context_sources",
    "generation_workflows",
    "model_attempts",
    "plan_decisions",
    "prompt_versions",
    "workflow_artifacts",
    "workflow_prompt_snapshots",
    "workflow_steps",
}


def _remove_database_artifacts(database_path: Path, temporary_root: Path) -> None:
    for candidate in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    ):
        if candidate.exists():
            resolved_candidate = candidate.resolve(strict=True)
            assert resolved_candidate.parent == temporary_root
            assert resolved_candidate.name in {
                database_path.name,
                f"{database_path.name}-wal",
                f"{database_path.name}-shm",
            }
            resolved_candidate.unlink()


@pytest.fixture
def migrated_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Engine]:
    temporary_root = tmp_path.resolve(strict=True)
    assert temporary_root.is_relative_to(PROJECT_ROOT.resolve())
    database_path = temporary_root / "orchestration-schema.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.chdir(temporary_root)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()
        _remove_database_artifacts(database_path, temporary_root)


def _column_map(engine: Engine, table_name: str) -> dict[str, dict[str, object]]:
    return {column["name"]: column for column in inspect(engine).get_columns(table_name)}


def _unique_column_sets(engine: Engine, table_name: str) -> set[tuple[str, ...]]:
    inspector = inspect(engine)
    constraints = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints(table_name)
    }
    indexes = {
        tuple(index["column_names"])
        for index in inspector.get_indexes(table_name)
        if index["unique"]
    }
    return constraints | indexes


def _foreign_keys(engine: Engine, table_name: str) -> set[tuple[str, str, str]]:
    return {
        (
            foreign_key["constrained_columns"][0],
            foreign_key["referred_table"],
            foreign_key["referred_columns"][0],
        )
        for foreign_key in inspect(engine).get_foreign_keys(table_name)
    }


def test_stage_two_tables_and_fts_exist(migrated_engine: Engine) -> None:
    names = set(inspect(migrated_engine).get_table_names())
    assert STAGE_TWO_TABLES <= names
    with migrated_engine.connect() as connection:
        fts = connection.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='context_source_fts'"
            )
        ).scalar_one()
    assert fts == "context_source_fts"


def test_stage_two_columns_nullability_and_server_defaults(
    migrated_engine: Engine,
) -> None:
    projects = _column_map(migrated_engine, "novel_projects")
    batches = _column_map(migrated_engine, "writing_batches")
    assert projects["active_workflow_id"]["nullable"] is True
    assert batches["source_workflow_id"]["nullable"] is True

    workflow = _column_map(migrated_engine, "generation_workflows")
    assert workflow["candidate_batch_id"]["nullable"] is True
    assert workflow["last_error_code"]["nullable"] is True
    assert workflow["last_error_detail"]["nullable"] is True
    assert workflow["current_position"]["default"] == "0"
    assert workflow["actual_input_tokens"]["default"] == "0"
    assert workflow["actual_output_tokens"]["default"] == "0"
    assert workflow["revision"]["default"] == "1"

    prompt = _column_map(migrated_engine, "prompt_versions")
    assert prompt["active"]["default"] == "0"

    step = _column_map(migrated_engine, "workflow_steps")
    assert step["ordinal"]["nullable"] is True
    assert step["active_artifact_id"]["nullable"] is True
    assert step["lease_owner"]["nullable"] is True
    assert step["lease_expires_at"]["nullable"] is True
    assert step["attempt_count"]["default"] == "0"
    assert step["revision"]["default"] == "1"

    item = _column_map(migrated_engine, "context_packet_items")
    assert item["source_id"]["nullable"] is True
    assert item["excerpt_start"]["nullable"] is True
    assert item["excerpt_end"]["nullable"] is True
    assert item["trim_reason"]["nullable"] is True
    assert item["selected"]["default"] == "0"
    assert item["required"]["default"] == "0"
    assert Base.metadata.tables["workflow_steps"].c.lease_expires_at.type.timezone is True


def test_stage_two_uniqueness_and_partial_active_prompt_index(
    migrated_engine: Engine,
) -> None:
    assert ("source_workflow_id",) in _unique_column_sets(
        migrated_engine, "writing_batches"
    )
    assert ("role", "version_number") in _unique_column_sets(
        migrated_engine, "prompt_versions"
    )
    assert ("workflow_id", "role") in _unique_column_sets(
        migrated_engine, "workflow_prompt_snapshots"
    )
    assert ("workflow_id", "position") in _unique_column_sets(
        migrated_engine, "workflow_steps"
    )
    assert ("workflow_id", "kind", "ordinal") in _unique_column_sets(
        migrated_engine, "workflow_steps"
    )
    assert ("step_id", "attempt_number") in _unique_column_sets(
        migrated_engine, "model_attempts"
    )
    assert (
        "project_id",
        "source_type",
        "source_id",
        "source_version",
        "state_scope",
    ) in _unique_column_sets(migrated_engine, "context_sources")

    indexes = {
        index["name"]: index
        for index in inspect(migrated_engine).get_indexes("prompt_versions")
    }
    active_index = indexes["uq_prompt_versions_one_active_role"]
    assert active_index["unique"] == 1
    assert active_index["column_names"] == ["role"]
    assert str(active_index["dialect_options"]["sqlite_where"]) == "active = 1"


def test_stage_two_foreign_keys_and_intentional_non_foreign_key_pointers(
    migrated_engine: Engine,
) -> None:
    assert _foreign_keys(migrated_engine, "generation_workflows") == {
        ("base_outline_version_id", "outline_versions", "id"),
        ("candidate_batch_id", "writing_batches", "id"),
        ("project_id", "novel_projects", "id"),
    }
    assert _foreign_keys(migrated_engine, "workflow_prompt_snapshots") == {
        ("prompt_version_id", "prompt_versions", "id"),
        ("workflow_id", "generation_workflows", "id"),
    }
    assert _foreign_keys(migrated_engine, "workflow_steps") == {
        ("workflow_id", "generation_workflows", "id"),
    }
    assert _foreign_keys(migrated_engine, "model_attempts") == {
        ("step_id", "workflow_steps", "id"),
    }
    assert _foreign_keys(migrated_engine, "workflow_artifacts") == {
        ("step_id", "workflow_steps", "id"),
        ("workflow_id", "generation_workflows", "id"),
    }
    assert _foreign_keys(migrated_engine, "plan_decisions") == {
        ("workflow_id", "generation_workflows", "id"),
    }
    assert _foreign_keys(migrated_engine, "context_sources") == {
        ("project_id", "novel_projects", "id"),
    }
    assert _foreign_keys(migrated_engine, "context_packets") == {
        ("step_id", "workflow_steps", "id"),
        ("workflow_id", "generation_workflows", "id"),
    }
    assert _foreign_keys(migrated_engine, "context_packet_items") == {
        ("packet_id", "context_packets", "id"),
        ("source_id", "context_sources", "id"),
    }
    assert "active_workflow_id" not in {
        key[0] for key in _foreign_keys(migrated_engine, "novel_projects")
    }
    assert "source_workflow_id" not in {
        key[0] for key in _foreign_keys(migrated_engine, "writing_batches")
    }
    assert "active_artifact_id" not in {
        key[0] for key in _foreign_keys(migrated_engine, "workflow_steps")
    }


def test_stage_two_metadata_matches_the_migrated_database(
    migrated_engine: Engine,
) -> None:
    database_tables = set(inspect(migrated_engine).get_table_names())
    assert STAGE_TWO_TABLES <= set(Base.metadata.tables)
    assert set(Base.metadata.tables) <= database_tables

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", str(migrated_engine.url))
    command.check(config)


def test_stage_two_migration_downgrades_and_re_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_root = tmp_path.resolve(strict=True)
    assert temporary_root.is_relative_to(PROJECT_ROOT.resolve())
    database_path = temporary_root / "orchestration-round-trip.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.chdir(temporary_root)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    try:
        command.upgrade(config, "head")
        command.downgrade(config, "0001_foundation")
        engine = create_engine(database_url)
        try:
            names = set(inspect(engine).get_table_names())
            assert not (STAGE_TWO_TABLES & names)
            assert "context_source_fts" not in names
            assert "active_workflow_id" not in _column_map(engine, "novel_projects")
            assert "source_workflow_id" not in _column_map(engine, "writing_batches")
        finally:
            engine.dispose()

        command.upgrade(config, "head")
        engine = create_engine(database_url)
        try:
            assert STAGE_TWO_TABLES <= set(inspect(engine).get_table_names())
            with engine.connect() as connection:
                assert connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='context_source_fts'"
                    )
                ).scalar_one() == "context_source_fts"
        finally:
            engine.dispose()
    finally:
        _remove_database_artifacts(database_path, temporary_root)

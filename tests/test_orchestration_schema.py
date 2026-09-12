from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from ainovel.db import create_engine_for_url
from ainovel.models import (
    Base,
    GenerationWorkflow,
    NovelProject,
    OutlineVersion,
    WorkflowStep,
)


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
    monkeypatch.delenv("AINOVEL_DATABASE_URL", raising=False)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine_for_url(database_url)
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


def _revision_number(engine: Engine) -> str:
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()


def _insert_project_outline_workflow(connection) -> None:
    connection.execute(
        Base.metadata.tables["novel_projects"].insert(),
        {
            "id": "project",
            "title": "Project",
            "target_chars_min": 100,
            "target_chars_max": 200,
        },
    )
    connection.execute(
        Base.metadata.tables["outline_versions"].insert(),
        {
            "id": "outline",
            "project_id": "project",
            "version_number": 1,
            "status": "official",
            "reason": "test",
        },
    )
    connection.execute(
        Base.metadata.tables["generation_workflows"].insert(),
        {
            "id": "workflow",
            "project_id": "project",
            "base_outline_version_id": "outline",
            "provider_name": "fake",
            "model_name": "fake",
            "requested_chapters": 1,
            "status": "PLANNING",
            "planner_input_tokens": 16_000,
            "planner_output_tokens": 4_000,
            "writer_input_tokens": 32_000,
            "writer_output_tokens": 12_000,
            "summarizer_input_tokens": 16_000,
            "summarizer_output_tokens": 4_000,
            "reviewer_input_tokens": 32_000,
            "reviewer_output_tokens": 6_000,
        },
    )


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
    lease_checks = {
        constraint["name"]: constraint["sqltext"]
        for constraint in inspect(migrated_engine).get_check_constraints(
            "workflow_steps"
        )
    }
    assert "lease_owner IS NULL AND lease_expires_at IS NULL" in lease_checks[
        "ck_workflow_steps_lease_pair"
    ]
    assert "lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL" in lease_checks[
        "ck_workflow_steps_lease_pair"
    ]

    item = _column_map(migrated_engine, "context_packet_items")
    assert item["source_id"]["nullable"] is True
    assert item["source_type"]["nullable"] is False
    assert item["source_version"]["nullable"] is False
    assert item["state_scope"]["nullable"] is False
    assert item["source_content_hash"]["nullable"] is False
    assert item["excerpt_start"]["nullable"] is True
    assert item["excerpt_end"]["nullable"] is True
    assert item["trim_reason"]["nullable"] is True
    assert item["selected"]["default"] == "0"
    assert item["required"]["default"] == "0"
    source = _column_map(migrated_engine, "context_sources")
    assert source["source_version"]["type"].length == 64
    assert source["explicitly_requested"]["nullable"] is False
    assert source["explicitly_requested"]["default"] == "0"
    assert source["canonical_source_type"]["nullable"] is True
    assert source["canonical_source_id"]["nullable"] is True
    assert source["excerpt_start"]["nullable"] is True
    assert source["excerpt_end"]["nullable"] is True
    assert item["explicitly_requested"]["nullable"] is False
    assert item["canonical_source_type"]["nullable"] is True
    assert item["canonical_source_id"]["nullable"] is True
    assert Base.metadata.tables["context_sources"].c.source_version.type.length == 64
    for column_name in (
        "source_type",
        "source_version",
        "state_scope",
        "source_content_hash",
    ):
        assert Base.metadata.tables["context_packet_items"].c[column_name].nullable is False
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
    null_ordinal_index = {
        index["name"]: index
        for index in inspect(migrated_engine).get_indexes("workflow_steps")
    }["uq_workflow_steps_one_null_ordinal_kind"]
    assert null_ordinal_index["unique"] == 1
    assert null_ordinal_index["column_names"] == ["workflow_id", "kind"]
    assert str(null_ordinal_index["dialect_options"]["sqlite_where"]) == "ordinal IS NULL"


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


def test_alembic_uses_programmatic_url_when_environment_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_root = tmp_path.resolve(strict=True)
    configured_path = temporary_root / "configured.db"
    fallback_path = temporary_root / "ainovel.db"
    configured_url = f"sqlite+pysqlite:///{configured_path.as_posix()}"
    monkeypatch.chdir(temporary_root)
    monkeypatch.delenv("AINOVEL_DATABASE_URL", raising=False)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", configured_url)

    command.upgrade(config, "head")

    engine = create_engine(configured_url)
    try:
        assert _revision_number(engine) == "0002_orchestration_context"
    finally:
        engine.dispose()
    assert not fallback_path.exists()


def test_alembic_environment_url_intentionally_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_root = tmp_path.resolve(strict=True)
    configured_path = temporary_root / "configured.db"
    environment_path = temporary_root / "environment.db"
    configured_url = f"sqlite+pysqlite:///{configured_path.as_posix()}"
    environment_url = f"sqlite+pysqlite:///{environment_path.as_posix()}"
    monkeypatch.chdir(temporary_root)
    monkeypatch.setenv("AINOVEL_DATABASE_URL", environment_url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", configured_url)

    command.upgrade(config, "head")

    engine = create_engine(environment_url)
    try:
        assert _revision_number(engine) == "0002_orchestration_context"
    finally:
        engine.dispose()
    assert not configured_path.exists()


def test_sqlite_enforces_active_prompt_and_batch_provenance_uniqueness(
    migrated_engine: Engine,
) -> None:
    prompts = Base.metadata.tables["prompt_versions"]
    batches = Base.metadata.tables["writing_batches"]
    with migrated_engine.begin() as connection:
        _insert_project_outline_workflow(connection)
        connection.execute(
            prompts.insert(),
            [
                {
                    "id": "prompt-1",
                    "role": "planner",
                    "version_number": 1,
                    "body": "one",
                    "content_hash": "1" * 64,
                    "active": True,
                    "source": "test",
                },
                {
                    "id": "prompt-2",
                    "role": "planner",
                    "version_number": 2,
                    "body": "two",
                    "content_hash": "2" * 64,
                    "active": False,
                    "source": "test",
                },
            ],
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    prompts.insert(),
                    {
                        "id": "prompt-3",
                        "role": "planner",
                        "version_number": 3,
                        "body": "three",
                        "content_hash": "3" * 64,
                        "active": True,
                        "source": "test",
                    },
                )

        connection.execute(
            batches.insert(),
            [
                {
                    "id": "ordinary-1",
                    "project_id": "project",
                    "base_outline_version_id": "outline",
                    "sequence_number": 1,
                    "planned_chapters": 1,
                    "status": "draft",
                    "source_workflow_id": None,
                },
                {
                    "id": "ordinary-2",
                    "project_id": "project",
                    "base_outline_version_id": "outline",
                    "sequence_number": 2,
                    "planned_chapters": 1,
                    "status": "draft",
                    "source_workflow_id": None,
                },
                {
                    "id": "workflow-batch",
                    "project_id": "project",
                    "base_outline_version_id": "outline",
                    "sequence_number": 3,
                    "planned_chapters": 1,
                    "status": "draft",
                    "source_workflow_id": "workflow",
                },
            ],
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    batches.insert(),
                    {
                        "id": "duplicate-workflow-batch",
                        "project_id": "project",
                        "base_outline_version_id": "outline",
                        "sequence_number": 4,
                        "planned_chapters": 1,
                        "status": "draft",
                        "source_workflow_id": "workflow",
                    },
                )


def test_sqlite_enforces_declared_foreign_keys(migrated_engine: Engine) -> None:
    with migrated_engine.begin() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        with pytest.raises(IntegrityError):
            connection.execute(
                Base.metadata.tables["workflow_steps"].insert(),
                {
                    "id": "orphan",
                    "workflow_id": "missing",
                    "kind": "PLANNING",
                    "position": 0,
                    "status": "PENDING",
                },
            )


def test_sqlite_enforces_null_and_non_null_workflow_step_ordinals(
    migrated_engine: Engine,
) -> None:
    steps = Base.metadata.tables["workflow_steps"]
    with migrated_engine.begin() as connection:
        _insert_project_outline_workflow(connection)
        connection.execute(
            steps.insert(),
            {
                "id": "singleton",
                "workflow_id": "workflow",
                "kind": "PLANNING",
                "ordinal": None,
                "position": 0,
                "status": "PENDING",
            },
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    steps.insert(),
                    {
                        "id": "duplicate-singleton",
                        "workflow_id": "workflow",
                        "kind": "PLANNING",
                        "ordinal": None,
                        "position": 1,
                        "status": "PENDING",
                    },
                )

        connection.execute(
            steps.insert(),
            {
                "id": "chapter-1",
                "workflow_id": "workflow",
                "kind": "WRITING",
                "ordinal": 1,
                "position": 2,
                "status": "PENDING",
            },
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    steps.insert(),
                    {
                        "id": "duplicate-chapter-1",
                        "workflow_id": "workflow",
                        "kind": "WRITING",
                        "ordinal": 1,
                        "position": 3,
                        "status": "PENDING",
                    },
                )


@pytest.mark.parametrize(
    ("lease_owner", "lease_expires_at"),
    [
        ("worker", None),
        (None, datetime(2026, 9, 4, tzinfo=timezone.utc)),
    ],
)
def test_sqlite_rejects_half_populated_step_leases(
    migrated_engine: Engine,
    lease_owner: str | None,
    lease_expires_at: datetime | None,
) -> None:
    with migrated_engine.begin() as connection:
        _insert_project_outline_workflow(connection)
        with pytest.raises(IntegrityError):
            connection.execute(
                Base.metadata.tables["workflow_steps"].insert(),
                {
                    "id": "leased-step",
                    "workflow_id": "workflow",
                    "kind": "PLANNING",
                    "position": 0,
                    "status": "RUNNING",
                    "lease_owner": lease_owner,
                    "lease_expires_at": lease_expires_at,
                },
            )


def _workflow_objects() -> tuple[NovelProject, OutlineVersion, GenerationWorkflow]:
    project = NovelProject(
        id="project",
        title="Project",
        target_chars_min=100,
        target_chars_max=200,
    )
    outline = OutlineVersion(
        id="outline",
        project_id=project.id,
        version_number=1,
        status="official",
        reason="test",
    )
    workflow = GenerationWorkflow(
        id="workflow",
        project_id=project.id,
        base_outline_version_id=outline.id,
        provider_name="fake",
        model_name="fake",
        requested_chapters=1,
        status="PLANNING",
        planner_input_tokens=16_000,
        planner_output_tokens=4_000,
        writer_input_tokens=32_000,
        writer_output_tokens=12_000,
        summarizer_input_tokens=16_000,
        summarizer_output_tokens=4_000,
        reviewer_input_tokens=32_000,
        reviewer_output_tokens=6_000,
    )
    return project, outline, workflow


def _persist_workflow(session: Session) -> None:
    project, outline, workflow = _workflow_objects()
    session.add(project)
    session.flush()
    session.add(outline)
    session.flush()
    session.add(workflow)
    session.flush()


def test_sqlite_lease_round_trip_normalizes_to_aware_utc(
    migrated_engine: Engine,
) -> None:
    non_utc = datetime(2026, 9, 4, 15, 30, tzinfo=timezone(timedelta(hours=8)))
    with Session(migrated_engine) as session:
        _persist_workflow(session)
        session.add(
            WorkflowStep(
                id="step",
                workflow_id="workflow",
                kind="PLANNING",
                position=0,
                status="RUNNING",
                lease_owner="worker",
                lease_expires_at=non_utc,
            )
        )
        session.commit()
        session.expire_all()

        stored = session.get(WorkflowStep, "step")
        assert stored is not None
        assert stored.lease_expires_at == datetime(
            2026, 9, 4, 7, 30, tzinfo=timezone.utc
        )
        assert stored.lease_expires_at.tzinfo is timezone.utc


def test_sqlite_lease_rejects_naive_datetime(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        _persist_workflow(session)
        session.add(
            WorkflowStep(
                id="step",
                workflow_id="workflow",
                kind="PLANNING",
                position=0,
                status="RUNNING",
                lease_owner="worker",
                lease_expires_at=datetime(2026, 9, 4, 7, 30),
            )
        )

        with pytest.raises(StatementError, match="timezone-aware"):
            session.flush()


def test_stage_two_migration_downgrades_and_re_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_root = tmp_path.resolve(strict=True)
    assert temporary_root.is_relative_to(PROJECT_ROOT.resolve())
    database_path = temporary_root / "orchestration-round-trip.db"
    ambient_database_path = temporary_root / "ambient-database-must-not-be-used.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.setenv(
        "AINOVEL_DATABASE_URL",
        f"sqlite+pysqlite:///{ambient_database_path.as_posix()}",
    )
    monkeypatch.delenv("AINOVEL_DATABASE_URL", raising=False)
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
            source = _column_map(engine, "context_sources")
            item = _column_map(engine, "context_packet_items")
            assert source["source_version"]["type"].length == 64
            assert {
                "source_type",
                "source_version",
                "state_scope",
                "source_content_hash",
            } <= set(item)
            with engine.connect() as connection:
                assert connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='context_source_fts'"
                    )
                ).scalar_one() == "context_source_fts"
        finally:
            engine.dispose()
        assert not ambient_database_path.exists()
    finally:
        _remove_database_artifacts(database_path, temporary_root)
        _remove_database_artifacts(ambient_database_path, temporary_root)

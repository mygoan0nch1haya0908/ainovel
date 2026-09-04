from copy import deepcopy

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from ainovel.agents.prompts import AGENT_PARAMETERS, AGENT_SCHEMAS, BUILTIN_PROMPTS
from ainovel.models import GenerationWorkflow, PromptVersion
from ainovel.services.prompts import PromptService


@pytest.fixture
def workflow(session, project, official_outline):
    row = GenerationWorkflow(
        id="prompt-workflow",
        project_id=project.id,
        base_outline_version_id=official_outline.id,
        provider_name="fake",
        model_name="fake-model",
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
    session.add(row)
    session.commit()
    return row


def test_builtin_seeding_is_idempotent_by_role_and_content(session) -> None:
    service = PromptService(session)

    first = service.ensure_builtins()
    second = service.ensure_builtins()

    assert set(BUILTIN_PROMPTS) == {
        "batch_planner",
        "chapter_writer",
        "chapter_summarizer",
        "batch_reviewer",
    }
    assert [row.id for row in second] == [row.id for row in first]
    assert session.scalar(select(func.count()).select_from(PromptVersion)) == 4
    assert {row.role for row in first if row.active} == set(BUILTIN_PROMPTS)


@pytest.mark.parametrize("body", ["", " \n\t "])
def test_create_version_rejects_blank_prompt_body(session, body: str) -> None:
    with pytest.raises(ValueError, match="prompt body is required"):
        PromptService(session).create_version("chapter_writer", body, "author")


def test_create_version_rejects_unknown_role(session) -> None:
    with pytest.raises(ValueError, match="unknown prompt role"):
        PromptService(session).create_version("unknown", "instructions", "author")


def test_create_version_retries_an_allocation_collision(session, monkeypatch) -> None:
    service = PromptService(session)
    service.create_version("chapter_writer", "first instructions", "author")
    original_flush = session.flush
    flush_calls = 0

    def collide_once(*args, **kwargs):
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 1:
            raise IntegrityError("insert", {}, Exception("duplicate version"))
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(session, "flush", collide_once)

    version = service.create_version("chapter_writer", "second instructions", "author")

    assert flush_calls >= 2
    assert version.version_number == 2


def test_activation_leaves_exactly_one_active_version_for_role(session) -> None:
    service = PromptService(session)
    service.ensure_builtins()
    replacement = service.create_version("chapter_writer", "replacement instructions", "author")

    active = service.activate(replacement.id)

    active_versions = session.scalars(
        select(PromptVersion).where(
            PromptVersion.role == "chapter_writer", PromptVersion.active.is_(True)
        )
    ).all()
    assert active.id == replacement.id
    assert [row.id for row in active_versions] == [replacement.id]


def test_workflow_snapshot_does_not_change_when_prompt_is_replaced(session, workflow) -> None:
    service = PromptService(session)
    service.ensure_builtins()
    before = service.snapshot(workflow.id, AGENT_SCHEMAS, AGENT_PARAMETERS)
    session.commit()
    replacement = service.create_version("chapter_writer", "新的完整主笔提示词", "author")
    service.activate(replacement.id)

    after = service.list_snapshots(workflow.id)

    assert [(row.prompt_body, row.output_schema) for row in after] == [
        (row.prompt_body, row.output_schema) for row in before
    ]


def test_snapshot_copies_parameters_and_creates_exactly_four_rows(session, workflow) -> None:
    service = PromptService(session)
    service.ensure_builtins()
    parameters = deepcopy(AGENT_PARAMETERS)

    snapshots = service.snapshot(workflow.id, AGENT_SCHEMAS, parameters)
    parameters["chapter_writer"]["max_input_tokens"] = 1
    session.commit()

    assert len(snapshots) == 4
    assert len(service.list_snapshots(workflow.id)) == 4
    writer = next(row for row in snapshots if row.role == "chapter_writer")
    assert writer.parameters == {"max_input_tokens": 32_000, "max_output_tokens": 12_000}


def test_snapshot_flushes_without_committing_the_callers_transaction(session, workflow) -> None:
    service = PromptService(session)
    service.ensure_builtins()

    snapshots = service.snapshot(workflow.id, AGENT_SCHEMAS, AGENT_PARAMETERS)
    session.rollback()

    assert len(snapshots) == 4
    assert service.list_snapshots(workflow.id) == []

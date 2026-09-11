from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from threading import Barrier

import pytest
from sqlalchemy import event, func, select, text, update

from ainovel.agents.contracts import BatchPlanDraft, ChapterPlan
from ainovel.context import RequiredContextOverflow
from ainovel.models import (
    AuditEvent,
    GenerationWorkflow,
    ModelAttempt,
    NovelProject,
    PlanDecision,
    PromptVersion,
    WorkflowArtifact,
    WorkflowPromptSnapshot,
    WorkflowStep,
    WritingBatch,
)
from ainovel.providers.contracts import (
    ModelResponse,
    ProviderAuthenticationError,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
)
from ainovel.providers.fake import FakeProvider
from ainovel.services.context import ContextIndexService
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService
from ainovel.services.prompts import PromptService
from ainovel.services.workflows import (
    DEFAULT_BUDGETS,
    PROVIDER_NAMES,
    STEP_STATUSES,
    WORKFLOW_STATUSES,
    WorkflowBudgets,
    WorkflowService,
)


class FrozenClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 5, 4, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def session_factory(client):
    return client.app.state.session_factory


@pytest.fixture
def ready_project(session, project, official_outline) -> NovelProject:
    ProjectService(session).add_constitution(
        project.id,
        {"genre": "historical fantasy", "voice": "close third"},
        author_approved=True,
    )
    session.expire_all()
    ready = session.get(NovelProject, project.id)
    assert ready is not None
    assert ready.official_outline_version_id == official_outline.id
    return ready


@pytest.fixture
def workflow(session, ready_project, clock) -> GenerationWorkflow:
    return WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 2, DEFAULT_BUDGETS
    )


def _response(
    *, input_tokens: int | None = 11, output_tokens: int | None = 7
) -> ModelResponse:
    return ModelResponse(
        structured={"ok": True},
        text=None,
        provider_response_id="response-1",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=23,
    )


def _plan_payload(chapter_count: int) -> dict[str, object]:
    return BatchPlanDraft(
        chapters=[
            ChapterPlan(
                ordinal=ordinal,
                title=f"Chapter {ordinal}",
                goal=f"Goal {ordinal}",
                ending_hook=f"Hook {ordinal}",
            )
            for ordinal in range(1, chapter_count + 1)
        ]
    ).model_dump(mode="json")


def _complete_plan(
    service: WorkflowService, workflow: GenerationWorkflow, worker_id: str = "planner"
) -> WorkflowArtifact:
    step = service.claim_step(workflow.id, {"PLANNING"}, worker_id)
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)
    return service.complete_attempt(
        attempt.id,
        _response(),
        {"kind": "batch_plan", "payload": _plan_payload(workflow.requested_chapters)},
    )


def _prepare_reviewer_step(
    session, service: WorkflowService, workflow: GenerationWorkflow
) -> WorkflowStep:
    _complete_plan(service, workflow)
    service.approve_plan(workflow.id, "author")
    reviewer = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "REVIEWING",
        )
    )
    assert reviewer is not None
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status="REVIEWING_BATCH", current_position=reviewer.position)
    )
    session.execute(
        update(WorkflowStep)
        .where(WorkflowStep.id == reviewer.id)
        .values(status="PENDING")
    )
    session.commit()
    return reviewer


def test_workflow_contract_constants_are_exact_and_budgets_are_frozen() -> None:
    assert PROVIDER_NAMES == frozenset({"fake", "ollama", "openai"})
    assert WORKFLOW_STATUSES == frozenset(
        {
            "PREPARING",
            "PLANNING",
            "AWAITING_PLAN_APPROVAL",
            "GENERATING_CHAPTERS",
            "REVIEWING_BATCH",
            "CREATING_CANDIDATE_BATCH",
            "AWAITING_CONTENT_APPROVAL",
            "PAUSED_PROVIDER",
            "PAUSED_CONTEXT_OVERFLOW",
            "PAUSED_ATTEMPTS",
            "PAUSED_REVIEW",
            "PAUSED_STALE_VERSION",
            "COMPLETED",
            "REJECTED",
            "CANCELLED",
            "FAILED",
        }
    )
    assert STEP_STATUSES == frozenset(
        {"PENDING", "RUNNING", "COMPLETED", "PAUSED", "FAILED"}
    )
    assert DEFAULT_BUDGETS == WorkflowBudgets()
    assert DEFAULT_BUDGETS == WorkflowBudgets(
        planner_input=16_000,
        planner_output=4_000,
        writer_input=32_000,
        writer_output=12_000,
        summarizer_input=16_000,
        summarizer_output=4_000,
        reviewer_input=32_000,
        reviewer_output=6_000,
    )
    with pytest.raises(FrozenInstanceError):
        DEFAULT_BUDGETS.planner_input = 1  # type: ignore[misc]


@pytest.mark.parametrize("requested_chapters", [0, 6, True, 1.5])
def test_start_rejects_invalid_chapter_count_before_ownership(
    session, ready_project, requested_chapters
) -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        WorkflowService(session).start(
            ready_project.id,
            "fake",
            "scripted",
            requested_chapters,
            DEFAULT_BUDGETS,
        )
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


def test_start_rejects_unknown_provider_before_database_side_effects(
    session, ready_project
) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        WorkflowService(session).start(
            ready_project.id, "mystery", "model", 1, DEFAULT_BUDGETS
        )
    assert session.scalar(select(func.count()).select_from(PromptVersion)) == 0
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


@pytest.mark.parametrize(
    "budgets",
    [
        replace(DEFAULT_BUDGETS, planner_input=0),
        replace(DEFAULT_BUDGETS, reviewer_output=-1),
        replace(DEFAULT_BUDGETS, writer_input=True),
    ],
)
def test_start_rejects_non_positive_or_non_exact_integer_budgets(
    session, ready_project, budgets
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        WorkflowService(session).start(
            ready_project.id, "fake", "scripted", 1, budgets
        )
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


def test_start_requires_approved_constitution(session, project, official_outline) -> None:
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, False)
    with pytest.raises(ValueError, match="approved constitution"):
        WorkflowService(session).start(
            project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
        )
    assert not session.in_transaction()


def test_start_rejects_stale_official_outline_pointer(session, ready_project) -> None:
    candidate = OutlineService(session).create_candidate(
        ready_project.id,
        [
            OutlineNodeInput(
                key="new-book",
                parent_key=None,
                kind="book",
                title="Unapproved replacement",
                order=0,
            )
        ],
        reason="not approved",
    )
    session.execute(
        update(NovelProject)
        .where(NovelProject.id == ready_project.id)
        .values(official_outline_version_id=candidate.id)
    )
    session.commit()

    with pytest.raises(ValueError, match="current official outline"):
        WorkflowService(session).start(
            ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
        )
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


def test_start_rejects_active_batch(session, ready_project) -> None:
    ready_project.active_batch_id = "existing-batch"
    session.commit()

    with pytest.raises(ValueError, match="active batch"):
        WorkflowService(session).start(
            ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
        )
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


def test_start_atomically_owns_project_snapshots_prompts_and_creates_planning_step(
    session, ready_project
) -> None:
    budgets = replace(DEFAULT_BUDGETS, planner_input=12_345, reviewer_output=5_432)
    workflow = WorkflowService(session).start(
        ready_project.id, "openai", "gpt-test", 3, budgets
    )

    session.expire_all()
    project = session.get(NovelProject, ready_project.id)
    steps = session.scalars(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    ).all()
    snapshots = session.scalars(
        select(WorkflowPromptSnapshot).where(
            WorkflowPromptSnapshot.workflow_id == workflow.id
        )
    ).all()
    audits = session.scalars(
        select(AuditEvent).where(
            AuditEvent.entity_type == "generation_workflow",
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "workflow_started",
        )
    ).all()

    assert project.active_workflow_id == workflow.id
    assert workflow.status == "PLANNING"
    assert workflow.revision == 1
    assert workflow.planner_input_tokens == 12_345
    assert workflow.reviewer_output_tokens == 5_432
    assert [(row.kind, row.ordinal, row.position, row.status) for row in steps] == [
        ("PLANNING", None, 0, "PENDING")
    ]
    assert len(snapshots) == 4
    assert next(row for row in snapshots if row.role == "batch_planner").parameters == {
        "max_input_tokens": 12_345,
        "max_output_tokens": 4_000,
    }
    assert next(row for row in snapshots if row.role == "batch_reviewer").parameters == {
        "max_input_tokens": 32_000,
        "max_output_tokens": 5_432,
    }
    assert len(audits) == 1


def test_start_rolls_back_ownership_workflow_snapshots_step_and_audit_together(
    session, ready_project, monkeypatch
) -> None:
    original_snapshot = PromptService.snapshot

    def fail_after_snapshot(self, workflow_id, schemas, parameters):
        original_snapshot(self, workflow_id, schemas, parameters)
        raise RuntimeError("forced snapshot failure")

    monkeypatch.setattr(PromptService, "snapshot", fail_after_snapshot)

    with pytest.raises(RuntimeError, match="forced snapshot failure"):
        WorkflowService(session).start(
            ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
        )

    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None
    assert session.scalar(select(func.count()).select_from(GenerationWorkflow)) == 0
    assert session.scalar(select(func.count()).select_from(WorkflowPromptSnapshot)) == 0
    assert session.scalar(select(func.count()).select_from(WorkflowStep)) == 0
    assert session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(AuditEvent.entity_type == "generation_workflow")
    ) == 0


def test_two_sessions_start_only_one_project_workflow(
    session_factory, ready_project
) -> None:
    left = session_factory()
    right = session_factory()
    try:
        first = WorkflowService(left).start(
            ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS
        )
        with pytest.raises(ValueError, match="active workflow"):
            WorkflowService(right).start(
                ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS
            )
        left.expire_all()
        assert first.id == left.get(NovelProject, ready_project.id).active_workflow_id
    finally:
        left.close()
        right.close()


def test_barrier_two_session_start_has_one_winner_workflow_and_audit(
    client, session_factory, ready_project
) -> None:
    project_id = ready_project.id
    ownership_barrier = Barrier(2)

    def synchronize_ownership(_conn, _cursor, statement, _parameters, _context, _many):
        normalized = " ".join(statement.casefold().split())
        if normalized.startswith("update novel_projects set active_workflow_id="):
            ownership_barrier.wait(timeout=5)

    def start():
        with session_factory() as independent_session:
            try:
                workflow = WorkflowService(independent_session).start(
                    project_id, "fake", "scripted", 2, DEFAULT_BUDGETS
                )
            except ValueError as error:
                return "error", str(error)
            return "ok", workflow.id

    event.listen(client.app.state.engine, "before_cursor_execute", synchronize_ownership)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            outcomes = [
                future.result(timeout=15)
                for future in [workers.submit(start), workers.submit(start)]
            ]
    finally:
        event.remove(
            client.app.state.engine, "before_cursor_execute", synchronize_ownership
        )

    assert [kind for kind, _value in outcomes].count("ok") == 1
    assert [kind for kind, _value in outcomes].count("error") == 1
    with session_factory() as verify_session:
        assert verify_session.scalar(
            select(func.count())
            .select_from(GenerationWorkflow)
            .where(GenerationWorkflow.project_id == project_id)
        ) == 1
        assert verify_session.scalar(
            select(func.count())
            .select_from(AuditEvent)
            .where(
                AuditEvent.project_id == project_id,
                AuditEvent.action == "workflow_started",
            )
        ) == 1


def test_claim_uses_status_and_revision_cas_and_commits_finite_utc_lease(
    session, session_factory, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    assert service.claim_step(workflow.id, {"GENERATING_CHAPTERS"}, "worker-a") is None

    claimed = service.claim_step(
        workflow.id, {"PLANNING"}, "worker-a", lease_seconds=30
    )

    assert claimed is not None
    assert claimed.status == "RUNNING"
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_expires_at == clock.now() + timedelta(seconds=30)
    assert claimed.lease_expires_at.tzinfo is timezone.utc
    assert claimed.revision == 2
    assert not session.in_transaction()

    with session_factory() as competing_session:
        assert (
            WorkflowService(competing_session, clock=clock).claim_step(
                workflow.id, {"PLANNING"}, "worker-b"
            )
            is None
        )


def test_stale_session_cannot_claim_a_step_that_another_session_claimed(
    session_factory, workflow, clock
) -> None:
    left = session_factory()
    right = session_factory()
    try:
        stale = left.scalar(
            select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
        )
        assert stale is not None and stale.status == "PENDING"
        winner = WorkflowService(right, clock=clock).claim_step(
            workflow.id, {"PLANNING"}, "worker-b"
        )
        assert winner is not None

        assert (
            WorkflowService(left, clock=clock).claim_step(
                workflow.id, {"PLANNING"}, "worker-a"
            )
            is None
        )
        left.expire_all()
        refreshed = left.get(WorkflowStep, stale.id)
        assert refreshed.lease_owner == "worker-b"
        assert refreshed.revision == 2
    finally:
        left.close()
        right.close()


def test_claim_pauses_when_the_official_outline_has_changed(
    session, workflow, ready_project, clock
) -> None:
    candidate = OutlineService(session).create_candidate(
        ready_project.id,
        [
            OutlineNodeInput(
                key="replacement",
                parent_key=None,
                kind="book",
                title="Replacement",
                order=0,
            )
        ],
        reason="replacement",
    )
    OutlineService(session).approve(candidate.id)

    assert (
        WorkflowService(session, clock=clock).claim_step(
            workflow.id, {"PLANNING"}, "worker-a"
        )
        is None
    )
    session.expire_all()
    stored_workflow = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert stored_workflow.status == "PAUSED_STALE_VERSION"
    assert step.status == "PAUSED"
    assert ready_project.id == stored_workflow.project_id


def test_only_expired_unfinished_claim_is_recovered(session, workflow, clock) -> None:
    service = WorkflowService(session, clock=clock)
    claimed = service.claim_step(
        workflow.id, {"PLANNING"}, "worker-a", lease_seconds=30
    )
    assert claimed is not None
    assert service.recover_expired_claims(workflow.id, clock.now()) == 0
    clock.advance(seconds=31)
    assert service.recover_expired_claims(workflow.id, clock.now()) == 1

    session.expire_all()
    recovered = session.get(WorkflowStep, claimed.id)
    assert recovered.status == "PENDING"
    assert recovered.lease_owner is None
    assert recovered.lease_expires_at is None
    assert recovered.revision == 3


def test_completed_or_active_artifact_step_is_never_recovered(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    artifact = _complete_plan(service, workflow)
    clock.advance(seconds=301)

    assert service.recover_expired_claims(workflow.id, clock.now()) == 0
    session.expire_all()
    step = session.get(WorkflowStep, artifact.step_id)
    assert step.status == "COMPLETED"
    assert step.active_artifact_id == artifact.id


def test_running_step_with_active_artifact_is_not_recovered(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    artifact = _complete_plan(service, workflow)
    step = session.get(WorkflowStep, artifact.step_id)
    session.execute(
        update(WorkflowStep)
        .where(WorkflowStep.id == step.id)
        .values(
            status="RUNNING",
            lease_owner="stale-worker",
            lease_expires_at=clock.now() - timedelta(seconds=1),
        )
    )
    session.commit()

    assert service.recover_expired_claims(workflow.id, clock.now()) == 0
    session.expire_all()
    stored = session.get(WorkflowStep, step.id)
    assert stored.status == "RUNNING"
    assert stored.active_artifact_id == artifact.id


def test_attempts_are_monotonic_and_second_retryable_failure_pauses(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    first_step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert first_step is not None
    first = service.record_attempt_start(first_step.id, "1" * 64, claim_revision=first_step.revision)
    retrying = service.fail_attempt(first.id, ProviderTimeout("secret first timeout"))
    assert first.attempt_number == 1
    assert retrying.status == "PLANNING"

    second_step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert second_step is not None
    second = service.record_attempt_start(second_step.id, "2" * 64, claim_revision=second_step.revision)
    paused = service.fail_attempt(second.id, ProviderProtocolError("secret payload"))

    assert second.attempt_number == 2
    assert paused.status == "PAUSED_ATTEMPTS"
    session.expire_all()
    stored_step = session.get(WorkflowStep, first_step.id)
    assert stored_step.attempt_count == 2
    assert stored_step.status == "PAUSED"
    assert stored_step.lease_owner is None
    assert [row.attempt_number for row in session.scalars(
        select(ModelAttempt)
        .where(ModelAttempt.step_id == stored_step.id)
        .order_by(ModelAttempt.attempt_number)
    )] == [1, 2]


def test_claim_and_attempt_start_commit_before_external_provider_execution(
    session, session_factory, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)
    assert attempt.status == "RUNNING"
    assert not session.in_transaction()

    with session_factory() as independent_session:
        independent_session.add(
            AuditEvent(
                id="external-call-overlap",
                project_id=workflow.project_id,
                entity_type="provider_probe",
                entity_id=workflow.id,
                action="independent_write",
                actor="test",
                details={},
            )
        )
        independent_session.commit()

    assert not session.in_transaction()


def test_complete_attempt_atomically_persists_artifact_usage_and_plan_gate(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)

    artifact = service.complete_attempt(
        attempt.id,
        _response(input_tokens=101, output_tokens=37),
        {"kind": "batch_plan", "payload": _plan_payload(2)},
    )

    assert not session.in_transaction()
    session.expire_all()
    stored_attempt = session.get(ModelAttempt, attempt.id)
    stored_step = session.get(WorkflowStep, step.id)
    stored_workflow = session.get(GenerationWorkflow, workflow.id)
    assert stored_attempt.status == "COMPLETED"
    assert stored_attempt.provider_response_id == "response-1"
    assert stored_attempt.input_tokens == 101
    assert stored_attempt.output_tokens == 37
    assert stored_step.status == "COMPLETED"
    assert stored_step.active_artifact_id == artifact.id
    assert stored_step.lease_owner is None
    assert stored_workflow.status == "AWAITING_PLAN_APPROVAL"
    assert stored_workflow.actual_input_tokens == 101
    assert stored_workflow.actual_output_tokens == 37
    assert len(artifact.content_hash) == 64


def test_complete_attempt_rolls_back_all_effects_when_artifact_is_invalid(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)

    with pytest.raises((TypeError, ValueError)):
        service.complete_attempt(
            attempt.id,
            _response(input_tokens=101, output_tokens=37),
            {"kind": "batch_plan", "payload": {"bad": object()}},
        )

    assert not session.in_transaction()
    session.expire_all()
    assert session.get(ModelAttempt, attempt.id).status == "RUNNING"
    assert session.get(WorkflowStep, step.id).status == "RUNNING"
    stored_workflow = session.get(GenerationWorkflow, workflow.id)
    assert stored_workflow.status == "PLANNING"
    assert stored_workflow.actual_input_tokens == 0
    assert stored_workflow.actual_output_tokens == 0
    assert session.scalar(select(func.count()).select_from(WorkflowArtifact)) == 0


def test_complete_attempt_rejects_unvalidated_hidden_reasoning_payload(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)

    with pytest.raises(ValueError, match="artifact payload failed validation"):
        service.complete_attempt(
            attempt.id,
            _response(),
            {
                "kind": "batch_plan",
                "payload": {
                    "chapters": [],
                    "hidden_reasoning": "must never be persisted",
                },
            },
        )

    assert not session.in_transaction()
    assert session.scalar(select(func.count()).select_from(WorkflowArtifact)) == 0


def test_writer_artifact_rejects_text_or_count_that_disagrees_with_validated_payload(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    service.approve_plan(workflow.id, "author")
    step = service.claim_step(
        workflow.id, {"GENERATING_CHAPTERS"}, "writer-a"
    )
    assert step is not None and step.kind == "WRITING"
    attempt = service.record_attempt_start(step.id, "b" * 64, claim_revision=step.revision)

    with pytest.raises(ValueError, match="artifact metadata does not match payload"):
        service.complete_attempt(
            attempt.id,
            _response(),
            {
                "kind": "chapter_draft",
                "payload": {"title": "Chapter 1", "body": "甲" * 4_500},
                "text_content": "tampered body",
                "visible_char_count": 1,
            },
        )

    assert not session.in_transaction()
    assert session.scalar(
        select(func.count())
        .select_from(WorkflowArtifact)
        .where(WorkflowArtifact.kind == "chapter_draft")
    ) == 0


def test_reviewer_evidence_artifact_is_immutable_nonfinal_and_reclaimable(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    service.approve_plan(workflow.id, "author")
    steps = session.scalars(
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow.id)
        .order_by(WorkflowStep.position)
    ).all()
    reviewer = next(row for row in steps if row.kind == "REVIEWING")
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status="REVIEWING_BATCH", current_position=reviewer.position)
    )
    session.execute(
        update(WorkflowStep)
        .where(WorkflowStep.id == reviewer.id)
        .values(status="PENDING")
    )
    session.commit()

    claimed = service.claim_step(workflow.id, {"REVIEWING_BATCH"}, "reviewer-a")
    assert claimed is not None and claimed.id == reviewer.id
    attempt = service.record_attempt_start(claimed.id, "c" * 64, claim_revision=claimed.revision)
    evidence = service.complete_attempt(
        attempt.id,
        _response(),
        {
            "kind": "batch_review",
            "payload": {
                "passed": False,
                "issues": ["timeline unclear"],
                "evidence_queries": ["chapter 1 bridge"],
            },
        },
        finalize_step=False,
    )

    session.expire_all()
    stored = session.get(WorkflowStep, reviewer.id)
    assert evidence.id is not None
    assert stored.status == "PENDING"
    assert stored.active_artifact_id is None
    assert stored.lease_owner is None
    assert session.get(GenerationWorkflow, workflow.id).status == "REVIEWING_BATCH"
    second_claim = service.claim_step(
        workflow.id, {"REVIEWING_BATCH"}, "reviewer-a"
    )
    assert second_claim is not None
    second_attempt = service.record_attempt_start(second_claim.id, "d" * 64, claim_revision=second_claim.revision)
    blocked = service.complete_attempt(
        second_attempt.id,
        _response(),
        {
            "kind": "batch_review",
            "payload": {
                "passed": False,
                "issues": ["timeline conflict confirmed"],
                "evidence_queries": [],
            },
        },
    )
    session.expire_all()
    stored = session.get(WorkflowStep, reviewer.id)
    assert session.get(GenerationWorkflow, workflow.id).status == "PAUSED_REVIEW"
    assert stored.status == "PAUSED"
    assert stored.active_artifact_id == blocked.id
    assert session.get(WorkflowArtifact, evidence.id).payload["evidence_queries"] == [
        "chapter 1 bridge"
    ]


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            ProviderUnavailable("http://token@example.invalid unavailable"),
            "PAUSED_PROVIDER",
            "provider_unavailable",
        ),
        (
            ProviderAuthenticationError("sk-top-secret rejected"),
            "PAUSED_PROVIDER",
            "provider_authentication",
        ),
    ],
)
def test_nonretryable_provider_failures_pause_with_secret_safe_categories(
    session, workflow, clock, error, expected_status, expected_code
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    attempt = service.record_attempt_start(step.id, "a" * 64, claim_revision=step.revision)

    failed = service.fail_attempt(attempt.id, error)

    assert failed.status == expected_status
    assert failed.last_error_code == expected_code
    persisted = " ".join(
        value or ""
        for value in (
            failed.last_error_detail,
            session.get(ModelAttempt, attempt.id).error_detail,
        )
    )
    assert "secret" not in persisted.casefold()
    assert "sk-" not in persisted.casefold()
    assert "token@" not in persisted.casefold()


def test_plan_approval_requires_active_plan_then_creates_strict_step_order_and_audit(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    assert [row.kind for row in session.scalars(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )] == ["PLANNING"]
    _complete_plan(service, workflow)

    approved = service.approve_plan(workflow.id, "author@example")

    assert approved.status == "GENERATING_CHAPTERS"
    steps = session.scalars(
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow.id)
        .order_by(WorkflowStep.position)
    ).all()
    assert [(row.kind, row.ordinal, row.position, row.status) for row in steps] == [
        ("PLANNING", None, 0, "COMPLETED"),
        ("WRITING", 1, 1, "PENDING"),
        ("SUMMARIZING", 1, 2, "PENDING"),
        ("WRITING", 2, 3, "PENDING"),
        ("SUMMARIZING", 2, 4, "PENDING"),
        ("REVIEWING", None, 5, "PENDING"),
        ("CREATING_CANDIDATE_BATCH", None, 6, "PENDING"),
    ]
    decision = session.scalar(
        select(PlanDecision).where(PlanDecision.workflow_id == workflow.id)
    )
    assert (decision.decision, decision.reason, decision.actor) == (
        "approved",
        "",
        "author@example",
    )
    audit = session.scalar(
        select(AuditEvent).where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "plan_approved",
        )
    )
    assert audit is not None and audit.actor == "author@example"
    with pytest.raises(ValueError, match="plan approval conflict"):
        service.approve_plan(workflow.id, "second-author")


def test_completed_steps_advance_through_writer_summary_review_in_strict_order(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    service.approve_plan(workflow.id, "author")

    expected = [
        ("WRITING", 1, "GENERATING_CHAPTERS"),
        ("SUMMARIZING", 1, "GENERATING_CHAPTERS"),
        ("WRITING", 2, "GENERATING_CHAPTERS"),
        ("SUMMARIZING", 2, "REVIEWING_BATCH"),
    ]
    for index, (kind, ordinal, status_after) in enumerate(expected, start=1):
        claimed = service.claim_step(
            workflow.id,
            {"GENERATING_CHAPTERS"},
            "sequential-worker",
        )
        assert claimed is not None
        assert (claimed.kind, claimed.ordinal) == (kind, ordinal)
        attempt = service.record_attempt_start(claimed.id, f"{index:x}" * 64, claim_revision=claimed.revision)
        if kind == "WRITING":
            artifact = {
                "kind": "chapter_draft",
                "payload": {"title": f"Chapter {ordinal}", "body": "甲" * 4_500},
            }
        else:
            artifact = {
                "kind": "chapter_summary_delta",
                "payload": {
                    "summary": f"Summary {ordinal}",
                    "state_delta": {"ordinal": ordinal},
                },
            }
        service.complete_attempt(attempt.id, _response(), artifact)
        session.expire_all()
        assert session.get(GenerationWorkflow, workflow.id).status == status_after

    reviewer = service.claim_step(
        workflow.id, {"REVIEWING_BATCH"}, "sequential-worker"
    )
    assert reviewer is not None and reviewer.kind == "REVIEWING"
    review_attempt = service.record_attempt_start(reviewer.id, "f" * 64, claim_revision=reviewer.revision)
    service.complete_attempt(
        review_attempt.id,
        _response(),
        {
            "kind": "batch_review",
            "payload": {"passed": True, "issues": [], "evidence_queries": []},
        },
    )

    session.expire_all()
    stored_workflow = session.get(GenerationWorkflow, workflow.id)
    current = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.position == stored_workflow.current_position,
        )
    )
    assert stored_workflow.status == "CREATING_CANDIDATE_BATCH"
    assert current.kind == "CREATING_CANDIDATE_BATCH"
    assert current.status == "PENDING"

    claimed_candidate = service.claim_step(
        workflow.id, {"CREATING_CANDIDATE_BATCH"}, "batch-worker"
    )
    assert claimed_candidate is not None and claimed_candidate.id == current.id
    batch = WritingBatch(
        id="strict-order-batch",
        project_id=workflow.project_id,
        base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=1,
        planned_chapters=workflow.requested_chapters,
        status="ready_for_review",
        source_workflow_id=workflow.id,
    )
    session.add(batch)
    session.commit()
    service.reconcile_batch_decision(workflow.id)

    session.expire_all()
    assert session.get(GenerationWorkflow, workflow.id).status == "AWAITING_CONTENT_APPROVAL"
    stored_candidate = session.get(WorkflowStep, claimed_candidate.id)
    assert stored_candidate.status == "COMPLETED"
    assert stored_candidate.lease_owner is None
    assert stored_candidate.lease_expires_at is None


def test_plan_approval_rejects_gate_without_active_plan_artifact(
    session, workflow
) -> None:
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status="AWAITING_PLAN_APPROVAL")
    )
    session.commit()

    with pytest.raises(ValueError, match="active plan artifact"):
        WorkflowService(session).approve_plan(workflow.id, "author")
    assert session.scalar(
        select(func.count())
        .select_from(WorkflowStep)
        .where(WorkflowStep.kind == "WRITING")
    ) == 0


def test_plan_rejection_requires_reason_preserves_audit_and_releases_matching_owner(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    with pytest.raises(ValueError, match="rejection reason is required"):
        service.reject_plan(workflow.id, " \n ", "author")

    rejected = service.reject_plan(workflow.id, "  pacing is wrong  ", "author")

    assert rejected.status == "REJECTED"
    session.expire_all()
    assert session.get(NovelProject, workflow.project_id).active_workflow_id is None
    decision = session.scalar(
        select(PlanDecision).where(PlanDecision.workflow_id == workflow.id)
    )
    assert (decision.decision, decision.reason, decision.actor) == (
        "rejected",
        "pacing is wrong",
        "author",
    )
    audit = session.scalar(
        select(AuditEvent).where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "plan_rejected",
        )
    )
    assert audit.details["reason"] == "pacing is wrong"


def test_plan_rejection_never_releases_another_workflows_ownership(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    session.execute(
        update(NovelProject)
        .where(NovelProject.id == workflow.project_id)
        .values(active_workflow_id="another-workflow")
    )
    session.commit()

    service.reject_plan(workflow.id, "superseded run", "author")

    session.expire_all()
    assert (
        session.get(NovelProject, workflow.project_id).active_workflow_id
        == "another-workflow"
    )


@pytest.mark.parametrize(
    "paused_status", ["PAUSED_CONTEXT_OVERFLOW", "PAUSED_PROVIDER"]
)
def test_resume_whitelist_returns_paused_current_step_to_pending(
    session, workflow, paused_status
) -> None:
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status=paused_status)
    )
    session.execute(
        update(WorkflowStep)
        .where(WorkflowStep.id == step.id)
        .values(status="PAUSED", lease_owner=None, lease_expires_at=None)
    )
    session.commit()

    service = WorkflowService(session)
    project = session.get(NovelProject, workflow.project_id)
    assert project is not None
    project.title = "uncommitted caller state"
    before = (
        session.get(GenerationWorkflow, workflow.id).revision,
        session.get(WorkflowStep, step.id).revision,
    )
    assert service.can_resume(workflow.id) is True
    assert project.title == "uncommitted caller state"
    assert session.is_modified(project) is True
    session.rollback()
    session.expire_all()
    assert (
        session.get(GenerationWorkflow, workflow.id).revision,
        session.get(WorkflowStep, step.id).revision,
    ) == before

    resumed = service.resume(workflow.id)

    assert resumed.status == "PLANNING"
    session.expire_all()
    assert session.get(WorkflowStep, step.id).status == "PENDING"
    audit = session.scalar(
        select(AuditEvent).where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "workflow_resumed",
        )
    )
    assert audit.details == {
        "from_status": paused_status,
        "to_status": "PLANNING",
    }


@pytest.mark.parametrize(
    "invalid_state",
    [
        "status",
        "ownership",
        "outline",
        "current_step",
        "step_status",
        "lease",
        "active_artifact",
        "attempts",
    ],
)
def test_can_resume_and_resume_share_the_complete_rejection_predicate(
    session, workflow, clock, invalid_state
) -> None:
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    project = session.get(NovelProject, workflow.project_id)
    assert step is not None and project is not None
    workflow.status = "PAUSED_PROVIDER"
    step.status = "PAUSED"
    step.lease_owner = None
    step.lease_expires_at = None

    if invalid_state == "status":
        workflow.status = "PAUSED_REVIEW"
    elif invalid_state == "ownership":
        project.active_workflow_id = "another-workflow"
    elif invalid_state == "outline":
        project.official_outline_version_id = None
    elif invalid_state == "current_step":
        workflow.current_position = 99
    elif invalid_state == "step_status":
        step.status = "PENDING"
    elif invalid_state == "lease":
        step.lease_owner = "stale-worker"
        step.lease_expires_at = clock.now() + timedelta(minutes=5)
    elif invalid_state == "active_artifact":
        step.active_artifact_id = "already-active"
    elif invalid_state == "attempts":
        step.attempt_count = 2
    session.commit()

    service = WorkflowService(session, clock=clock)
    assert service.can_resume(workflow.id) is False
    with pytest.raises(ValueError):
        service.resume(workflow.id)


def test_can_resume_returns_false_for_missing_workflow(session) -> None:
    assert WorkflowService(session).can_resume("missing") is False


@pytest.mark.parametrize(
    "status",
    [
        "PAUSED_ATTEMPTS",
        "PAUSED_REVIEW",
        "PAUSED_STALE_VERSION",
        "COMPLETED",
        "REJECTED",
        "CANCELLED",
        "FAILED",
    ],
)
def test_resume_rejects_every_status_outside_strict_whitelist(
    session, workflow, status
) -> None:
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status=status)
    )
    session.commit()

    with pytest.raises(ValueError, match="not resumable"):
        WorkflowService(session).resume(workflow.id)


@pytest.mark.parametrize(
    ("batch_status", "expected_workflow_status"),
    [("approved", "COMPLETED"), ("rejected", "REJECTED")],
)
def test_batch_reconciliation_uses_terminal_phase_one_decision_and_is_idempotent(
    session, workflow, batch_status, expected_workflow_status
) -> None:
    batch = WritingBatch(
        id=f"batch-{batch_status}",
        project_id=workflow.project_id,
        base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=1,
        planned_chapters=workflow.requested_chapters,
        status=batch_status,
        source_workflow_id=workflow.id,
    )
    session.add(batch)
    session.flush()
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(status="CREATING_CANDIDATE_BATCH", candidate_batch_id=None)
    )
    session.commit()

    service = WorkflowService(session)
    first = service.reconcile_batch_decision(workflow.id)
    second = service.reconcile_batch_decision(workflow.id)

    assert first.status == expected_workflow_status
    assert second.status == expected_workflow_status
    assert first.candidate_batch_id == batch.id
    session.expire_all()
    assert session.get(NovelProject, workflow.project_id).active_workflow_id is None
    action = (
        "workflow_completed"
        if expected_workflow_status == "COMPLETED"
        else "workflow_rejected"
    )
    assert session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(AuditEvent.entity_id == workflow.id, AuditEvent.action == action)
    ) == 1


def test_batch_reconciliation_links_nonterminal_batch_without_releasing_owner(
    session, workflow
) -> None:
    candidate = WorkflowStep(
        id="ready-candidate-step",
        workflow_id=workflow.id,
        kind="CREATING_CANDIDATE_BATCH",
        ordinal=None,
        position=1,
        status="PENDING",
        attempt_count=0,
        revision=1,
    )
    batch = WritingBatch(
        id="ready-candidate-batch",
        project_id=workflow.project_id,
        base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=1,
        planned_chapters=workflow.requested_chapters,
        status="ready_for_review",
        source_workflow_id=workflow.id,
    )
    session.add_all([candidate, batch])
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(
            status="CREATING_CANDIDATE_BATCH",
            current_position=candidate.position,
            candidate_batch_id=None,
        )
    )
    session.commit()

    service = WorkflowService(session)
    first = service.reconcile_batch_decision(workflow.id)
    second = service.reconcile_batch_decision(workflow.id)

    assert first.status == second.status == "AWAITING_CONTENT_APPROVAL"
    assert first.candidate_batch_id == second.candidate_batch_id == batch.id
    session.expire_all()
    assert session.get(WorkflowStep, candidate.id).status == "COMPLETED"
    assert (
        session.get(NovelProject, workflow.project_id).active_workflow_id
        == workflow.id
    )
    assert session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "candidate_batch_linked",
        )
    ) == 1


def test_terminal_reconciliation_never_clears_another_workflows_owner(
    session, workflow
) -> None:
    batch = WritingBatch(
        id="approved-batch-other-owner",
        project_id=workflow.project_id,
        base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=1,
        planned_chapters=workflow.requested_chapters,
        status="approved",
        source_workflow_id=workflow.id,
    )
    session.add(batch)
    session.flush()
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(
            status="AWAITING_CONTENT_APPROVAL", candidate_batch_id=batch.id
        )
    )
    session.execute(
        update(NovelProject)
        .where(NovelProject.id == workflow.project_id)
        .values(active_workflow_id="another-workflow")
    )
    session.commit()

    WorkflowService(session).reconcile_batch_decision(workflow.id)

    session.expire_all()
    assert (
        session.get(NovelProject, workflow.project_id).active_workflow_id
        == "another-workflow"
    )


def test_summary_completion_produces_text_hash_accepted_by_context_index(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    _complete_plan(service, workflow)
    service.approve_plan(workflow.id, "author")
    writer = service.claim_step(
        workflow.id, {"GENERATING_CHAPTERS"}, "writer-a"
    )
    assert writer is not None and writer.kind == "WRITING"
    writer_attempt = service.record_attempt_start(writer.id, "b" * 64, claim_revision=writer.revision)
    service.complete_attempt(
        writer_attempt.id,
        _response(),
        {
            "kind": "chapter_draft",
            "payload": {"title": "Chapter 1", "body": "甲" * 4_500},
        },
    )
    summarizer = service.claim_step(
        workflow.id, {"GENERATING_CHAPTERS"}, "summarizer-a"
    )
    assert summarizer is not None and summarizer.kind == "SUMMARIZING"
    summary_attempt = service.record_attempt_start(summarizer.id, "c" * 64, claim_revision=summarizer.revision)
    summary_text = "Chapter one closes with the bridge still contested."
    summary = service.complete_attempt(
        summary_attempt.id,
        _response(),
        {
            "kind": "chapter_summary_delta",
            "payload": {
                "summary": summary_text,
                "state_delta": {"bridge": "contested"},
            },
        },
    )
    session.execute(
        text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS context_source_fts "
            "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
        )
    )
    session.commit()

    indexed = ContextIndexService(session).index_workflow_artifact(summary.id)

    assert summary.content_hash == sha256(summary_text.encode("utf-8")).hexdigest()
    assert indexed.text == summary_text
    assert indexed.content_hash == summary.content_hash


def test_draft_batch_reconciliation_only_binds_and_leaves_crashed_step_recoverable(
    session, workflow, clock
) -> None:
    candidate = WorkflowStep(
        id="draft-candidate-step",
        workflow_id=workflow.id,
        kind="CREATING_CANDIDATE_BATCH",
        ordinal=None,
        position=1,
        status="RUNNING",
        attempt_count=0,
        lease_owner="batch-worker",
        lease_expires_at=clock.now() + timedelta(seconds=300),
        revision=1,
    )
    batch = WritingBatch(
        id="draft-candidate-batch",
        project_id=workflow.project_id,
        base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=1,
        planned_chapters=workflow.requested_chapters,
        status="draft",
        source_workflow_id=workflow.id,
    )
    session.add_all([candidate, batch])
    session.execute(
        update(GenerationWorkflow)
        .where(GenerationWorkflow.id == workflow.id)
        .values(
            status="CREATING_CANDIDATE_BATCH",
            current_position=candidate.position,
            candidate_batch_id=None,
        )
    )
    session.commit()

    service = WorkflowService(session, clock=clock)
    first = service.reconcile_batch_decision(workflow.id)
    second = service.reconcile_batch_decision(workflow.id)

    assert first.status == second.status == "CREATING_CANDIDATE_BATCH"
    assert first.candidate_batch_id == second.candidate_batch_id == batch.id
    session.expire_all()
    stored_step = session.get(WorkflowStep, candidate.id)
    assert stored_step.status == "RUNNING"
    assert stored_step.lease_owner == "batch-worker"
    assert stored_step.lease_expires_at == clock.now() + timedelta(seconds=300)
    assert session.get(NovelProject, workflow.project_id).active_workflow_id == workflow.id
    assert session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "candidate_batch_linked",
        )
    ) == 1

    clock.advance(seconds=301)
    assert service.recover_expired_claims(workflow.id, clock.now()) == 1
    session.expire_all()
    recovered = session.get(WorkflowStep, candidate.id)
    assert recovered.status == "PENDING"
    assert recovered.lease_owner is None
    assert recovered.lease_expires_at is None


def test_required_context_overflow_pauses_claim_before_attempt_or_provider_call(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    provider = FakeProvider([])
    step = service.claim_step(
        workflow.id, {"PLANNING"}, "context-worker", lease_seconds=30
    )
    assert step is not None
    overflow = RequiredContextOverflow(
        "constitution:sk-secret-material", required_tokens=9_001, capacity=4_000
    )

    with pytest.raises(ValueError, match="context overflow pause conflict"):
        service.pause_context_overflow(step.id, "different-worker", overflow, claim_revision=step.revision)
    paused = service.pause_context_overflow(step.id, "context-worker", overflow, claim_revision=step.revision)

    assert paused.status == "PAUSED_CONTEXT_OVERFLOW"
    assert paused.last_error_code == "required_context_overflow"
    assert paused.last_error_detail == "required context exceeds the available input budget"
    assert "secret" not in paused.last_error_detail
    session.expire_all()
    stored_step = session.get(WorkflowStep, step.id)
    assert stored_step.status == "PAUSED"
    assert stored_step.lease_owner is None
    assert stored_step.lease_expires_at is None
    assert session.scalar(
        select(func.count()).select_from(ModelAttempt).where(ModelAttempt.step_id == step.id)
    ) == 0
    assert provider.requests == []
    audit = session.scalar(
        select(AuditEvent).where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "workflow_paused",
        )
    )
    assert audit is not None
    assert audit.details == {"reason": "required_context_overflow"}


def test_required_context_overflow_rejects_an_expired_lease(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(
        workflow.id, {"PLANNING"}, "context-worker", lease_seconds=30
    )
    assert step is not None
    clock.advance(seconds=31)

    with pytest.raises(ValueError, match="context overflow pause conflict"):
        service.pause_context_overflow(
            step.id,
            "context-worker",
            RequiredContextOverflow("constitution", required_tokens=2, capacity=1),
            claim_revision=step.revision,
        )

    session.expire_all()
    assert session.get(GenerationWorkflow, workflow.id).status == "PLANNING"
    assert session.get(WorkflowStep, step.id).status == "RUNNING"
    assert session.scalar(
        select(func.count()).select_from(ModelAttempt).where(ModelAttempt.step_id == step.id)
    ) == 0


def test_required_context_overflow_rejects_step_with_running_attempt_atomically(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(
        workflow.id, {"PLANNING"}, "context-worker", lease_seconds=30
    )
    assert step is not None
    claimed_step_revision = step.revision
    attempt = service.record_attempt_start(step.id, "1" * 64, claim_revision=step.revision)
    session.expire_all()
    before_workflow = session.get(GenerationWorkflow, workflow.id)
    before_step = session.get(WorkflowStep, step.id)
    workflow_revision = before_workflow.revision
    step_revision = before_step.revision
    lease_expires_at = before_step.lease_expires_at
    assert step_revision == claimed_step_revision + 1

    with pytest.raises(ValueError, match="context overflow pause conflict"):
        service.pause_context_overflow(
            step.id,
            "context-worker",
            RequiredContextOverflow("constitution", required_tokens=2, capacity=1),
            claim_revision=step.revision,
        )

    session.expire_all()
    stored_workflow = session.get(GenerationWorkflow, workflow.id)
    stored_step = session.get(WorkflowStep, step.id)
    stored_attempt = session.get(ModelAttempt, attempt.id)
    assert stored_workflow.status == "PLANNING"
    assert stored_workflow.revision == workflow_revision
    assert stored_workflow.last_error_code is None
    assert stored_workflow.last_error_detail is None
    assert stored_step.status == "RUNNING"
    assert stored_step.revision == step_revision
    assert stored_step.attempt_count == 1
    assert stored_step.active_artifact_id is None
    assert stored_step.lease_owner == "context-worker"
    assert stored_step.lease_expires_at == lease_expires_at
    assert stored_attempt.status == "RUNNING"
    assert stored_attempt.attempt_number == 1
    assert stored_attempt.request_digest == "1" * 64
    assert stored_attempt.provider_response_id is None
    assert stored_attempt.input_tokens is None
    assert stored_attempt.output_tokens is None
    assert stored_attempt.latency_ms is None
    assert stored_attempt.error_code is None
    assert stored_attempt.error_detail is None
    assert session.scalar(
        select(func.count())
        .select_from(AuditEvent)
        .where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "workflow_paused",
        )
    ) == 0


def test_expired_attempt_failure_is_fenced_and_lease_recovery_wins(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(
        workflow.id, {"PLANNING"}, "worker-a", lease_seconds=30
    )
    assert step is not None
    attempt = service.record_attempt_start(step.id, "d" * 64, claim_revision=step.revision)
    clock.advance(seconds=31)

    with pytest.raises(ValueError, match="attempt failure conflict"):
        service.fail_attempt(attempt.id, ProviderTimeout("late timeout with secret"))

    assert service.recover_expired_claims(workflow.id, clock.now()) == 1
    session.expire_all()
    stored_attempt = session.get(ModelAttempt, attempt.id)
    stored_step = session.get(WorkflowStep, step.id)
    assert stored_attempt.status == "FAILED"
    assert stored_attempt.error_code == "lease_expired"
    assert stored_step.status == "PENDING"
    assert stored_step.lease_owner is None


@pytest.mark.parametrize(
    "request_digest",
    [
        "A" * 64,
        "g" * 64,
        "a" * 63,
        "a" * 65,
        "sk-live-secret-material",
        f" {'a' * 64} ",
    ],
)
def test_attempt_start_rejects_noncanonical_sha256_without_mutation(
    session, workflow, clock, request_digest
) -> None:
    service = WorkflowService(session, clock=clock)
    step = service.claim_step(workflow.id, {"PLANNING"}, "worker-a")
    assert step is not None
    initial_revision = session.get(WorkflowStep, step.id).revision

    with pytest.raises(ValueError, match="canonical SHA-256"):
        service.record_attempt_start(step.id, request_digest, claim_revision=step.revision)

    session.expire_all()
    stored_step = session.get(WorkflowStep, step.id)
    assert stored_step.attempt_count == 0
    assert stored_step.revision == initial_revision
    assert session.scalar(
        select(func.count()).select_from(ModelAttempt).where(ModelAttempt.step_id == step.id)
    ) == 0


def test_nonfinal_review_requires_first_attempt_with_real_evidence_request(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    reviewer = _prepare_reviewer_step(session, service, workflow)
    claimed = service.claim_step(workflow.id, {"REVIEWING_BATCH"}, "reviewer-a")
    assert claimed is not None and claimed.id == reviewer.id
    attempt = service.record_attempt_start(claimed.id, "e" * 64, claim_revision=claimed.revision)

    with pytest.raises(ValueError, match="reviewer evidence request is invalid"):
        service.complete_attempt(
            attempt.id,
            _response(),
            {
                "kind": "batch_review",
                "payload": {"passed": True, "issues": [], "evidence_queries": []},
            },
            finalize_step=False,
        )

    session.expire_all()
    assert session.get(ModelAttempt, attempt.id).status == "RUNNING"
    assert session.get(WorkflowStep, reviewer.id).status == "RUNNING"
    assert session.scalar(
        select(func.count())
        .select_from(WorkflowArtifact)
        .where(WorkflowArtifact.kind == "batch_review")
    ) == 0


def test_second_review_evidence_request_pauses_instead_of_becoming_pending(
    session, workflow, clock
) -> None:
    service = WorkflowService(session, clock=clock)
    reviewer = _prepare_reviewer_step(session, service, workflow)
    first_claim = service.claim_step(
        workflow.id, {"REVIEWING_BATCH"}, "reviewer-a"
    )
    assert first_claim is not None and first_claim.id == reviewer.id
    first_attempt = service.record_attempt_start(first_claim.id, "e" * 64, claim_revision=first_claim.revision)
    service.complete_attempt(
        first_attempt.id,
        _response(),
        {
            "kind": "batch_review",
            "payload": {
                "passed": False,
                "issues": ["timeline unclear"],
                "evidence_queries": ["chapter 1 bridge"],
            },
        },
        finalize_step=False,
    )
    second_claim = service.claim_step(
        workflow.id, {"REVIEWING_BATCH"}, "reviewer-b"
    )
    assert second_claim is not None
    second_attempt = service.record_attempt_start(second_claim.id, "f" * 64, claim_revision=second_claim.revision)

    artifact = service.complete_attempt(
        second_attempt.id,
        _response(),
        {
            "kind": "batch_review",
            "payload": {
                "passed": False,
                "issues": ["timeline still unclear"],
                "evidence_queries": ["chapter 2 bridge"],
            },
        },
        finalize_step=False,
    )

    session.expire_all()
    stored_step = session.get(WorkflowStep, reviewer.id)
    assert session.get(GenerationWorkflow, workflow.id).status == "PAUSED_REVIEW"
    assert stored_step.status == "PAUSED"
    assert stored_step.active_artifact_id == artifact.id

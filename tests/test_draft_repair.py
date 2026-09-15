from __future__ import annotations

import pytest
from sqlalchemy import select

from ainovel.models import (
    GenerationWorkflow,
    ModelAttempt,
    PromptVersion,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.fake import FakeProvider
from ainovel.services.prompts import PromptService
from ainovel.services.draft_repair import DraftRepairService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService

from test_orchestrator import (
    FrozenClock,
    clock,
    make_orchestrator,
    ready_project,
    response,
    session_factory,
)


def v2_plan_payload() -> dict[str, object]:
    return {
        "chapters": [
            {
                "ordinal": 1,
                "title": "城门夜变",
                "goal": "主角设法进入封锁的城门",
                "ending_hook": "暗处传来警示铃",
                "scenes": [
                    {
                        "ordinal": 1,
                        "description": "主角与守卫交锋后进入城门",
                        "target_characters": 2600,
                    },
                    {
                        "ordinal": 2,
                        "description": "主角发现暗处的警示",
                        "target_characters": 2600,
                    },
                ],
            }
        ]
    }


def body_with_evidence(visible_count: int) -> str:
    evidence = "守卫终于让开了城门。暗处的铃声忽然响起。"
    return evidence + "城" * (visible_count - len(evidence))


def start_approved_v2(
    session_factory, session, ready_project, clock, provider
) -> tuple[GenerationWorkflow, object]:
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id,
        "fake",
        "scripted",
        1,
        DEFAULT_BUDGETS,
        generation_version=2,
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    return workflow, orchestrator


def test_v2_short_drafts_are_durably_repaired_twice_before_evidenced_promotion(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    first_body = body_with_evidence(2799)
    second_body = body_with_evidence(3386)
    final_body = body_with_evidence(5200)
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": first_body}, 2, output_tokens=701),
            response({"title": "城门夜变", "body": second_body}, 3, output_tokens=823),
            response({"title": "城门夜变", "body": final_body}, 4, output_tokens=1199),
            response(
                {
                    "goal": {"passed": True, "excerpt": "守卫终于让开了城门"},
                    "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
                },
                5,
            ),
        ]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id,
        "fake",
        "scripted",
        1,
        DEFAULT_BUDGETS,
        generation_version=2,
    )
    from ainovel.models import ChapterDraftRepair

    orchestrator = make_orchestrator(session_factory, provider, clock)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    first = orchestrator.advance(workflow.id)

    session.expire_all()
    work_draft = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    )
    assert first.completed_step is None
    assert work_draft.visible_count == 2799
    assert work_draft.repair_count == 0
    assert work_draft.latest_payload == {"title": "城门夜变", "body": first_body}
    view = DraftRepairService(session).list_for_workflow(workflow.id)[0]
    assert view.writing_step_id == work_draft.writing_step_id
    assert view.ordinal == 1
    assert view.title == "城门夜变"
    assert view.body == first_body
    assert view.visible_count == 2799
    assert view.repair_count == 0
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ).all()

    second = orchestrator.advance(workflow.id)

    next_payload = provider.requests[2].input_payload
    session.expire_all()
    work_draft = session.get(ChapterDraftRepair, work_draft.id)
    assert second.completed_step is None
    assert next_payload["repair"]["visible_count"] == 2799
    assert next_payload["repair"]["minimum_gap"] == 1701
    assert next_payload["repair"]["draft"]["body"] == first_body
    assert work_draft.visible_count == 3386
    assert work_draft.repair_count == 1
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ).all()

    accepted = orchestrator.advance(workflow.id)

    latest_payload = provider.requests[3].input_payload
    session.expire_all()
    work_draft = session.get(ChapterDraftRepair, work_draft.id)
    assert accepted.completed_step == "WRITING"
    assert latest_payload["repair"]["visible_count"] == 3386
    assert latest_payload["repair"]["minimum_gap"] == 1114
    assert latest_payload["repair"]["draft"]["body"] == second_body
    assert work_draft.visible_count == 5200
    assert work_draft.repair_count == 2
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ).all()

    covered = orchestrator.advance(workflow.id)

    session.expire_all()
    chapter = session.scalar(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    )
    stored_workflow = session.get(type(workflow), workflow.id)
    assert covered.completed_step == "VALIDATING_CHAPTER"
    assert chapter is not None
    assert chapter.payload == {"title": "城门夜变", "body": final_body}
    assert stored_workflow.actual_output_tokens == 50 + 701 + 823 + 1199 + 50
    assert [request.metadata["agent_role"] for request in provider.requests] == [
        "batch_planner",
        "chapter_writer",
        "chapter_writer",
        "chapter_writer",
        "chapter_coverage_reviewer",
    ]


def test_v2_third_short_draft_pauses_and_recreated_orchestrator_cannot_reset_repairs(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": body_with_evidence(2799)}, 2),
            response({"title": "城门夜变", "body": body_with_evidence(3386)}, 3),
            response({"title": "城门夜变", "body": body_with_evidence(4100)}, 4),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step is None
    assert orchestrator.advance(workflow.id).completed_step is None
    assert orchestrator.advance(workflow.id).completed_step is None

    recreated = make_orchestrator(session_factory, provider, clock)
    result = recreated.advance(workflow.id)

    session.expire_all()
    from ainovel.models import ChapterDraftRepair

    state = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    )
    assert result.status == "PAUSED_REVIEW"
    assert state.visible_count == 4100
    assert state.repair_count == 2
    assert len(provider.requests) == 4
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ).all()


def test_v2_total_call_budget_pauses_before_provider_call(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    provider = FakeProvider([response(v2_plan_payload(), 1)])
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    session.execute(
        GenerationWorkflow.__table__.update()
        .where(GenerationWorkflow.id == workflow.id)
        .values(model_call_limit=1)
    )
    session.commit()

    result = orchestrator.advance(workflow.id)

    session.expire_all()
    stored = session.get(GenerationWorkflow, workflow.id)
    assert result.status == "PAUSED_ATTEMPTS"
    assert stored.model_calls_used == 1
    assert stored.last_error_code == "workflow_budget_exhausted"
    assert len(provider.requests) == 1


def test_v2_budget_rejection_does_not_spend_an_undispatched_repair(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response(
                {"title": "城门夜变", "body": body_with_evidence(2799)}, 2
            ),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step is None
    session.execute(
        GenerationWorkflow.__table__.update()
        .where(GenerationWorkflow.id == workflow.id)
        .values(model_call_limit=2)
    )
    session.commit()

    result = orchestrator.advance(workflow.id)

    from ainovel.models import ChapterDraftRepair

    session.expire_all()
    state = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    )
    assert result.status == "PAUSED_ATTEMPTS"
    assert state.repair_count == 0
    assert state.repair_pending is False
    assert len(provider.requests) == 2


def test_malformed_v2_draft_is_protocol_failure_not_repairable_state(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    leaked_body = body_with_evidence(2799)
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response(
                {"title": "城门夜变", "body": leaked_body, "unexpected": "field"},
                2,
            ),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )

    result = orchestrator.advance(workflow.id)

    from ainovel.models import ChapterDraftRepair

    attempt = session.scalar(
        select(ModelAttempt)
        .join(WorkflowStep, WorkflowStep.id == ModelAttempt.step_id)
        .where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
        )
    )
    assert result.status == "GENERATING_CHAPTERS"
    assert attempt.status == "FAILED"
    assert attempt.error_code == "provider_protocol"
    assert leaked_body not in (attempt.error_detail or "")
    assert session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    ) is None


@pytest.mark.parametrize(
    "coverage",
    [
        {
            "goal": {"passed": True, "excerpt": ""},
            "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
        },
        {
            "goal": {"passed": True, "excerpt": "正文中不存在的伪造证据"},
            "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
        },
        {
            "goal": {"passed": False, "excerpt": ""},
            "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
        },
    ],
    ids=["missing", "forged", "negative"],
)
def test_v2_invalid_coverage_pauses_without_promoting_body(
    session_factory,
    session,
    ready_project,
    clock: FrozenClock,
    coverage: dict[str, object],
) -> None:
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response(
                {"title": "城门夜变", "body": body_with_evidence(5200)}, 2
            ),
            response(coverage, 3),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step == "WRITING"

    result = orchestrator.advance(workflow.id)

    assert result.status == "PAUSED_REVIEW"
    assert len(provider.requests) == 3
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ).all()


def test_v2_snapshot_selects_frozen_builtin_without_changing_active_v1_prompt(
    session, ready_project, clock: FrozenClock
) -> None:
    prompts = PromptService(session)
    prompts.ensure_builtins()
    custom = prompts.create_version(
        "chapter_writer", "作者自定义旧版主笔提示词", "author"
    )
    prompts.activate(custom.id)

    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id,
        "fake",
        "scripted",
        1,
        DEFAULT_BUDGETS,
        generation_version=2,
    )

    active = session.scalar(
        select(PromptVersion).where(
            PromptVersion.role == "chapter_writer",
            PromptVersion.active.is_(True),
        )
    )
    snapshots = {
        row.role: row for row in PromptService(session).list_snapshots(workflow.id)
    }
    assert active.id == custom.id
    assert snapshots["chapter_writer"].prompt_body != custom.body
    assert "repair" in snapshots["chapter_writer"].prompt_body
    assert set(snapshots) == {
        "batch_planner",
        "chapter_writer",
        "chapter_coverage_reviewer",
        "chapter_summarizer",
        "batch_reviewer",
    }


def test_demo_fake_provider_supports_v2_plan_writer_and_coverage(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    provider = DemoFakeProvider()
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id,
        "fake",
        "demo",
        1,
        DEFAULT_BUDGETS,
        generation_version=2,
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    assert orchestrator.advance(workflow.id).completed_step == "WRITING"
    result = orchestrator.advance(workflow.id)

    assert result.completed_step == "VALIDATING_CHAPTER"
    assert session.scalar(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
        )
    ) is not None


def test_expired_writer_lease_cannot_persist_work_draft(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    class LeaseExpiringProvider(FakeProvider):
        def generate(self, request):
            if request.metadata["agent_role"] == "chapter_writer":
                clock.advance(seconds=301)
            return super().generate(request)

    provider = LeaseExpiringProvider(
        [
            response(v2_plan_payload(), 1),
            response(
                {"title": "城门夜变", "body": body_with_evidence(2799)}, 2
            ),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )

    with pytest.raises(ValueError, match="attempt completion conflict"):
        orchestrator.advance(workflow.id)

    from ainovel.models import ChapterDraftRepair

    assert session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    ) is None
    assert not session.scalars(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind.in_({"chapter_work_draft", "chapter_draft"}),
        )
    ).all()


def test_v2_initial_writer_expired_attempts_reach_protocol_limit(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    class LeaseExpiringProvider(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.metadata["agent_role"] == "chapter_writer":
                clock.advance(seconds=301)
            return result

    provider = LeaseExpiringProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": body_with_evidence(5200)}, 2),
            response({"title": "城门夜变", "body": body_with_evidence(5200)}, 3),
            response({"title": "城门夜变", "body": body_with_evidence(5200)}, 4),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )

    with pytest.raises(ValueError, match="attempt completion conflict"):
        orchestrator.advance(workflow.id)
    with pytest.raises(ValueError, match="attempt completion conflict"):
        orchestrator.advance(workflow.id)
    paused = orchestrator.advance(workflow.id)

    session.expire_all()
    writing = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
        )
    )
    assert paused.status == "PAUSED_ATTEMPTS"
    assert writing.attempt_count == 2
    assert writing.protocol_failure_count == 2
    assert len(
        [
            request
            for request in provider.requests
            if request.metadata["agent_role"] == "chapter_writer"
        ]
    ) == 2


def test_v2_pending_repair_expired_attempts_preserve_semantic_reservation(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    class RepairLeaseExpiringProvider(FakeProvider):
        writer_calls = 0

        def generate(self, request):
            result = super().generate(request)
            if request.metadata["agent_role"] == "chapter_writer":
                self.writer_calls += 1
                if self.writer_calls > 1:
                    clock.advance(seconds=301)
            return result

    provider = RepairLeaseExpiringProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": body_with_evidence(2799)}, 2),
            response({"title": "城门夜变", "body": body_with_evidence(3386)}, 3),
            response({"title": "城门夜变", "body": body_with_evidence(3386)}, 4),
            response({"title": "城门夜变", "body": body_with_evidence(3386)}, 5),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step is None

    with pytest.raises(ValueError, match="attempt completion conflict"):
        orchestrator.advance(workflow.id)
    with pytest.raises(ValueError, match="attempt completion conflict"):
        orchestrator.advance(workflow.id)
    paused = orchestrator.advance(workflow.id)

    from ainovel.models import ChapterDraftRepair

    session.expire_all()
    writing = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
        )
    )
    repair = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    )
    assert paused.status == "PAUSED_ATTEMPTS"
    assert writing.attempt_count == 3
    assert writing.protocol_failure_count == 2
    assert repair.repair_count == 1
    assert repair.repair_pending is True
    assert provider.writer_calls == 3


def test_v2_expired_claim_without_model_attempt_does_not_consume_protocol_retry(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": body_with_evidence(5200)}, 2),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    service = WorkflowService(session, clock=clock)
    claimed = service.claim_step(
        workflow.id, {"GENERATING_CHAPTERS"}, "stalled-worker", lease_seconds=300
    )
    assert claimed is not None
    clock.advance(seconds=301)

    assert service.recover_expired_claims(workflow.id, clock.now()) == 1
    session.expire_all()
    writing = session.get(WorkflowStep, claimed.id)
    assert writing.protocol_failure_count == 0
    assert orchestrator.advance(workflow.id).completed_step == "WRITING"


@pytest.mark.parametrize(
    "body",
    [
        "城" * 6001,
        ("城门下的守卫反复盘问来客。" * 100)
        + "\n\n"
        + ("城门下的守卫反复盘问来客。" * 100),
    ],
    ids=["oversized", "repeated"],
)
def test_v2_business_rejected_writer_response_records_usage(
    session_factory, session, ready_project, clock: FrozenClock, body: str
) -> None:
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1, input_tokens=100, output_tokens=50),
            response(
                {"title": "城门夜变", "body": body},
                2,
                input_tokens=211,
                output_tokens=322,
            ),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )

    result = orchestrator.advance(workflow.id)

    session.expire_all()
    failed_attempt = session.scalar(
        select(ModelAttempt)
        .join(WorkflowStep, WorkflowStep.id == ModelAttempt.step_id)
        .where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
            ModelAttempt.status == "FAILED",
        )
    )
    workflow = session.get(GenerationWorkflow, workflow.id)
    assert result.status == "GENERATING_CHAPTERS"
    assert failed_attempt.input_tokens == 211
    assert failed_attempt.output_tokens == 322
    assert workflow.actual_input_tokens == 311
    assert workflow.actual_output_tokens == 372


def test_v2_protocol_retry_does_not_consume_an_extra_semantic_repair(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    first_body = body_with_evidence(2799)
    second_body = body_with_evidence(3386)
    final_body = body_with_evidence(5200)
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": first_body}, 2),
            response(
                {"title": "城门夜变", "body": second_body, "extra": "bad"},
                3,
                input_tokens=211,
                output_tokens=322,
            ),
            response({"title": "城门夜变", "body": second_body}, 4),
            response({"title": "城门夜变", "body": final_body}, 5),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step is None

    failed = orchestrator.advance(workflow.id)

    from ainovel.models import ChapterDraftRepair

    session.expire_all()
    state = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.workflow_id == workflow.id
        )
    )
    failed_attempt = session.scalar(
        select(ModelAttempt)
        .join(WorkflowStep, WorkflowStep.id == ModelAttempt.step_id)
        .where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
            ModelAttempt.status == "FAILED",
        )
    )
    assert failed.status == "GENERATING_CHAPTERS"
    assert state.repair_count == 1
    assert state.repair_pending is True
    assert failed_attempt.input_tokens == 211
    assert failed_attempt.output_tokens == 322

    assert orchestrator.advance(workflow.id).completed_step is None
    session.expire_all()
    state = session.get(ChapterDraftRepair, state.id)
    assert state.repair_count == 1
    assert state.repair_pending is False
    assert orchestrator.advance(workflow.id).completed_step == "WRITING"
    session.expire_all()
    state = session.get(ChapterDraftRepair, state.id)
    assert state.repair_count == 2


def test_v2_batch_review_receives_evidenced_coverage_not_literal_label_checks(
    session_factory, session, ready_project, clock: FrozenClock
) -> None:
    final_body = body_with_evidence(5200)
    coverage = {
        "goal": {"passed": True, "excerpt": "守卫终于让开了城门"},
        "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
    }
    provider = FakeProvider(
        [
            response(v2_plan_payload(), 1),
            response({"title": "城门夜变", "body": final_body}, 2),
            response(coverage, 3),
            response({"summary": "主角进入城门并听见铃声", "state_delta": {}}, 4),
            response({"passed": True, "issues": [], "evidence_queries": []}, 5),
        ]
    )
    workflow, orchestrator = start_approved_v2(
        session_factory, session, ready_project, clock, provider
    )
    assert orchestrator.advance(workflow.id).completed_step == "WRITING"
    assert orchestrator.advance(workflow.id).completed_step == "VALIDATING_CHAPTER"
    assert orchestrator.advance(workflow.id).completed_step == "SUMMARIZING"

    orchestrator.advance(workflow.id)

    validation = provider.requests[4].input_payload["chapter_reports"][0][
        "validation_results"
    ]
    assert validation["coverage"] == coverage
    assert validation["coverage_evidence_valid"] is True
    assert "approved_goal_present" not in validation
    assert "approved_key_event_present" not in validation

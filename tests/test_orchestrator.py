from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlalchemy import func, select, text, update

from ainovel.agents.runner import AgentRunner
from ainovel.context import ConservativeEstimator
from ainovel.models import (
    ContextPacket,
    ContextSource,
    GenerationWorkflow,
    ModelAttempt,
    NovelProject,
    WorkflowArtifact,
    WorkflowStep,
    WritingBatch,
)
from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderTimeout,
)
from ainovel.providers.fake import FakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.batches import BatchService
from ainovel.services.context import ContextIndexService, ContextService
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService
from ainovel.services.prompts import PromptService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowBudgets, WorkflowService
from ainovel.workflows.orchestrator import (
    EXECUTABLE_WORKFLOW_STATUSES,
    WorkflowOrchestrator,
    digest_request,
)


class FrozenClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def response(
    structured: dict[str, object],
    number: int,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: int = 5,
) -> ModelResponse:
    return ModelResponse(
        structured=structured,
        text=None,
        provider_response_id=f"fake-{number}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def plan_payload(count: int) -> dict[str, object]:
    return {
        "chapters": [
            {
                "ordinal": ordinal,
                "title": f"第{ordinal}章",
                "goal": f"推进第{ordinal}章目标",
                "ending_hook": f"第{ordinal}章悬念",
            }
            for ordinal in range(1, count + 1)
        ]
    }


def chapter_body(ordinal: int, visible_count: int = 4500) -> str:
    required = f"推进第{ordinal}章目标。第{ordinal}章悬念。"
    return required + chr(0x4E00 + ordinal) * (visible_count - len(required))


def success_script(count: int = 5) -> list[ModelResponse]:
    scripted = [
        response(
            plan_payload(count),
            1,
            input_tokens=321,
            output_tokens=123,
            latency_ms=47,
        )
    ]
    call = 2
    for ordinal in range(1, count + 1):
        scripted.append(
            response(
                {
                    "title": f"第{ordinal}章",
                    "body": chapter_body(ordinal, 4499 + ordinal),
                },
                call,
            )
        )
        call += 1
        scripted.append(
            response(
                {
                    "summary": f"候选摘要{ordinal}",
                    "state_delta": {"last_completed_ordinal": ordinal},
                },
                call,
            )
        )
        call += 1
    scripted.append(
        response({"passed": True, "issues": [], "evidence_queries": []}, call)
    )
    return scripted


@pytest.fixture
def session_factory(client):
    return client.app.state.session_factory


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def ready_project(session, project, official_outline):
    ProjectService(session).add_constitution(
        project.id,
        {"genre": "historical fantasy", "voice": "close third"},
        author_approved=True,
    )
    session.execute(
        text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS context_source_fts "
            "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
        )
    )
    session.commit()
    ContextIndexService(session).rebuild_official(project.id)
    session.expire_all()
    return session.get(NovelProject, project.id)


@pytest.fixture
def scripted_five_chapter_provider() -> FakeProvider:
    return FakeProvider(success_script())


def make_orchestrator(session_factory, provider, clock) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(
        session_factory,
        ProviderRegistry({"fake": lambda: provider}),
        AgentRunner(),
        ContextService,
        PromptService,
        clock=clock,
        worker_id="test-orchestrator",
    )


@pytest.fixture
def orchestrator(session_factory, scripted_five_chapter_provider, clock):
    return make_orchestrator(session_factory, scripted_five_chapter_provider, clock)


@pytest.fixture
def workflow(session, ready_project, clock) -> GenerationWorkflow:
    return WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS
    )


@pytest.fixture
def approved_plan_workflow(orchestrator, workflow, session, clock):
    planned = orchestrator.advance(workflow.id)
    assert planned.status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    session.expire_all()
    return session.get(GenerationWorkflow, workflow.id)


def make_request_for_digest() -> ModelRequest:
    return ModelRequest(
        model="m",
        system_prompt="系统",
        input_payload={"b": 2, "a": "甲"},
        output_schema={"type": "object"},
        max_input_tokens=10,
        max_output_tokens=2,
        timeout_seconds=1.0,
        metadata={"agent_role": "batch_planner", "schema_name": "batch_plan"},
    )


def test_contract_constants_and_digest_are_stable() -> None:
    assert EXECUTABLE_WORKFLOW_STATUSES == frozenset(
        {
            "PLANNING",
            "GENERATING_CHAPTERS",
            "REVIEWING_BATCH",
            "CREATING_CANDIDATE_BATCH",
        }
    )
    request = make_request_for_digest()
    assert digest_request(request) == digest_request(request)
    assert len(digest_request(request)) == 64


def test_advance_completes_at_most_one_persisted_step(
    orchestrator, scripted_five_chapter_provider, workflow, session
) -> None:
    result = orchestrator.advance(workflow.id)

    steps = session.scalars(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    ).all()
    assert result.completed_step == "PLANNING"
    assert result.status == "AWAITING_PLAN_APPROVAL"
    assert len(scripted_five_chapter_provider.requests) == 1
    assert [(step.kind, step.status) for step in steps] == [
        ("PLANNING", "COMPLETED")
    ]


def test_plan_gate_blocks_every_chapter_call(
    orchestrator, scripted_five_chapter_provider, workflow
) -> None:
    result = orchestrator.run_until_blocked(workflow.id)

    assert result.status == "AWAITING_PLAN_APPROVAL"
    assert result.waiting_for == "plan_approval"
    assert [
        request.metadata["agent_role"]
        for request in scripted_five_chapter_provider.requests
    ] == ["batch_planner"]


def test_planner_request_contains_only_snapshotted_authoritative_inputs(
    orchestrator, scripted_five_chapter_provider, workflow
) -> None:
    orchestrator.advance(workflow.id)

    payload = scripted_five_chapter_provider.requests[0].input_payload
    assert payload["requested_chapters"] == 5
    assert payload["official_outline_id"] == workflow.base_outline_version_id
    assert payload["official_outline_tree"][0]["title"] == "全书总纲"
    assert payload["project_constitution"] == {
        "genre": "historical fantasy",
        "voice": "close third",
    }
    assert payload["official_chapter_statistics"] == {
        "chapter_count": 0,
        "visible_character_count": 0,
    }
    assert payload["budgets"]["writer_input"] == 32_000


def test_real_provider_metadata_is_persisted_without_fabrication(
    orchestrator, workflow, session
) -> None:
    orchestrator.advance(workflow.id)

    attempt = session.scalar(select(ModelAttempt))
    assert attempt is not None
    assert attempt.input_tokens == 321
    assert attempt.output_tokens == 123
    assert attempt.latency_ms == 47
    assert attempt.provider_response_id == "fake-1"


def test_provider_call_occurs_after_database_transaction_is_closed(
    session_factory, session, ready_project, clock
) -> None:
    class TransactionCheckingProvider(FakeProvider):
        def generate(self, request):
            with session_factory() as independent:
                independent.execute(
                    update(NovelProject)
                    .where(NovelProject.id == ready_project.id)
                    .values(title=NovelProject.title)
                )
                independent.commit()
            return super().generate(request)

    provider = TransactionCheckingProvider([response(plan_payload(1), 1)])
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )

    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    assert result.status == "AWAITING_PLAN_APPROVAL"


def test_outline_replacement_during_provider_call_atomically_pauses_completion(
    session_factory, session, ready_project, clock
) -> None:
    class OutlineReplacingProvider(FakeProvider):
        def generate(self, request):
            with session_factory() as independent:
                replacement = OutlineService(independent).create_candidate(
                    ready_project.id,
                    [
                        OutlineNodeInput(
                            key="replacement-during-call",
                            parent_key=None,
                            kind="book",
                            title="调用期间替换的大纲",
                            order=0,
                        )
                    ],
                    reason="replace during provider call",
                )
                OutlineService(independent).approve(replacement.id)
            return super().generate(request)

    provider = OutlineReplacingProvider(
        [response(plan_payload(1), 1, input_tokens=321, output_tokens=123, latency_ms=47)]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )

    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    session.expire_all()
    stored = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    attempt = session.scalar(select(ModelAttempt).where(ModelAttempt.step_id == step.id))
    assert result.status == "PAUSED_STALE_VERSION"
    assert result.completed_step is None
    assert stored.status == "PAUSED_STALE_VERSION"
    assert stored.current_position == 0
    assert stored.actual_input_tokens == 321
    assert stored.actual_output_tokens == 123
    assert step.status == "PAUSED"
    assert step.active_artifact_id is None
    assert attempt.status == "FAILED"
    assert attempt.error_code == "stale_outline"
    assert attempt.provider_response_id == "fake-1"
    assert attempt.input_tokens == 321
    assert attempt.output_tokens == 123
    assert attempt.latency_ms == 47
    assert session.scalar(select(func.count()).select_from(WorkflowArtifact)) == 0
    assert session.get(NovelProject, ready_project.id).active_workflow_id == workflow.id


def test_context_packet_uses_provider_effective_input_capacity(
    session_factory, session, ready_project, clock
) -> None:
    class SmallWindowProvider(FakeProvider):
        def capabilities(self, model: str) -> ProviderCapabilities:
            return ProviderCapabilities(5000, 1000, True, True, True, False)

    provider = SmallWindowProvider([response(plan_payload(1), 1)])
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )

    make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    packet = session.scalar(select(ContextPacket))
    assert provider.requests[0].max_input_tokens == 2976
    assert provider.requests[0].max_output_tokens == 1000
    assert packet.max_input_tokens == 2976
    assert packet.reserved_output_tokens == 1000
    serialized = json.dumps(
        asdict(provider.requests[0]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert (
        ConservativeEstimator().estimate(serialized)
        <= provider.requests[0].max_input_tokens
    )


def test_five_chapters_are_generated_and_copied_to_one_candidate_batch(
    orchestrator,
    approved_plan_workflow,
    scripted_five_chapter_provider,
    session,
) -> None:
    result = orchestrator.run_until_blocked(approved_plan_workflow.id)

    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert result.candidate_batch_id is not None
    chapters = BatchService(session).list_chapters(result.candidate_batch_id)
    assert [chapter.ordinal for chapter in chapters] == [1, 2, 3, 4, 5]
    assert [chapter.visible_char_count for chapter in chapters] == [
        4500,
        4501,
        4502,
        4503,
        4504,
    ]
    assert BatchService(session).official_chapter_statistics(
        approved_plan_workflow.project_id
    ).chapter_count == 0
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 1
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 12

    writer_requests = [
        request
        for request in scripted_five_chapter_provider.requests
        if request.metadata["agent_role"] == "chapter_writer"
    ]
    for ordinal, request in enumerate(writer_requests, start=1):
        prior = request.input_payload["previous_candidate_summaries"]
        assert [item["ordinal"] for item in prior] == list(range(1, ordinal))
        serialized = json.dumps(request.input_payload, ensure_ascii=False)
        for future in range(ordinal, 6):
            assert f"候选摘要{future}" not in serialized

    summarizer_requests = [
        request
        for request in scripted_five_chapter_provider.requests
        if request.metadata["agent_role"] == "chapter_summarizer"
    ]
    assert all(
        set(request.input_payload) == {"chapter", "chapter_plan"}
        for request in summarizer_requests
    )
    reviewer = next(
        request
        for request in scripted_five_chapter_provider.requests
        if request.metadata["agent_role"] == "batch_reviewer"
    )
    reviewer_json = json.dumps(reviewer.input_payload, ensure_ascii=False)
    assert "chapter_reports" in reviewer.input_payload
    for ordinal in range(1, 6):
        assert chr(0x4E00 + ordinal) * 100 not in reviewer_json

    indexed_summaries = session.scalars(
        select(ContextSource).where(
            ContextSource.state_scope == f"workflow:{approved_plan_workflow.id}",
            ContextSource.source_type == "chapter_summary_delta",
        )
    ).all()
    assert len(indexed_summaries) == 5


def test_invalid_length_retries_then_pauses_before_candidate_creation(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider(
        [
            response(plan_payload(1), 1),
            response({"title": "第1章", "body": "短" * 4499}, 2),
            response({"title": "第1章", "body": "短" * 4499}, 3),
        ]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    result = orchestrator.run_until_blocked(workflow.id)

    assert result.status == "PAUSED_ATTEMPTS"
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 0
    assert [request.metadata["agent_role"] for request in provider.requests] == [
        "batch_planner",
        "chapter_writer",
        "chapter_writer",
    ]


def test_unrelated_full_length_chapter_fails_deterministic_plan_coverage(
    session_factory, session, ready_project, clock
) -> None:
    unrelated = "无" * 4500
    provider = FakeProvider(
        [
            response(plan_payload(1), 1),
            response({"title": "第1章", "body": unrelated}, 2),
        ]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    result = orchestrator.advance(workflow.id)

    session.expire_all()
    writer = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
        )
    )
    attempt = session.scalar(
        select(ModelAttempt).where(ModelAttempt.step_id == writer.id)
    )
    assert result.completed_step is None
    assert writer.status == "PENDING"
    assert writer.active_artifact_id is None
    assert attempt.status == "FAILED"


def test_validation_recovery_recomputes_body_coverage_instead_of_hardcoding_success(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider([response(plan_payload(1), 1)])
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    service = WorkflowService(session, clock=clock)
    writer = service.claim_step(
        workflow.id, {"GENERATING_CHAPTERS"}, "recovery-writer"
    )
    attempt = service.record_attempt_start(writer.id, "a" * 64)
    unrelated = "无" * 4500
    chapter = service.complete_attempt(
        attempt.id,
        response({"title": "第1章", "body": unrelated}, 2),
        {"title": "第1章", "body": unrelated},
    )
    assert session.scalar(
        select(func.count())
        .select_from(WorkflowArtifact)
        .where(WorkflowArtifact.kind == "chapter_validation")
    ) == 0

    recovered = orchestrator._validation_payload(session, chapter)

    assert recovered["approved_goal_present"] is False
    assert recovered["approved_key_event_present"] is False


def test_provider_attempt_exhaustion_pauses_without_skipping_the_step(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider(
        [response(plan_payload(1), 1), ProviderTimeout("one"), ProviderTimeout("two")]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    result = orchestrator.run_until_blocked(workflow.id)

    step = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "WRITING",
            WorkflowStep.ordinal == 1,
        )
    )
    assert result.status == "PAUSED_ATTEMPTS"
    assert step.status == "PAUSED"
    assert step.attempt_count == 2


def test_two_expired_running_attempts_pause_without_leaking_or_calling_provider(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider([])
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    service = WorkflowService(session, clock=clock)
    for attempt_number in (1, 2):
        step = service.claim_step(workflow.id, {"PLANNING"}, "crashing-worker")
        service.record_attempt_start(step.id, f"{attempt_number}" * 64)
        clock.advance(seconds=301)
        assert service.recover_expired_claims(workflow.id, clock.now()) == 1

    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    session.expire_all()
    stored = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    attempts = session.scalars(
        select(ModelAttempt)
        .where(ModelAttempt.step_id == step.id)
        .order_by(ModelAttempt.attempt_number)
    ).all()
    assert result.status == "PAUSED_ATTEMPTS"
    assert stored.status == "PAUSED_ATTEMPTS"
    assert step.status == "PAUSED"
    assert step.attempt_count == 2
    assert [attempt.status for attempt in attempts] == ["FAILED", "FAILED"]
    assert provider.requests == []


def test_required_overflow_pauses_before_attempt_and_provider_call(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider([response(plan_payload(1), 1)])
    tiny = WorkflowBudgets(
        planner_input=1,
        planner_output=1,
        writer_input=1,
        writer_output=1,
        summarizer_input=1,
        summarizer_output=1,
        reviewer_input=1,
        reviewer_output=1,
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, tiny
    )

    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    assert result.status == "PAUSED_CONTEXT_OVERFLOW"
    assert provider.requests == []
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 0
    assert session.scalar(select(func.count()).select_from(ContextPacket)) == 0


def test_stale_outline_pauses_without_provider_call(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider([response(plan_payload(1), 1)])
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    replacement = OutlineService(session).create_candidate(
        ready_project.id,
        [
            OutlineNodeInput(
                key="replacement",
                parent_key=None,
                kind="book",
                title="新版总纲",
                order=0,
            )
        ],
        reason="replacement",
    )
    OutlineService(session).approve(replacement.id)

    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)

    assert result.status == "PAUSED_STALE_VERSION"
    assert provider.requests == []


def test_reviewer_evidence_is_bounded_offset_bearing_and_first_query_is_preserved(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider(
        [
            response(plan_payload(1), 1),
            response({"title": "第1章", "body": chapter_body(1)}, 2),
            response({"summary": "候选摘要1", "state_delta": {"stage": 1}}, 3),
            response(
                {
                    "passed": False,
                    "issues": ["需要核对大纲"],
                    "evidence_queries": [" 全书总纲 ", "全书总纲"],
                },
                4,
            ),
            response({"passed": True, "issues": [], "evidence_queries": []}, 5),
        ]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    result = orchestrator.run_until_blocked(workflow.id)

    reviews = session.scalars(
        select(WorkflowArtifact)
        .where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "batch_review",
        )
        .order_by(WorkflowArtifact.created_at)
    ).all()
    review_requests = [
        request
        for request in provider.requests
        if request.metadata["agent_role"] == "batch_reviewer"
    ]
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert len(reviews) == 2
    assert reviews[0].payload["evidence_queries"] == [" 全书总纲 ", "全书总纲"]
    evidence = review_requests[1].input_payload["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["query"] == "全书总纲"
    assert evidence[0]["excerpt_start"] >= 0
    assert evidence[0]["excerpt_end"] > evidence[0]["excerpt_start"]
    assert evidence[0]["excerpt_end"] - evidence[0]["excerpt_start"] <= 1200


def test_failed_final_review_keeps_first_evidence_query_and_never_creates_batch(
    session_factory, session, ready_project, clock
) -> None:
    provider = FakeProvider(
        [
            response(plan_payload(1), 1),
            response({"title": "第1章", "body": chapter_body(1)}, 2),
            response({"summary": "候选摘要1", "state_delta": {}}, 3),
            response(
                {
                    "passed": False,
                    "issues": ["核对"],
                    "evidence_queries": ["全书总纲"],
                },
                4,
            ),
            response(
                {
                    "passed": False,
                    "issues": ["仍不一致"],
                    "evidence_queries": [],
                },
                5,
            ),
        ]
    )
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")

    result = orchestrator.run_until_blocked(workflow.id)

    first_review = session.scalar(
        select(WorkflowArtifact)
        .where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "batch_review",
        )
        .order_by(WorkflowArtifact.created_at)
    )
    assert result.status == "PAUSED_REVIEW"
    assert first_review.payload["evidence_queries"] == ["全书总纲"]
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 0


def test_candidate_batch_creation_is_exactly_once_after_create_then_crash(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    assert orchestrator.advance(workflow.id).completed_step == "WRITING"
    assert orchestrator.advance(workflow.id).completed_step == "SUMMARIZING"
    assert orchestrator.advance(workflow.id).completed_step == "REVIEWING"

    original_save = BatchService.save_candidate_chapter
    crashed = False

    def crash_once(self, *args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated post-create crash")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(BatchService, "save_candidate_chapter", crash_once)
    with pytest.raises(RuntimeError, match="post-create crash"):
        orchestrator.advance(workflow.id)

    session.expire_all()
    created = session.scalars(
        select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)
    ).all()
    assert len(created) == 1
    assert created[0].status == "draft"

    clock.advance(seconds=301)
    result = orchestrator.advance(workflow.id)

    session.expire_all()
    batches = session.scalars(
        select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)
    ).all()
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert len(batches) == 1
    assert BatchService(session).list_chapters(batches[0].id)[0].ordinal == 1


def test_committed_candidate_copy_is_reused_exactly_once_after_crash(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)

    original_save = BatchService.save_candidate_chapter
    crashed = False

    def crash_after_committed_copy(self, *args, **kwargs):
        nonlocal crashed
        chapter = original_save(self, *args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated committed-copy crash")
        return chapter

    monkeypatch.setattr(
        BatchService, "save_candidate_chapter", crash_after_committed_copy
    )
    with pytest.raises(RuntimeError, match="committed-copy crash"):
        orchestrator.advance(workflow.id)

    session.expire_all()
    batch = session.scalar(
        select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)
    )
    assert len(BatchService(session).list_chapters(batch.id)) == 1

    clock.advance(seconds=301)
    result = orchestrator.advance(workflow.id)

    session.expire_all()
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert session.scalar(
        select(func.count())
        .select_from(WritingBatch)
        .where(WritingBatch.source_workflow_id == workflow.id)
    ) == 1
    assert len(BatchService(session).list_chapters(batch.id)) == 1


def test_post_reconcile_commit_crash_is_idempotent_on_next_advance(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)

    original_reconcile = WorkflowService.reconcile_batch_decision
    calls = 0

    def crash_after_final_reconcile(self, workflow_id):
        nonlocal calls
        result = original_reconcile(self, workflow_id)
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated post-reconcile-commit crash")
        return result

    monkeypatch.setattr(
        WorkflowService, "reconcile_batch_decision", crash_after_final_reconcile
    )
    with pytest.raises(RuntimeError, match="post-reconcile-commit crash"):
        orchestrator.advance(workflow.id)

    session.expire_all()
    stored = session.get(GenerationWorkflow, workflow.id)
    candidate_step = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow.id,
            WorkflowStep.kind == "CREATING_CANDIDATE_BATCH",
        )
    )
    assert stored.status == "AWAITING_CONTENT_APPROVAL"
    assert candidate_step.status == "COMPLETED"

    result = orchestrator.advance(workflow.id)

    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert result.candidate_batch_id == stored.candidate_batch_id
    assert session.scalar(
        select(func.count())
        .select_from(WritingBatch)
        .where(WritingBatch.source_workflow_id == workflow.id)
    ) == 1


def test_terminal_candidate_rejection_seen_by_later_lookup_reconciles_directly(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)

    original_get = BatchService.get
    original_reject = BatchService.reject
    draft_lookups = 0
    deciding = False

    def reject_on_second_draft_lookup(self, batch_id):
        nonlocal deciding, draft_lookups
        batch = original_get(self, batch_id)
        if not deciding and batch.status == "draft":
            draft_lookups += 1
            if draft_lookups == 2:
                deciding = True
                try:
                    original_reject(self, batch_id, "author rejected during copy")
                finally:
                    deciding = False
                return original_get(self, batch_id)
        return batch

    monkeypatch.setattr(BatchService, "get", reject_on_second_draft_lookup)

    result = orchestrator.advance(workflow.id)

    session.expire_all()
    assert result.status == "REJECTED"
    assert session.get(GenerationWorkflow, workflow.id).status == "REJECTED"
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None


def test_ready_candidate_batch_reconciles_once_after_pre_reconcile_crash(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)

    original_reconcile = WorkflowService.reconcile_batch_decision
    calls = 0

    def crash_on_ready(self, workflow_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated ready-before-reconcile crash")
        return original_reconcile(self, workflow_id)

    monkeypatch.setattr(WorkflowService, "reconcile_batch_decision", crash_on_ready)
    with pytest.raises(RuntimeError, match="ready-before-reconcile"):
        orchestrator.advance(workflow.id)

    session.expire_all()
    batch = session.scalar(
        select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)
    )
    assert batch.status == "ready_for_review"
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 4

    clock.advance(seconds=301)
    result = orchestrator.advance(workflow.id)

    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert result.candidate_batch_id == batch.id
    assert session.scalar(
        select(func.count())
        .select_from(WritingBatch)
        .where(WritingBatch.source_workflow_id == workflow.id)
    ) == 1


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approve", "COMPLETED"), ("reject", "REJECTED")],
)
def test_terminal_candidate_decision_reconciles_after_ready_before_reconcile_crash(
    session_factory,
    session,
    ready_project,
    clock,
    monkeypatch,
    decision,
    expected_status,
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)
    orchestrator.advance(workflow.id)

    original_reconcile = WorkflowService.reconcile_batch_decision
    calls = 0

    def crash_on_ready(self, workflow_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated ready-before-reconcile crash")
        return original_reconcile(self, workflow_id)

    monkeypatch.setattr(WorkflowService, "reconcile_batch_decision", crash_on_ready)
    with pytest.raises(RuntimeError, match="ready-before-reconcile"):
        orchestrator.advance(workflow.id)

    session.expire_all()
    batch = session.scalar(
        select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)
    )
    if decision == "approve":
        BatchService(session).approve(batch.id, workflow.base_outline_version_id)
    else:
        BatchService(session).reject(batch.id, "author rejected candidate")

    clock.advance(seconds=301)
    result = orchestrator.advance(workflow.id)

    session.expire_all()
    assert result.status == expected_status
    assert result.candidate_batch_id == batch.id
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None
    assert session.scalar(
        select(func.count())
        .select_from(WritingBatch)
        .where(WritingBatch.source_workflow_id == workflow.id)
    ) == 1


def test_completed_summary_is_indexed_on_recovery_without_repeating_provider_call(
    session_factory, session, ready_project, clock, monkeypatch
) -> None:
    provider = FakeProvider(success_script(1))
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = make_orchestrator(session_factory, provider, clock)
    orchestrator.advance(workflow.id)
    WorkflowService(session, clock=clock).approve_plan(workflow.id, "author")
    orchestrator.advance(workflow.id)

    original_index = ContextIndexService.index_workflow_artifact
    crashed = False

    def crash_once(self, artifact_id):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated post-summary crash")
        return original_index(self, artifact_id)

    monkeypatch.setattr(ContextIndexService, "index_workflow_artifact", crash_once)
    with pytest.raises(RuntimeError, match="post-summary crash"):
        orchestrator.advance(workflow.id)

    assert [request.metadata["agent_role"] for request in provider.requests] == [
        "batch_planner",
        "chapter_writer",
        "chapter_summarizer",
    ]
    assert session.scalar(
        select(func.count())
        .select_from(ContextSource)
        .where(ContextSource.source_type == "chapter_summary_delta")
    ) == 0

    result = orchestrator.advance(workflow.id)

    assert result.completed_step == "REVIEWING"
    assert [request.metadata["agent_role"] for request in provider.requests].count(
        "chapter_summarizer"
    ) == 1
    assert session.scalar(
        select(func.count())
        .select_from(ContextSource)
        .where(
            ContextSource.state_scope == f"workflow:{workflow.id}",
            ContextSource.source_type == "chapter_summary_delta",
        )
    ) == 1

import pytest

from test_orchestrator import ready_project, response, session_factory, clock, make_orchestrator
from ainovel.providers.fake import FakeProvider
from ainovel.providers.demo import DemoFakeProvider
from ainovel.services.workflows import WorkflowService
from ainovel.services.batches import BatchService


def roadmap_payload(count=7):
    return {
        "goal": "找到主使", "start_state": "城门封锁", "end_state": "真相公开",
        "key_events": ["入城", "破案"], "foreshadowing": ["铜铃"],
        "nodes": [{"node_id": f"node-{i}", "ordinal": i, "title": f"第{i}步",
                   "goal": f"取得线索{i}", "dependencies": [f"node-{i-1}"] if i > 1 else []}
                  for i in range(1, count + 1)],
    }


def proposed(service, project_id, payload=None):
    stage = service.create(project_id, "入城查案，七次追查后揭露主使", "author")
    version = service.propose_roadmap(stage.id, "author", "fake", "scripted")
    version = service.generate_roadmap(version.id, FakeProvider([response(payload or roadmap_payload(), 1)]))
    return stage, version


def test_stage_requires_explicit_roadmap_approval_before_start(session, ready_project):
    from ainovel.services.stages import StageService

    service = StageService(session)
    stage = service.create(ready_project.id, "主角入城，调查失踪案并找到幕后主使", "author")
    assert stage.confirmed_chapters == 0
    with pytest.raises(ValueError, match="approved roadmap"):
        service.start_next_batch(stage.id, "author", "fake", "scripted")


def test_model_roadmap_is_persisted_with_actual_usage_and_needs_approval(session, ready_project):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    assert version.status == "PROPOSED"
    assert version.estimated_chapters == 7
    assert version.attempts_used == 1
    assert version.actual_input_tokens == 100
    assert version.actual_output_tokens == 50
    with pytest.raises(ValueError, match="approved roadmap"):
        service.start_next_batch(stage.id, "author", "fake", "scripted")
    service.approve_roadmap(stage.id, version.id, "author")
    first = service.start_next_batch(stage.id, "author", "fake", "demo")
    assert [n.stage_ordinal for n in first.nodes] == [1, 2, 3, 4, 5]
    assert [n.book_ordinal for n in first.nodes] == [1, 2, 3, 4, 5]
    assert first.workflow.generation_version == 2
    assert service.get(stage.id).confirmed_chapters == 0


def test_rolling_content_approval_advances_five_then_two_and_completes(
    session, ready_project, session_factory, clock
):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    first = service.start_next_batch(stage.id, "author", "fake", "demo")
    with pytest.raises(ValueError):
        service.start_next_batch(stage.id, "author", "fake", "demo")
    orchestrator = make_orchestrator(session_factory, DemoFakeProvider(), clock)
    for expected, batch_start in [(5, first), (7, None)]:
        if batch_start is None:
            batch_start = service.start_next_batch(stage.id, "author", "fake", "demo")
            assert [n.stage_ordinal for n in batch_start.nodes] == [6, 7]
            assert [n.book_ordinal for n in batch_start.nodes] == [6, 7]
        workflow_id = batch_start.workflow.id
        assert orchestrator.advance(workflow_id).status == "AWAITING_PLAN_APPROVAL"
        WorkflowService(session, clock=clock).approve_plan(workflow_id, "author")
        result = orchestrator.run_until_blocked(workflow_id, max_steps=50)
        assert result.status == "AWAITING_CONTENT_APPROVAL"
        assert service.get(stage.id).confirmed_chapters == expected - len(batch_start.nodes)
        BatchService(session).approve(result.candidate_batch_id, ready_project.official_outline_version_id)
        assert service.get(stage.id).confirmed_chapters == expected
        with pytest.raises(ValueError, match="approval conflict"):
            BatchService(session).approve(result.candidate_batch_id, ready_project.official_outline_version_id)
        assert service.get(stage.id).confirmed_chapters == expected
        WorkflowService(session, clock=clock).reconcile_batch_decision(workflow_id)
    with pytest.raises(ValueError, match="complete"):
        service.start_next_batch(stage.id, "author", "fake", "demo")


def test_stage_context_reserves_nodes_and_does_not_repeat_full_roadmap_in_writing(session, ready_project, session_factory, clock):
    from ainovel.services.stages import StageService
    from ainovel.models import WorkflowStep
    from sqlalchemy import select
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    started = service.start_next_batch(stage.id, "author", "fake", "demo", 1)
    orchestrator = make_orchestrator(session_factory, DemoFakeProvider(), clock)
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == started.workflow.id))
    request = orchestrator._build_request(step)
    assert request.input_payload["stage"]["nodes"][0]["node_id"] == "node-1"
    assert len(request.input_payload["stage"]["roadmap"]["nodes"]) == 7
    assert orchestrator.advance(started.workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(started.workflow.id, "author")
    session.expire_all()
    writer = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == started.workflow.id, WorkflowStep.kind == "WRITING"))
    request = orchestrator._build_request(writer)
    assert request.input_payload["stage"]["nodes"][0]["stage_ordinal"] == 1
    assert request.input_payload["stage"]["nodes"][0]["book_ordinal"] == 1
    assert request.input_payload["chapter_plan"]["goal"] == "取得线索1"
    assert "roadmap" not in request.input_payload["stage"]


def test_demo_provider_can_propose_a_stage_roadmap(session, ready_project):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, "入城查案", "author")
    version = service.propose_roadmap(stage.id, "author", "fake", "demo")
    result = service.generate_roadmap(version.id, DemoFakeProvider())
    assert result.status == "PROPOSED"
    assert result.estimated_chapters > 5


@pytest.mark.parametrize("fault", ["order", "dependency", "duplicate", "oversize", "extra"], ids=["order", "dependency", "duplicate", "oversize", "extra"])
def test_invalid_roadmap_is_rejected_and_attempt_budget_survives_reload(session, ready_project, fault):
    from ainovel.services.stages import StageService
    from ainovel.models import StageModelAttempt
    from sqlalchemy import select
    payload = roadmap_payload(101 if fault == "oversize" else 7)
    if fault == "order":
        payload["nodes"][0]["ordinal"] = 2
    elif fault == "dependency":
        payload["nodes"][0]["dependencies"] = ["node-7"]
    elif fault == "duplicate":
        payload["nodes"][1]["node_id"] = "node-1"
    elif fault == "extra":
        payload["estimated_chapters"] = 8
    service = StageService(session)
    stage, version = proposed(service, ready_project.id, payload)
    assert version.status == "PAUSED_INVALID"
    assert version.payload is None
    version = StageService(session).generate_roadmap(version.id, FakeProvider([response(payload, 2)]))
    assert version.attempts_used == 2
    version = StageService(session).generate_roadmap(version.id, FakeProvider([]))
    assert version.status == "PAUSED_BUDGET"
    assert version.attempts_used == 2
    attempts = session.scalars(select(StageModelAttempt).where(StageModelAttempt.roadmap_id == version.id)).all()
    assert len(attempts) == 2
    assert sum(a.output_tokens for a in attempts) == 100


@pytest.mark.parametrize("budget_kind", ["input", "total", "output"], ids=["input", "total", "output"])
def test_roadmap_budget_limits_do_not_truncate_or_dispatch_over_budget(session, ready_project, budget_kind):
    from ainovel.services.stages import StageService, StageBudgets
    service = StageService(session)
    stage = service.create(ready_project.id, "架构" * 20000 if budget_kind == "input" else "查案", "author")
    budgets = StageBudgets(total_output_tokens=1) if budget_kind == "total" else StageBudgets()
    version = service.propose_roadmap(stage.id, "author", "fake", "scripted", budgets=budgets)
    provider = FakeProvider([response(roadmap_payload(), 1, output_tokens=8001)]) if budget_kind == "output" else FakeProvider([])
    result = service.generate_roadmap(version.id, provider)
    assert result.status == ("PAUSED_CONTEXT_OVERFLOW" if budget_kind == "input" else "PAUSED_BUDGET")
    assert result.payload is None
    assert result.attempts_used == (1 if budget_kind == "output" else 0)
    if budget_kind == "input":
        assert len(result.input_snapshot["architecture"]) == 40000


def test_roadmap_completion_is_fenced_when_author_revises_inputs_during_call(session, ready_project, session_factory):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, "查案", "author")
    version = service.propose_roadmap(stage.id, "author", "fake", "scripted")
    class RevisingProvider(DemoFakeProvider):
        def generate(self, request):
            with session_factory() as other:
                StageService(other).propose_roadmap(stage.id, "author", "fake", "scripted", architecture="重新调查")
            return response(roadmap_payload(), 1)
    result = service.generate_roadmap(version.id, RevisingProvider())
    assert result.status == "PAUSED_STALE_VERSION"
    assert result.actual_input_tokens == 100
    assert result.payload is None


def test_stale_stage_workflow_cannot_accept_provider_completion(session, ready_project, session_factory, clock):
    from ainovel.services.stages import StageService
    from ainovel.models import StoryStage, WorkflowArtifact
    from sqlalchemy import update, select
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    started = service.start_next_batch(stage.id, "author", "fake", "demo", 1)
    class StaleProvider(DemoFakeProvider):
        def generate(self, request):
            with session_factory() as other:
                other.execute(update(StoryStage).where(StoryStage.id == stage.id).values(approved_roadmap_id=None))
                other.commit()
            return super().generate(request)
    result = make_orchestrator(session_factory, StaleProvider(), clock).advance(started.workflow.id)
    assert result.status == "PAUSED_STALE_VERSION"
    assert session.scalar(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == started.workflow.id)) is None


def test_atomic_stage_start_rolls_back_workflow_and_ownership_on_mapping_failure(session, ready_project, monkeypatch):
    from ainovel.services.stages import StageService
    from ainovel.models import GenerationWorkflow, NovelProject, StageWorkflow
    from sqlalchemy import select
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    original = StageService._audit
    def fail_start(self, stage, action, actor, details):
        if action == "stage_batch_started":
            raise RuntimeError("mapping persistence failed")
        return original(self, stage, action, actor, details)
    monkeypatch.setattr(StageService, "_audit", fail_start)
    with pytest.raises(RuntimeError, match="mapping persistence"):
        service.start_next_batch(stage.id, "author", "fake", "demo")
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None
    assert session.scalar(select(GenerationWorkflow)) is None
    assert session.scalar(select(StageWorkflow)) is None


def test_concurrent_starts_reserve_one_contiguous_range(session, ready_project, session_factory):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from ainovel.services.stages import StageService
    from ainovel.models import StageWorkflow, StageWorkflowNode
    from sqlalchemy import select
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    barrier = Barrier(2)
    stage_id = stage.id
    def start():
        with session_factory() as other:
            barrier.wait(timeout=5)
            try:
                return StageService(other).start_next_batch(stage_id, "author", "fake", "demo").workflow.id
            except ValueError:
                return None
    session.rollback()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start(), [1, 2]))
    assert sum(value is not None for value in results) == 1
    assert len(session.scalars(select(StageWorkflow)).all()) == 1
    assert [n.stage_ordinal for n in session.scalars(select(StageWorkflowNode).order_by(StageWorkflowNode.ordinal))] == [1, 2, 3, 4, 5]


def candidate_batch(service, stage, session, session_factory, clock, count=1):
    started = service.start_next_batch(stage.id, "author", "fake", "demo", count)
    orchestrator = make_orchestrator(session_factory, DemoFakeProvider(), clock)
    assert orchestrator.advance(started.workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session, clock=clock).approve_plan(started.workflow.id, "author")
    result = orchestrator.run_until_blocked(started.workflow.id, max_steps=50)
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    return started, result.candidate_batch_id


def test_revision_diff_preserves_approved_records_and_confirmed_node_prefix(session, ready_project, session_factory, clock):
    from copy import deepcopy
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage, original = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, original.id, "author")
    old_payload = deepcopy(original.payload)
    started, batch_id = candidate_batch(service, stage, session, session_factory, clock)
    BatchService(session).approve(batch_id, ready_project.official_outline_version_id)
    WorkflowService(session).reconcile_batch_decision(started.workflow.id)
    changed = deepcopy(old_payload)
    changed["nodes"][1]["goal"] = "改为调查码头"
    new = service.propose_roadmap(stage.id, "author", "fake", "scripted", architecture="改查码头")
    service.generate_roadmap(new.id, FakeProvider([response(changed, 2)]))
    diff = service.roadmap_diff(stage.id, new.id)
    assert diff["nodes"]["before"][1]["goal"] == "取得线索2"
    assert diff["nodes"]["after"][1]["goal"] == "改为调查码头"
    service.approve_roadmap(stage.id, new.id, "author")
    assert service.roadmap(original.id).status == "APPROVED"
    assert service.roadmap(original.id).payload == old_payload
    assert service.get(stage.id).confirmed_chapters == 1
    changed["nodes"][0]["goal"] = "改写已确认章"
    invalid = service.propose_roadmap(stage.id, "author", "fake", "scripted")
    service.generate_roadmap(invalid.id, FakeProvider([response(changed, 3)]))
    with pytest.raises(ValueError, match="confirmed nodes"):
        service.approve_roadmap(stage.id, invalid.id, "author")
    assert service.get(stage.id).approved_roadmap_id == new.id


@pytest.mark.parametrize("decision", ["cancel", "reject", "fail"], ids=["cancel", "reject", "fail"])
def test_nonapproved_work_never_advances_stage(session, ready_project, session_factory, clock, decision):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    if decision == "reject":
        started, batch_id = candidate_batch(service, stage, session, session_factory, clock)
        BatchService(session).reject(batch_id, "重写")
        WorkflowService(session).reconcile_batch_decision(started.workflow.id)
    else:
        started = service.start_next_batch(stage.id, "author", "fake", "demo", 1)
        if decision == "fail":
            provider = FakeProvider([response({"wrong": "plan"}, 1), response({"wrong": "plan"}, 2)])
            result = make_orchestrator(session_factory, provider, clock).run_until_blocked(started.workflow.id)
            assert result.status == "PAUSED_ATTEMPTS"
        else:
            from ainovel.providers.contracts import ProviderUnavailable
            result = make_orchestrator(session_factory, FakeProvider([ProviderUnavailable("offline")]), clock).advance(started.workflow.id)
            assert result.status == "PAUSED_PROVIDER"
        WorkflowService(session).cancel(started.workflow.id, "author")
    assert service.get(stage.id).confirmed_chapters == 0
    again = service.start_next_batch(stage.id, "author", "fake", "demo", 1)
    assert [n.stage_ordinal for n in again.nodes] == [1]
    assert [n.book_ordinal for n in again.nodes] == [1]


def test_batch_progress_failure_rolls_back_official_chapters_counters_and_audits(session, ready_project, session_factory, clock, monkeypatch):
    from ainovel.services.stages import StageService
    from ainovel.models import Chapter, AuditEvent, NovelProject, StageWorkflow
    from sqlalchemy import select
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    started, batch_id = candidate_batch(service, stage, session, session_factory, clock)
    original = StageService._audit
    def fail_progress(self, stage, action, actor, details):
        if action == "stage_progress_confirmed":
            raise RuntimeError("stage write failed")
        return original(self, stage, action, actor, details)
    monkeypatch.setattr(StageService, "_audit", fail_progress)
    with pytest.raises(RuntimeError, match="stage write"):
        BatchService(session).approve(batch_id, ready_project.official_outline_version_id)
    session.expire_all()
    assert service.get(stage.id).confirmed_chapters == 0
    assert BatchService(session).get(batch_id).status == "ready_for_review"
    assert session.scalar(select(Chapter).where(Chapter.batch_id == batch_id)).status == "candidate"
    assert session.get(NovelProject, ready_project.id).next_official_chapter_number == 1
    assert session.get(NovelProject, ready_project.id).active_batch_id == batch_id
    assert session.get(StageWorkflow, started.workflow.id).committed_batch_id is None
    assert session.scalar(select(AuditEvent).where(AuditEvent.action == "batch_approved")) is None


def test_active_batch_blocks_roadmap_edits_and_stale_map_blocks_approval(session, ready_project, session_factory, clock):
    from ainovel.services.stages import StageService
    from ainovel.models import StoryStage
    from sqlalchemy import update
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    future = service.propose_roadmap(stage.id, "author", "fake", "scripted")
    service.generate_roadmap(future.id, FakeProvider([response(roadmap_payload(8), 2)]))
    started, batch_id = candidate_batch(service, stage, session, session_factory, clock)
    with pytest.raises(ValueError, match="active"):
        service.approve_roadmap(stage.id, future.id, "author")
    with pytest.raises(ValueError, match="active"):
        service.propose_roadmap(stage.id, "author", "fake", "scripted")
    session.execute(update(StoryStage).where(StoryStage.id == stage.id).values(approved_roadmap_id=future.id))
    session.commit()
    with pytest.raises(ValueError, match="stale"):
        BatchService(session).approve(batch_id, ready_project.official_outline_version_id)
    assert service.get(stage.id).confirmed_chapters == 0
    assert BatchService(session).get(batch_id).status == "ready_for_review"


@pytest.mark.parametrize("count", [0, 6, -1, True, 1.5], ids=["zero", "six", "negative", "bool", "float"])
def test_stage_batch_size_is_server_validated(session, ready_project, count):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    with pytest.raises(ValueError, match="from 1 to 5"):
        service.start_next_batch(stage.id, "author", "fake", "demo", count)


def test_missing_stage_usage_is_unknown_and_never_grants_free_retry(session, ready_project):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, "查案", "author")
    version = service.propose_roadmap(stage.id, "author", "fake", "scripted")
    result = service.generate_roadmap(version.id, FakeProvider([response({}, 1, input_tokens=None, output_tokens=None)]))
    assert result.actual_input_tokens is None
    assert result.actual_output_tokens is None
    assert result.attempts_used == 1
    result = service.generate_roadmap(version.id, FakeProvider([response(roadmap_payload(), 2)]))
    assert result.attempts_used == 2
    assert result.actual_input_tokens is None
    assert result.actual_output_tokens is None


def test_planner_cannot_swap_reserved_node_for_a_later_goal(session, ready_project, session_factory, clock):
    from ainovel.services.stages import StageService
    from test_draft_repair import v2_plan_payload
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    started = service.start_next_batch(stage.id, "author", "fake", "scripted", 1)
    payload = v2_plan_payload()
    payload["chapters"][0]["title"] = "第1步"
    payload["chapters"][0]["goal"] = "取得线索7"
    result = make_orchestrator(session_factory, FakeProvider([response(payload, 1), response(payload, 2)]), clock).run_until_blocked(started.workflow.id)
    assert result.status == "PAUSED_ATTEMPTS"
    assert service.get(stage.id).confirmed_chapters == 0


def test_stage_start_propagates_caller_workflow_budgets(session, ready_project):
    from ainovel.services.stages import StageService
    from ainovel.services.workflows import WorkflowBudgets
    service = StageService(session)
    stage, version = proposed(service, ready_project.id)
    service.approve_roadmap(stage.id, version.id, "author")
    started = service.start_next_batch(stage.id, "author", "fake", "demo", 1, budgets=WorkflowBudgets(reviewer_output=4000))
    assert started.workflow.reviewer_output_tokens == 4000

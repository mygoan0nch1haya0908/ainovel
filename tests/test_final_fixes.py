from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, update
from sqlalchemy import event

from ainovel.context import RequiredContextOverflow
from ainovel.models import ModelAttempt, NovelProject, WorkflowStep, WorkflowPromptSnapshot, GenerationWorkflow, WritingBatch, AuditEvent
from ainovel.services.batches import BatchService
from ainovel.agents.contracts import ChapterSummaryDelta
from ainovel.agents.runner import AgentRunner
from ainovel.providers.contracts import ModelRequest
from ainovel.providers.openai import OpenAIProvider
from ainovel.providers.ollama import OllamaProvider
import httpx
from ainovel.providers.contracts import ProviderUnavailable
from ainovel.services.projects import ProjectService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.workflows.orchestrator import WorkflowOrchestrator
from ainovel.services.context import ContextBuilder, ContextIndexService, ContextService
from ainovel.models import Chapter, ContextSource, WorkflowArtifact
from ainovel.context import ConservativeEstimator
from dataclasses import asdict


class Clock:
    value = datetime(2026, 9, 11, tzinfo=timezone.utc)

    def now(self):
        return self.value


@pytest.fixture
def ready_project(session, project, official_outline):
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    return project


@pytest.mark.parametrize("operation", ["start", "overflow", "provider"])
def test_expired_caller_cannot_use_reclaimed_lease(session, client, ready_project, operation):
    clock = Clock()
    service = WorkflowService(session, clock)
    workflow = service.start(ready_project.id, "fake", "test", 1, DEFAULT_BUDGETS)
    first = service.claim_step(workflow.id, {"PLANNING"}, "shared-worker", lease_seconds=1)
    first_revision, step_id = first.revision, first.id
    clock.value += timedelta(seconds=2)
    with client.app.state.session_factory() as other:
        second_service = WorkflowService(other, clock)
        assert second_service.recover_expired_claims(workflow.id, clock.now()) == 1
        second = second_service.claim_step(workflow.id, {"PLANNING"}, "shared-worker")
        second_revision = second.revision
    with pytest.raises(ValueError, match="claim|conflict|lease"):
        if operation == "start":
            service.record_attempt_start(step_id, "a" * 64, claim_revision=first_revision)
        elif operation == "overflow":
            service.pause_context_overflow(step_id, "shared-worker", RequiredContextOverflow("test", 2, 1), claim_revision=first_revision)
        else:
            service.pause_provider_failure(step_id, "shared-worker", ProviderUnavailable("offline"), claim_revision=first_revision)
    service.record_attempt_start(step_id, "b" * 64, claim_revision=second_revision)
    with pytest.raises(ValueError, match="claim|conflict|lease"):
        service.record_attempt_start(step_id, "c" * 64, claim_revision=second_revision)
    # Even a caller that re-reads the now advanced revision cannot bypass the
    # atomic one-RUNNING-attempt predicate.
    with pytest.raises(ValueError, match="conflict"):
        service.record_attempt_start(step_id, "d" * 64, claim_revision=session.get(WorkflowStep, step_id).revision)
    assert session.scalar(select(func.count()).select_from(ModelAttempt).where(ModelAttempt.status == "RUNNING")) == 1
    assert session.get(WorkflowStep, step_id).attempt_count == 1


def test_openai_summary_snapshot_is_strict_wire_and_preserves_dynamic_delta(session, ready_project):
    workflow = WorkflowService(session).start(ready_project.id, "openai", "test", 1, DEFAULT_BUDGETS)
    snapshots = session.scalars(select(WorkflowPromptSnapshot).where(WorkflowPromptSnapshot.workflow_id == workflow.id)).all()
    def assert_closed(schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                assert schema.get("additionalProperties") is False
                assert set(schema["required"]) == set(schema["properties"])
            for value in schema.values():
                assert_closed(value)
        elif isinstance(schema, list):
            for value in schema:
                assert_closed(value)
    for snapshot in snapshots:
        assert_closed(snapshot.output_schema)
    snapshot = next(row for row in snapshots if row.role == "chapter_summarizer")
    delta = {"人物": {"位置": "北门", "items": [1, None, True, {"秘密": "雪"}]}, "score": 1.25}
    request = ModelRequest("test", snapshot.prompt_body, {"chapter": "正文"}, snapshot.output_schema, 11000, 4000, 5, {"schema_name": "chapter_summary_delta"})
    def create(**kwargs):
        schema = kwargs["text"]["format"]["schema"]
        assert kwargs["text"]["format"]["strict"] is True
        assert_closed(schema)
        assert schema["properties"]["state_delta"]["type"] == "string"
        return SimpleNamespace(output_text=json.dumps({"summary": "完成", "state_delta": json.dumps(delta, ensure_ascii=False)}), id="wire-response", usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    provider = OpenAIProvider(SimpleNamespace(responses=SimpleNamespace(create=create)), True)
    result = AgentRunner().run_with_response(provider, request, ChapterSummaryDelta)
    assert result.result.state_delta == delta
    assert result.response.provider_response_id == "wire-response"
    assert snapshot.output_schema == request.output_schema


def test_ollama_uses_full_capability_window_after_budget_reservation():
    from ainovel.context import effective_input_capacity
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps({"summary": "ok", "state_delta": '{"door":"north"}'})}})
    provider = OllamaProvider(httpx.Client(transport=httpx.MockTransport(handle)), "http://localhost", context_window_limit=16000, max_output_tokens_limit=4000)
    capacity = effective_input_capacity(16000, 16000, 4000)
    request = ModelRequest("test", "summary", {}, ChapterSummaryDelta.model_json_schema(), capacity, 4000, 5, {"schema_name": "summary"})
    result = AgentRunner().run(provider, request, ChapterSummaryDelta)
    assert result.state_delta == {"door": "north"}
    assert capacity < 12000
    assert seen[0]["options"] == {"num_ctx": 16000, "num_predict": 4000}


@pytest.mark.parametrize("source", ["manual", "wrong-stage", "foreign-owner"])
def test_batch_creation_requires_idle_project_or_current_candidate_workflow(session, ready_project, source):
    workflow = WorkflowService(session).start(ready_project.id, "fake", "test", 1, DEFAULT_BUDGETS)
    source_id = None if source == "manual" else workflow.id
    if source == "foreign-owner":
        session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow.id).values(status="CREATING_CANDIDATE_BATCH"))
        session.execute(update(NovelProject).where(NovelProject.id == ready_project.id).values(active_workflow_id=None))
        session.commit()
    with pytest.raises(ValueError, match="workflow|state changed"):
        BatchService(session).create(ready_project.id, ready_project.official_outline_version_id, 1, source_workflow_id=source_id)
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 0


@pytest.mark.parametrize("status", ["PAUSED_ATTEMPTS", "PAUSED_REVIEW", "PAUSED_STALE_VERSION"])
def test_author_can_cancel_paused_workflow_and_start_again(session, client, ready_project, status):
    import re
    service = WorkflowService(session)
    workflow = service.start(ready_project.id, "fake", "test", 1, DEFAULT_BUDGETS)
    session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow.id).values(status=status))
    session.execute(update(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id).values(status="PAUSED"))
    session.commit()
    snapshots = session.scalars(select(WorkflowPromptSnapshot).where(WorkflowPromptSnapshot.workflow_id == workflow.id)).all()
    before = [(row.id, row.output_schema, row.prompt_body) for row in snapshots]
    path = f"/workflows/{workflow.id}"
    page = client.get(path)
    assert f'action="{path}/cancel"' in page.text
    assert "data-confirm=" in page.text
    assert client.post(f"{path}/cancel", follow_redirects=False).status_code == 403
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
    response = client.post(f"{path}/cancel", data={"csrf_token": token}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == path
    session.expire_all()
    assert session.get(GenerationWorkflow, workflow.id).status == "CANCELLED"
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None
    assert [(row.id, row.output_schema, row.prompt_body) for row in snapshots] == before
    assert session.scalar(select(AuditEvent).where(AuditEvent.entity_id == workflow.id, AuditEvent.action == "workflow_cancelled")).actor == "author"
    restarted = service.start(ready_project.id, "fake", "test", 1, DEFAULT_BUDGETS)
    assert restarted.id != workflow.id


@pytest.mark.parametrize("conflict", ["owner", "batch"])
def test_cancel_preserves_changed_owner_and_phase_one_candidate_authority(session, client, ready_project, conflict):
    service = WorkflowService(session)
    workflow = service.start(ready_project.id, "fake", "test", 1, DEFAULT_BUDGETS)
    workflow_id, project_id = workflow.id, ready_project.id
    with client.app.state.session_factory() as other:
        if conflict == "batch":
            other.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(status="CREATING_CANDIDATE_BATCH"))
            other.commit()
            BatchService(other).create(project_id, ready_project.official_outline_version_id, 1, source_workflow_id=workflow_id)
        else:
            other.execute(update(NovelProject).where(NovelProject.id == project_id).values(active_workflow_id="new-owner"))
        other.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(status="PAUSED_STALE_VERSION"))
        other.commit()
    with pytest.raises(ValueError, match="cancel|candidate|owner"):
        service.cancel(workflow_id, "author")
    session.expire_all()
    assert session.get(GenerationWorkflow, workflow_id).status == "PAUSED_STALE_VERSION"
    assert session.get(NovelProject, project_id).active_workflow_id == ("new-owner" if conflict == "owner" else workflow_id)
    assert session.scalar(select(func.count()).select_from(AuditEvent).where(AuditEvent.action == "workflow_cancelled")) == 0


class RecordingDemo(DemoFakeProvider):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


def generate_batch(session, client, project_id, provider, count=1):
    service = WorkflowService(session)
    workflow = service.start(project_id, "fake", "demo", count, DEFAULT_BUDGETS)
    orchestrator = WorkflowOrchestrator(client.app.state.session_factory, ProviderRegistry({"fake": lambda: provider}), AgentRunner())
    assert orchestrator.run_until_blocked(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    service.approve_plan(workflow.id, "author")
    result = orchestrator.run_until_blocked(workflow.id)
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    return workflow.id, result.candidate_batch_id


def test_second_batch_planner_and_writer_use_approved_summary_delta_without_bodies(session, client, ready_project):
    first_provider = RecordingDemo()
    workflow_id, batch_id = generate_batch(session, client, ready_project.id, first_provider)
    batch_service = BatchService(session)
    chapter = batch_service.list_chapters(batch_id)[0]
    ContextIndexService(session).rebuild_official(ready_project.id)
    assert session.scalar(select(func.count()).select_from(ContextSource).where(ContextSource.state_scope == "official", ContextSource.source_type.in_(["chapter_summary", "event_chain"]))) == 0
    batch_service.approve(batch_id, ready_project.official_outline_version_id)
    WorkflowService(session).reconcile_batch_decision(workflow_id)
    second_provider = RecordingDemo()
    generate_batch(session, client, ready_project.id, second_provider)
    for request in second_provider.requests:
        if request.metadata["agent_role"] not in {"batch_planner", "chapter_writer"}:
            continue
        items = request.input_payload["context_packet"]["items"]
        summaries = [item for item in items if item["source_type"] == "chapter_summary" and item["state_scope"] == "official"]
        deltas = [item for item in items if item["source_type"] == "event_chain" and item["state_scope"] == "official"]
        assert len(summaries) == len(deltas) == 1
        assert "演示章节摘要。" in summaries[0]["text"]
        assert json.loads(deltas[0]["text"])["state_delta"] == {"demo": True}
        serialized = json.dumps(asdict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert chapter.body not in serialized
        assert ConservativeEstimator().estimate(serialized) <= request.max_input_tokens


@pytest.mark.parametrize("invalid", ["edited", "rejected", "inactive", "unvalidated", "foreign"])
def test_official_memory_excludes_unapproved_or_mismatched_artifacts(session, client, ready_project, invalid):
    workflow_id, batch_id = generate_batch(session, client, ready_project.id, RecordingDemo())
    batches = BatchService(session)
    chapter = batches.list_chapters(batch_id)[0]
    if invalid == "edited":
        batches.replace_candidate_body(chapter.id, "改" * 4500)
    if invalid == "rejected":
        batches.reject(batch_id, "author rejected")
    else:
        batches.approve(batch_id, ready_project.official_outline_version_id)
    if invalid in {"inactive", "unvalidated"}:
        changes = {"active_artifact_id": None} if invalid == "inactive" else {"status": "FAILED"}
        session.execute(update(WorkflowStep).where(WorkflowStep.workflow_id == workflow_id, WorkflowStep.kind == "SUMMARIZING").values(**changes))
        session.commit()
    if invalid == "foreign":
        other = ProjectService(session).create("other", 100000, 200000)
        session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(project_id=other.id))
        session.commit()
    ContextIndexService(session).rebuild_official(ready_project.id)
    memories = session.scalars(select(ContextSource).where(ContextSource.project_id == ready_project.id, ContextSource.state_scope == "official", ContextSource.source_type.in_(["chapter_summary", "event_chain"]))).all()
    assert memories == []


def test_official_memory_windows_follow_formal_numbers_across_seven_batches(session, client, ready_project):
    for count in [5, 5, 5, 5, 5, 5, 1]:
        workflow_id, batch_id = generate_batch(session, client, ready_project.id, RecordingDemo(), count)
        BatchService(session).approve(batch_id, ready_project.official_outline_version_id)
        WorkflowService(session).reconcile_batch_decision(workflow_id)
    # Old high revisions must never outrank newer formal chapter numbers.
    session.execute(update(Chapter).where(Chapter.project_id == ready_project.id, Chapter.official_chapter_number <= 2).values(revision=9999))
    session.commit()
    ContextIndexService(session).rebuild_official(ready_project.id)
    workflow = WorkflowService(session).start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS)
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    candidates = ContextBuilder(session).candidates_for_step(workflow.id, step.id)
    for source_type, expected in [("chapter_summary", list(range(31, 26, -1))), ("event_chain", list(range(31, 1, -1)))]:
        selected = [json.loads(item.text)["official_chapter_number"] for item in candidates if item.source_type == source_type]
        assert selected == expected


def test_manual_batch_cas_loses_to_workflow_started_after_manual_read(session, client, ready_project):
    project_id, outline_id = ready_project.id, ready_project.official_outline_version_id
    owner = []
    def start_before_manual_cas(orm_state):
        if orm_state.is_update and orm_state.statement.table.name == "novel_projects" and not owner:
            with client.app.state.session_factory() as other:
                owner.append(WorkflowService(other).start(project_id, "fake", "demo", 1, DEFAULT_BUDGETS).id)
    event.listen(session, "do_orm_execute", start_before_manual_cas)
    try:
        with pytest.raises(ValueError, match="state changed"):
            BatchService(session).create(project_id, outline_id, 1)
    finally:
        event.remove(session, "do_orm_execute", start_before_manual_cas)
    session.expire_all()
    project = session.get(NovelProject, project_id)
    assert project.active_workflow_id == owner[0]
    assert project.active_batch_id is None
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 0


def test_ollama_rejects_budget_that_does_not_fit_output_and_safety():
    from ainovel.providers.contracts import ProviderProtocolError
    def forbidden(_request):
        pytest.fail("over-budget request reached the local provider")
    provider = OllamaProvider(httpx.Client(transport=httpx.MockTransport(forbidden)), "http://localhost")
    request = ModelRequest("test", "summary", {}, ChapterSummaryDelta.model_json_schema(), 12000, 4000, 5, {"schema_name": "summary"})
    with pytest.raises(ProviderProtocolError, match="budget"):
        provider.generate(request)


def test_cancel_real_review_pause_retains_artifacts_and_attempt_audit(session, client, ready_project):
    class ReviewBlocker(RecordingDemo):
        def generate(self, request):
            response = super().generate(request)
            if request.metadata["agent_role"] == "batch_reviewer":
                return replace(response, structured={"passed": False, "issues": ["author review required"], "evidence_queries": []})
            return response
    provider = ReviewBlocker()
    service = WorkflowService(session)
    workflow = service.start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS)
    orchestrator = WorkflowOrchestrator(client.app.state.session_factory, ProviderRegistry({"fake": lambda: provider}), AgentRunner())
    assert orchestrator.run_until_blocked(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    service.approve_plan(workflow.id, "author")
    assert orchestrator.run_until_blocked(workflow.id).status == "PAUSED_REVIEW"
    artifacts = session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == workflow.id)).all()
    before = [(item.id, item.payload, item.content_hash) for item in artifacts]
    attempt_ids = session.scalars(select(ModelAttempt.id).join(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)).all()
    assert len(before) >= 4 and len(attempt_ids) == 4
    service.cancel(workflow.id, "author")
    session.expire_all()
    assert [(item.id, item.payload, item.content_hash) for item in artifacts] == before
    assert session.scalars(select(ModelAttempt.id).join(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)).all() == attempt_ids
    assert session.scalar(select(func.count()).select_from(WritingBatch)) == 0


def test_cancel_cas_preserves_owner_changed_after_cancel_read(session, client, ready_project):
    service = WorkflowService(session)
    workflow = service.start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS)
    workflow_id, project_id = workflow.id, ready_project.id
    session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(status="PAUSED_ATTEMPTS"))
    session.commit()
    def change_owner(orm_state):
        if orm_state.is_update and orm_state.statement.table.name == "generation_workflows":
            with client.app.state.session_factory() as other:
                other.execute(update(NovelProject).where(NovelProject.id == project_id).values(active_workflow_id="replacement-owner"))
                other.commit()
    event.listen(session, "do_orm_execute", change_owner)
    try:
        with pytest.raises(ValueError, match="cancel conflict"):
            service.cancel(workflow_id, "author")
    finally:
        event.remove(session, "do_orm_execute", change_owner)
    session.expire_all()
    assert session.get(NovelProject, project_id).active_workflow_id == "replacement-owner"
    assert session.get(GenerationWorkflow, workflow_id).status == "PAUSED_ATTEMPTS"


def test_paused_candidate_must_follow_phase_one_rejection_then_reconcile(session, client, ready_project, monkeypatch):
    from ainovel.services.outlines import OutlineService, OutlineNodeInput
    provider = RecordingDemo()
    service = WorkflowService(session)
    workflow = service.start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS)
    orchestrator = WorkflowOrchestrator(client.app.state.session_factory, ProviderRegistry({"fake": lambda: provider}), AgentRunner())
    assert orchestrator.run_until_blocked(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    service.approve_plan(workflow.id, "author")
    original_ready = BatchService.mark_ready
    def crash_after_ready(batch_service, batch_id):
        original_ready(batch_service, batch_id)
        raise RuntimeError("crash after Phase 1 ready commit")
    with monkeypatch.context() as patch:
        patch.setattr(BatchService, "mark_ready", crash_after_ready)
        with pytest.raises(RuntimeError, match="crash after"):
            orchestrator.run_until_blocked(workflow.id)
    batch = session.scalar(select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id))
    outline_service = OutlineService(session)
    replacement = outline_service.create_candidate(ready_project.id, [OutlineNodeInput(key="book", parent_key=None, kind="book", title="revised", order=0)], reason="author changed outline")
    outline_service.approve(replacement.id)
    assert service.claim_step(workflow.id, {"CREATING_CANDIDATE_BATCH"}, "recovery") is None
    assert session.get(GenerationWorkflow, workflow.id).status == "PAUSED_STALE_VERSION"
    with pytest.raises(ValueError, match="candidate"):
        service.cancel(workflow.id, "author")
    page = client.get(f"/workflows/{workflow.id}")
    assert f'action="/workflows/{workflow.id}/reconcile"' in page.text
    assert f"#batch-{batch.id}" in page.text
    BatchService(session).reject(batch.id, "outline changed")
    assert service.reconcile_batch_decision(workflow.id).status == "REJECTED"
    session.expire_all()
    assert session.get(NovelProject, ready_project.id).active_workflow_id is None
    assert service.start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS).id != workflow.id


def test_missing_workflow_detail_remains_not_found_with_cancel_action(client):
    assert client.get("/workflows/missing-workflow").status_code == 404

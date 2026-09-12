from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import unescape
import json
import re
import socket
from threading import Barrier

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

import ainovel.app as app_module
from ainovel.agents.contracts import (
    BatchPlanDraft,
    BatchReview,
    ChapterDraft,
    ChapterPlan,
    ChapterSummaryDelta,
)
from ainovel.agents.runner import AgentRunner
from ainovel.app import create_app
from ainovel.context import ITEM_FRAMING_TOKENS
from ainovel.models.audit import AuditEvent
from ainovel.models.context import ContextPacket, ContextSource
from ainovel.models.project import NovelProject
from ainovel.models.workflow import (
    GenerationWorkflow,
    ModelAttempt,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.providers.contracts import ModelResponse, ProviderUnavailable
from ainovel.providers.fake import FakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.batches import BatchService
from ainovel.services.context import ContextIndexService, ContextService
from ainovel.services.counting import count_visible_characters
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.prompts import PromptService
from ainovel.services.projects import ProjectService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService
from ainovel.workflows.orchestrator import WorkflowOrchestrator


def _response(result: object, number: int) -> ModelResponse:
    return ModelResponse(
        structured=result.model_dump(mode="json"),
        text=None,
        provider_response_id=f"acceptance-{number}",
        input_tokens=10,
        output_tokens=5,
        latency_ms=1,
    )


def _csrf(client: TestClient, path: str) -> str:
    page = client.get(path)
    assert page.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match is not None
    return unescape(match.group(1))


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("acceptance tests must not access DNS or sockets")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def ready_project(session, project, official_outline) -> NovelProject:
    ProjectService(session).add_constitution(
        project.id,
        {"genre": "historical fantasy", "voice": "close third"},
        author_approved=True,
    )
    ContextIndexService(session).rebuild_official(project.id)
    session.expire_all()
    ready = session.get(NovelProject, project.id)
    assert ready is not None
    return ready


@pytest.fixture
def five_chapter_registry() -> ProviderRegistry:
    plan = BatchPlanDraft(
        chapters=[
            ChapterPlan(
                ordinal=ordinal,
                title=f"第{ordinal}章",
                goal="甲",
                ending_hook="甲",
            )
            for ordinal in range(1, 6)
        ]
    )
    scripted = [_response(plan, 1)]
    call_number = 2
    for ordinal in range(1, 6):
        scripted.extend(
            [
                _response(
                    ChapterDraft(
                        title=f"第{ordinal}章",
                        body="甲" * (4499 + ordinal),
                    ),
                    call_number,
                ),
                _response(
                    ChapterSummaryDelta(
                        summary=f"候选摘要{ordinal}",
                        state_delta={"last_completed_ordinal": ordinal},
                    ),
                    call_number + 1,
                ),
            ]
        )
        call_number += 2
    scripted.append(
        _response(BatchReview(passed=True, issues=[], evidence_queries=[]), call_number)
    )
    provider = FakeProvider(scripted)
    return ProviderRegistry({"fake": lambda: provider})


@pytest.fixture
def provider_registry(five_chapter_registry: ProviderRegistry) -> ProviderRegistry:
    return five_chapter_registry


@pytest.fixture
def session_factory(client: TestClient):
    return client.app.state.session_factory


def _one_chapter_script(*extra: ModelResponse) -> list[ModelResponse]:
    plan = BatchPlanDraft(
        chapters=[
            ChapterPlan(ordinal=1, title="第一章", goal="甲", ending_hook="甲")
        ]
    )
    return [
        _response(plan, 1),
        _response(ChapterDraft(title="第一章", body="甲" * 4500), 2),
        _response(
            ChapterSummaryDelta(summary="候选摘要1", state_delta={"stage": 1}), 3
        ),
        *extra,
    ]


def _orchestrator(
    session_factory,
    provider: FakeProvider,
    context_service=ContextService,
) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(
        session_factory,
        ProviderRegistry({"fake": lambda: provider}),
        AgentRunner(),
        context_service,
        PromptService,
        worker_id="acceptance-worker",
    )


def start_fake_workflow(client: TestClient, project_id: str, count: int) -> str:
    path = f"/projects/{project_id}"
    response = client.post(
        f"{path}/workflows",
        data={
            "provider_name": "fake",
            "model_name": "scripted",
            "requested_chapters": str(count),
            "csrf_token": _csrf(client, path),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/workflows/")
    return location.rsplit("/", 1)[-1]


def run_workflow(client: TestClient, workflow_id: str) -> None:
    path = f"/workflows/{workflow_id}"
    response = client.post(
        f"{path}/run",
        data={"csrf_token": _csrf(client, path)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def approve_plan(client: TestClient, workflow_id: str) -> None:
    path = f"/workflows/{workflow_id}"
    response = client.post(
        f"{path}/plan/approve",
        data={"csrf_token": _csrf(client, path)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def reconcile_workflow(client: TestClient, workflow_id: str) -> None:
    path = f"/workflows/{workflow_id}"
    response = client.post(
        f"{path}/reconcile",
        data={"csrf_token": _csrf(client, path)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == path


def approve_candidate_batch(client: TestClient, batch_id: str) -> None:
    response = client.post(
        f"/batches/{batch_id}/approve",
        data={"csrf_token": _csrf(client, "/")},
        follow_redirects=False,
    )
    assert response.status_code == 303


def workflow_status(session, workflow_id: str) -> str:
    session.expire_all()
    workflow = session.get(GenerationWorkflow, workflow_id)
    assert workflow is not None
    return workflow.status


def load_workflow(session, workflow_id: str) -> GenerationWorkflow:
    session.expire_all()
    workflow = session.get(GenerationWorkflow, workflow_id)
    assert workflow is not None
    return workflow


def test_author_can_plan_generate_and_approve_five_chapters_offline(
    client: TestClient,
    session,
    ready_project: NovelProject,
    five_chapter_registry: ProviderRegistry,
) -> None:
    workflow_id = start_fake_workflow(client, ready_project.id, 5)
    run_workflow(client, workflow_id)
    assert workflow_status(session, workflow_id) == "AWAITING_PLAN_APPROVAL"
    approve_plan(client, workflow_id)
    run_workflow(client, workflow_id)
    workflow = load_workflow(session, workflow_id)
    assert workflow.status == "AWAITING_CONTENT_APPROVAL"
    assert workflow.candidate_batch_id is not None
    chapters = BatchService(session).list_chapters(workflow.candidate_batch_id)
    assert [chapter.visible_char_count for chapter in chapters] == [
        4500,
        4501,
        4502,
        4503,
        4504,
    ]
    assert (
        BatchService(session).official_chapter_statistics(ready_project.id).chapter_count
        == 0
    )
    approve_candidate_batch(client, workflow.candidate_batch_id)
    assert workflow_status(session, workflow_id) == "AWAITING_CONTENT_APPROVAL"
    session.expire_all()
    assert (
        session.get(NovelProject, ready_project.id).active_workflow_id == workflow_id
    )
    workflow_page = client.get(f"/workflows/{workflow_id}")
    assert workflow_page.status_code == 200
    assert f'action="/workflows/{workflow_id}/reconcile"' in workflow_page.text
    assert "同步候选审批结果" in workflow_page.text
    reconcile_workflow(client, workflow_id)
    assert workflow_status(session, workflow_id) == "COMPLETED"
    assert (
        BatchService(session).official_chapter_statistics(ready_project.id).chapter_count
        == 5
    )


def test_reconcile_rejects_missing_csrf(
    client: TestClient, session, ready_project: NovelProject
) -> None:
    workflow_id = start_fake_workflow(client, ready_project.id, 5)

    response = client.post(
        f"/workflows/{workflow_id}/reconcile", follow_redirects=False
    )

    assert response.status_code == 403


def test_restart_after_each_model_step_reuses_every_valid_artifact(
    session,
    session_factory,
    ready_project: NovelProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        _one_chapter_script(
            _response(BatchReview(passed=True, issues=[], evidence_queries=[]), 4)
        )
    )
    workflow = WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )

    for expected_kind in ("PLANNING", "WRITING", "SUMMARIZING", "REVIEWING"):
        orchestrator = _orchestrator(session_factory, provider)

        def crash_after_commit(_workflow_id: str):
            raise RuntimeError(f"simulated crash after {expected_kind}")

        monkeypatch.setattr(orchestrator, "_current_result", crash_after_commit)
        with pytest.raises(RuntimeError, match=f"after {expected_kind}"):
            orchestrator.advance(workflow.id)

        session.expire_all()
        completed = session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.kind == expected_kind,
            )
        )
        assert completed is not None
        assert completed.status == "COMPLETED"
        assert completed.active_artifact_id is not None
        if expected_kind == "PLANNING":
            WorkflowService(session).approve_plan(workflow.id, "author")

    result = _orchestrator(session_factory, provider).advance(workflow.id)

    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert [request.metadata["agent_role"] for request in provider.requests] == [
        "batch_planner",
        "chapter_writer",
        "chapter_summarizer",
        "batch_reviewer",
    ]
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 4


def test_batch_approval_is_reconciled_after_a_process_gap(
    session, session_factory, ready_project: NovelProject
) -> None:
    provider = FakeProvider(
        _one_chapter_script(
            _response(BatchReview(passed=True, issues=[], evidence_queries=[]), 4)
        )
    )
    workflow = WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    orchestrator = _orchestrator(session_factory, provider)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session).approve_plan(workflow.id, "author")
    generated = orchestrator.run_until_blocked(workflow.id)
    assert generated.status == "AWAITING_CONTENT_APPROVAL"
    assert generated.candidate_batch_id is not None

    BatchService(session).approve(
        generated.candidate_batch_id, workflow.base_outline_version_id
    )
    assert workflow_status(session, workflow.id) == "AWAITING_CONTENT_APPROVAL"

    with session_factory() as restarted_session:
        reconciled = WorkflowService(restarted_session).reconcile_batch_decision(
            workflow.id
        )

    assert reconciled.status == "COMPLETED"
    session.expire_all()
    stored_project = session.get(NovelProject, ready_project.id)
    assert stored_project is not None
    assert stored_project.active_workflow_id is None
    assert (
        BatchService(session).official_chapter_statistics(ready_project.id).chapter_count
        == 1
    )


def test_two_concurrent_starts_leave_one_project_owner_and_one_workflow(
    client: TestClient, session_factory, ready_project: NovelProject
) -> None:
    ownership_barrier = Barrier(2)

    def synchronize_ownership(
        _connection, _cursor, statement, _parameters, _context, _many
    ) -> None:
        normalized = " ".join(statement.casefold().split())
        if normalized.startswith("update novel_projects set active_workflow_id="):
            ownership_barrier.wait(timeout=5)

    def start() -> tuple[str, str]:
        with session_factory() as independent_session:
            try:
                workflow = WorkflowService(independent_session).start(
                    ready_project.id, "fake", "scripted", 2, DEFAULT_BUDGETS
                )
            except ValueError as error:
                return "error", str(error)
            return "ok", workflow.id

    event.listen(
        client.app.state.engine, "before_cursor_execute", synchronize_ownership
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(start), workers.submit(start)]
            outcomes = [future.result(timeout=15) for future in futures]
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
            .where(GenerationWorkflow.project_id == ready_project.id)
        ) == 1
        stored_project = verify_session.get(NovelProject, ready_project.id)
        assert stored_project is not None
        assert stored_project.active_workflow_id == next(
            value for kind, value in outcomes if kind == "ok"
        )


def test_required_context_overflow_pauses_before_any_provider_request(
    session, session_factory, ready_project: NovelProject
) -> None:
    provider = FakeProvider([_one_chapter_script()[0]])
    constitution = ProjectService(session).add_constitution(
        ready_project.id,
        {"required_world_law": "界" * 9_000},
        author_approved=True,
    )
    workflow = WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    observed: dict[str, object] = {}

    class RecordingContextService(ContextService):
        def build_packet(
            self, workflow_id, step_id, required, optional, limits
        ):
            capacity = limits["input_capacity_tokens"]
            fixed = limits["fixed_overhead_tokens"]
            observed["capacity"] = capacity
            observed["fixed"] = fixed
            observed["required_keys"] = tuple(item.stable_key for item in required)
            observed["required_cost"] = sum(
                self.budgeter.estimator.estimate(item.text) + ITEM_FRAMING_TOKENS
                for item in required
            )
            return super().build_packet(
                workflow_id, step_id, required, optional, limits
            )

    result = _orchestrator(
        session_factory, provider, RecordingContextService
    ).advance(workflow.id)

    assert result.status == "PAUSED_CONTEXT_OVERFLOW"
    assert observed["required_keys"] == (f"constitution:{constitution.id}",)
    assert observed["fixed"] < observed["capacity"]
    assert observed["fixed"] + observed["required_cost"] > observed["capacity"]
    assert provider.requests == []
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 0
    assert session.scalar(select(func.count()).select_from(ContextPacket)) == 0
    stored = load_workflow(session, workflow.id)
    assert stored.last_error_code == "required_context_overflow"
    assert (
        stored.last_error_detail
        == "required context exceeds the available input budget"
    )
    assert "required_world_law" not in stored.last_error_detail
    pause = session.scalar(
        select(AuditEvent).where(
            AuditEvent.entity_id == workflow.id,
            AuditEvent.action == "workflow_paused",
        )
    )
    assert pause is not None
    assert pause.details == {"reason": "required_context_overflow"}


def test_paused_workflow_uses_its_prompt_snapshot_after_active_prompt_replacement(
    session, session_factory, ready_project: NovelProject
) -> None:
    plan = BatchPlanDraft(
        chapters=[
            ChapterPlan(ordinal=1, title="第一章", goal="甲", ending_hook="甲")
        ]
    )
    provider = FakeProvider(
        [ProviderUnavailable("temporarily unavailable"), _response(plan, 2)]
    )
    workflow = WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )
    original_snapshot = next(
        snapshot
        for snapshot in PromptService(session).list_snapshots(workflow.id)
        if snapshot.role == "batch_planner"
    )
    orchestrator = _orchestrator(session_factory, provider)
    assert orchestrator.advance(workflow.id).status == "PAUSED_PROVIDER"

    replacement_body = "新的批次规划提示词，只应供后续工作流使用"
    prompt_service = PromptService(session)
    replacement = prompt_service.create_version(
        "batch_planner", replacement_body, "author"
    )
    prompt_service.activate(replacement.id)
    WorkflowService(session).resume(workflow.id)

    result = orchestrator.advance(workflow.id)

    assert result.status == "AWAITING_PLAN_APPROVAL"
    assert provider.requests[-1].system_prompt == original_snapshot.prompt_body
    assert provider.requests[-1].system_prompt != replacement_body


def test_reviewer_retrieves_only_deduplicated_bounded_evidence(
    session, session_factory, ready_project: NovelProject
) -> None:
    query = "evidenceanchor"
    long_source = "前" * 1_000 + f" {query} " + "后" * 1_000
    replacement = OutlineService(session).create_candidate(
        ready_project.id,
        [
            OutlineNodeInput(
                key="evidence",
                parent_key=None,
                kind="book",
                title="长篇证据纲要",
                order=0,
                payload={"detail": long_source},
            )
        ],
        reason="review evidence acceptance",
    )
    replacement = OutlineService(session).approve(replacement.id)
    plan = BatchPlanDraft(
        chapters=[
            ChapterPlan(
                ordinal=ordinal,
                title=f"第{ordinal}章",
                goal=chr(0x4E00 + ordinal),
                ending_hook=chr(0x4E00 + ordinal),
            )
            for ordinal in range(1, 6)
        ]
    )
    script = [_response(plan, 1)]
    candidate_bodies: list[str] = []
    call_number = 2
    for ordinal in range(1, 6):
        body = chr(0x4E00 + ordinal) * (4_499 + ordinal)
        candidate_bodies.append(body)
        script.extend(
            [
                _response(
                    ChapterDraft(title=f"第{ordinal}章", body=body), call_number
                ),
                _response(
                    ChapterSummaryDelta(
                        summary=f"候选摘要{ordinal}",
                        state_delta={"last_completed_ordinal": ordinal},
                    ),
                    call_number + 1,
                ),
            ]
        )
        call_number += 2
    script.extend(
        [
            _response(
                BatchReview(
                    passed=False,
                    issues=["需要核对大纲"],
                    evidence_queries=[f" {query} ", query],
                ),
                call_number,
            ),
            _response(
                BatchReview(passed=True, issues=[], evidence_queries=[]),
                call_number + 1,
            ),
        ]
    )
    provider = FakeProvider(script)
    workflow = WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS
    )
    orchestrator = _orchestrator(session_factory, provider)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    WorkflowService(session).approve_plan(workflow.id, "author")

    result = orchestrator.run_until_blocked(workflow.id)

    review_requests = [
        request
        for request in provider.requests
        if request.metadata["agent_role"] == "batch_reviewer"
    ]
    reviews = session.scalars(
        select(WorkflowArtifact)
        .where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "batch_review",
        )
        .order_by(WorkflowArtifact.created_at)
    ).all()
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert len(review_requests) == 2
    assert reviews[0].payload["evidence_queries"] == [f" {query} ", query]
    evidence = review_requests[1].input_payload["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["query"] == query
    assert evidence[0]["excerpt_start"] >= 0
    assert evidence[0]["excerpt_end"] > evidence[0]["excerpt_start"]
    assert evidence[0]["excerpt_end"] - evidence[0]["excerpt_start"] <= 1200
    assert len(evidence[0]["text"]) <= 1_200
    assert count_visible_characters(evidence[0]["text"]) <= 1_200
    canonical = session.scalar(
        select(ContextSource).where(
            ContextSource.project_id == ready_project.id,
            ContextSource.source_type == "outline_node",
            ContextSource.source_id == f"{replacement.id}:evidence",
            ContextSource.state_scope == "official",
        )
    )
    assert canonical is not None
    assert len(canonical.text) > 1_200
    assert evidence[0]["canonical_source_type"] == canonical.source_type
    assert evidence[0]["canonical_source_id"] == canonical.source_id
    assert evidence[0]["text"] == canonical.text[
        evidence[0]["excerpt_start"] : evidence[0]["excerpt_end"]
    ]
    for request in review_requests:
        serialized = json.dumps(request.input_payload, ensure_ascii=False)
        assert all(body not in serialized for body in candidate_bodies)


def test_default_app_constructs_without_network_provider_clients(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked_client(*_args, **_kwargs):
        raise AssertionError("network provider client must remain lazy")

    monkeypatch.delenv("AINOVEL_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AINOVEL_ALLOW_REAL_OPENAI", raising=False)
    monkeypatch.setattr(app_module.httpx, "Client", blocked_client)
    monkeypatch.setattr(app_module, "OpenAI", blocked_client)

    app = create_app(database_url)

    assert app.state.provider_registry.contains("fake")
    assert app.state.provider_registry.contains("ollama")
    assert app.state.provider_registry.contains("openai")

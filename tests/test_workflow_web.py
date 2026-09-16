from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

import ainovel.app as app_module
from ainovel.agents.contracts import BatchPlanDraft, ChapterPlan
from ainovel.agents.runner import AgentRunner
from ainovel.app import create_app
from ainovel.models import Base
from ainovel.models.batch import Chapter
from ainovel.models.prompt import WorkflowPromptSnapshot
from ainovel.models.project import NovelProject
from ainovel.models.workflow import (
    ChapterDraftRepair,
    GenerationWorkflow,
    ModelAttempt,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderDiagnostic,
)
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.fake import FakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.counting import count_visible_characters
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService
from ainovel.workflows.orchestrator import WorkflowOrchestrator


def csrf(client: TestClient, path: str) -> str:
    response = client.get(path)
    assert response.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    assert match is not None
    return unescape(match.group(1))


def response(
    structured: dict[str, object],
    number: int,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> ModelResponse:
    return ModelResponse(
        structured=structured,
        text=None,
        provider_response_id=f"web-response-{number}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=number,
    )


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
    return ready


@pytest.fixture
def fake_provider() -> FakeProvider:
    goal = "推进第一章目标"
    hook = "留下第一章悬念"
    required = f"{goal}。{hook}。"
    body = required + "甲" * (4500 - count_visible_characters(required))
    plan = BatchPlanDraft(
        chapters=[
            ChapterPlan(
                ordinal=1,
                title="第一章",
                goal=goal,
                ending_hook=hook,
            )
        ]
    )
    return FakeProvider(
        [
            response(plan.model_dump(mode="json"), 1, input_tokens=123, output_tokens=45),
            response({"title": "第一章", "body": body}, 2),
            response(
                {"summary": "第一章候选摘要", "state_delta": {"progress": 1}},
                3,
            ),
            response({"passed": True, "issues": [], "evidence_queries": []}, 4),
        ]
    )


@pytest.fixture
def fake_registry(fake_provider: FakeProvider) -> ProviderRegistry:
    return ProviderRegistry({"fake": lambda: fake_provider})


@pytest.fixture
def provider_registry(fake_registry: ProviderRegistry) -> ProviderRegistry:
    return fake_registry


@pytest.fixture
def workflow(session, ready_project: NovelProject) -> GenerationWorkflow:
    return WorkflowService(session).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS
    )


def start_workflow(client: TestClient, project_id: str) -> str:
    start = client.post(
        f"/projects/{project_id}/workflows",
        data={
            "provider_name": "fake",
            "model_name": "scripted",
            "requested_chapters": "1",
            "csrf_token": csrf(client, f"/projects/{project_id}"),
        },
        follow_redirects=False,
    )
    assert start.status_code == 303
    assert start.headers["location"].startswith("/workflows/")
    return start.headers["location"].rsplit("/", 1)[-1]


def post_workflow_action(
    client: TestClient, workflow_id: str, suffix: str, **data: str
):
    return client.post(
        f"/workflows/{workflow_id}/{suffix}",
        data={
            **data,
            "csrf_token": csrf(client, f"/workflows/{workflow_id}"),
        },
        follow_redirects=False,
    )


def test_workflow_uses_stage_specific_actions_and_empty_preview(client, workflow):
    page = client.get(f'/workflows/{workflow.id}')
    assert 'aria-label="创作进度"' in page.text
    assert '生成章节计划</button>' in page.text
    assert '尚未生成正文' in page.text
    post_workflow_action(client, workflow.id, 'run')
    page = client.get(f'/workflows/{workflow.id}')
    assert '确认计划（不会调用模型）</button>' in page.text
    post_workflow_action(client, workflow.id, 'plan/approve')
    page = client.get(f'/workflows/{workflow.id}')
    assert '生成正文</button>' in page.text


def test_cancelled_workflow_remains_accessible_from_project(client, session, workflow):
    workflow.status = 'CANCELLED'
    workflow.last_error_code = 'provider_protocol'
    project = session.get(NovelProject, workflow.project_id)
    project.active_workflow_id = None
    session.commit()
    page = client.get(f'/projects/{project.id}')
    assert f'href="/workflows/{workflow.id}"' in page.text
    assert '已取消' in page.text
    detail = client.get(f'/workflows/{workflow.id}')
    assert '已取消' in detail.text
    assert 'provider_protocol' in detail.text
    assert '未记录细分原因' in detail.text
    assert f'action="/workflows/{workflow.id}/run"' not in detail.text


class DetailsVisibility(HTMLParser):
    def __init__(self):
        super().__init__()
        self.closed_details = []
        self.candidate_hidden = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'details':
            self.closed_details.append('open' not in attributes)
        if attributes.get('aria-labelledby') == 'candidate-chapters-heading':
            self.candidate_hidden = any(self.closed_details)

    def handle_endtag(self, tag):
        if tag == 'details':
            self.closed_details.pop()


def test_workflow_body_is_not_hidden_inside_budget_details(client, workflow):
    empty = DetailsVisibility()
    empty.feed(client.get(f'/workflows/{workflow.id}').text)
    assert empty.closed_details == []
    post_workflow_action(client, workflow.id, 'run')
    post_workflow_action(client, workflow.id, 'plan/approve')
    post_workflow_action(client, workflow.id, 'run')
    generated = DetailsVisibility()
    generated.feed(client.get(f'/workflows/{workflow.id}').text)
    assert generated.candidate_hidden is False
    assert generated.closed_details == []


def test_v2_failed_work_draft_is_escaped_separate_and_marks_partial_usage(
    client: TestClient, session, workflow: GenerationWorkflow
) -> None:
    workflow.generation_version = 2
    workflow.status = "PAUSED_REVIEW"
    workflow.last_error_code = "provider_protocol"
    workflow.last_error_detail = "provider returned an invalid response"
    writing = WorkflowStep(
        id=str(uuid4()), workflow_id=workflow.id, kind="WRITING", ordinal=1,
        position=1, status="PAUSED", attempt_count=2, protocol_failure_count=1,
    )
    coverage = WorkflowStep(
        id=str(uuid4()), workflow_id=workflow.id, kind="VALIDATING_CHAPTER",
        ordinal=1, position=2, status="PENDING",
    )
    attempt = ModelAttempt(
        id=str(uuid4()), step_id=writing.id, attempt_number=2, status="FAILED",
        request_digest="d" * 64, provider_response_id=None,
        input_tokens=None, output_tokens=None, latency_ms=None,
        error_code="provider_protocol", error_detail="provider returned an invalid response",
    )
    session.add_all([writing, coverage])
    session.flush()
    session.add(attempt)
    session.flush()
    session.add(ChapterDraftRepair(
        id=str(uuid4()), workflow_id=workflow.id, writing_step_id=writing.id,
        latest_attempt_id=attempt.id,
        latest_payload={"title": "<b>未批准</b>", "body": "<script>alert('x')</script>"},
        visible_count=18, repair_count=2, repair_pending=False, draft_revision=3,
    ))
    session.commit()

    page = client.get(f"/workflows/{workflow.id}")
    assert page.status_code == 200
    assert "隔离的未批准工作稿" in page.text
    assert "修补轮次：2 / 2" in page.text
    assert "章节覆盖检查" in page.text
    assert "含未知项，合计不完整" in page.text
    assert "未记录细分原因" in page.text
    assert "&lt;script&gt;alert" in page.text
    assert "<script>alert" not in page.text


def create_ready_project(session_factory, title: str) -> str:
    with session_factory() as session:
        project = ProjectService(session).create(title, 2_000_000, 5_000_000)
        project_id = project.id
        outline = OutlineService(session).create_candidate(
            project_id,
            [
                OutlineNodeInput(
                    key="book",
                    parent_key=None,
                    kind="book",
                    title="全书总纲",
                    order=0,
                )
            ],
            reason="default registry web e2e",
        )
        OutlineService(session).approve(outline.id)
        ProjectService(session).add_constitution(
            project_id,
            {"genre": "offline demo", "voice": "close third"},
            author_approved=True,
        )
        return project_id


def run_default_fake_workflow(client: TestClient, project_id: str) -> str:
    created = client.post(
        f"/projects/{project_id}/workflows",
        data={
            "provider_name": "fake",
            "model_name": "demo",
            "requested_chapters": "5",
            "csrf_token": csrf(client, f"/projects/{project_id}"),
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    workflow_id = created.headers["location"].rsplit("/", 1)[-1]
    assert post_workflow_action(client, workflow_id, "run").status_code == 303
    assert (
        post_workflow_action(client, workflow_id, "plan/approve").status_code
        == 303
    )
    assert post_workflow_action(client, workflow_id, "run").status_code == 303
    return workflow_id


def test_default_registry_runs_two_fresh_offline_five_chapter_web_workflows(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_external_client(*_args, **_kwargs):
        raise AssertionError("external provider client must not be constructed")

    monkeypatch.setattr(app_module.httpx, "Client", forbid_external_client)
    monkeypatch.setattr(app_module, "OpenAI", forbid_external_client)
    app = create_app(database_url)
    Base.metadata.create_all(app.state.engine)
    with app.state.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS context_source_fts "
                "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
            )
        )

    first_project_id = create_ready_project(app.state.session_factory, "默认演示一")
    second_project_id = create_ready_project(app.state.session_factory, "默认演示二")
    first_registry_provider = app.state.provider_registry.get("fake")
    second_registry_provider = app.state.provider_registry.get("fake")
    assert isinstance(first_registry_provider, DemoFakeProvider)
    assert isinstance(second_registry_provider, DemoFakeProvider)
    assert first_registry_provider is not second_registry_provider

    with TestClient(app) as default_client:
        workflow_ids = [
            run_default_fake_workflow(default_client, first_project_id),
            run_default_fake_workflow(default_client, second_project_id),
        ]
        with app.state.session_factory() as session:
            workflows = [
                session.get(GenerationWorkflow, workflow_id)
                for workflow_id in workflow_ids
            ]
            assert all(workflow is not None for workflow in workflows)
            candidate_ids = [
                workflow.candidate_batch_id
                for workflow in workflows
                if workflow is not None
            ]
            assert len(candidate_ids) == 2
            assert all(candidate_id is not None for candidate_id in candidate_ids)
            assert all(
                workflow.status == "AWAITING_CONTENT_APPROVAL"
                for workflow in workflows
                if workflow is not None
            )
            for candidate_id in candidate_ids:
                chapters = session.scalars(
                    select(Chapter)
                    .where(Chapter.batch_id == candidate_id)
                    .order_by(Chapter.ordinal)
                ).all()
                assert [chapter.ordinal for chapter in chapters] == [1, 2, 3, 4, 5]
                assert all(chapter.visible_char_count == 4500 for chapter in chapters)

        orchestrator = app.state.orchestrator_factory()
        workflow_providers = [
            provider
            for (workflow_id, provider_name, _model_name), provider
            in orchestrator._providers.items()
            if workflow_id in workflow_ids and provider_name == "fake"
        ]
        assert len(workflow_providers) == 2
        assert workflow_providers[0] is not workflow_providers[1]


def test_start_workflow_redirects_without_trusting_form_project_id(
    client: TestClient,
    session,
    ready_project: NovelProject,
    fake_registry: ProviderRegistry,
) -> None:
    result = client.post(
        f"/projects/{ready_project.id}/workflows",
        data={
            "project_id": "attacker-controlled",
            "provider_name": "fake",
            "model_name": "scripted",
            "requested_chapters": "1",
            "csrf_token": csrf(client, f"/projects/{ready_project.id}"),
        },
        follow_redirects=False,
    )

    assert result.status_code == 303
    assert result.headers["location"].startswith("/workflows/")
    session.expire_all()
    persisted = session.get(NovelProject, ready_project.id)
    assert persisted is not None
    assert persisted.active_workflow_id == result.headers["location"].rsplit("/", 1)[-1]


@pytest.mark.parametrize(
    ("provider_name", "model_name", "requested_chapters", "message"),
    [
        ("missing", "scripted", "1", "Provider"),
        ("fake", " ", "1", "模型"),
        ("fake", "scripted", "0", "1 至 5"),
        ("fake", "scripted", "six", "整数"),
    ],
)
def test_start_validation_returns_422_without_taking_project_ownership(
    client: TestClient,
    session,
    ready_project: NovelProject,
    provider_name: str,
    model_name: str,
    requested_chapters: str,
    message: str,
) -> None:
    result = client.post(
        f"/projects/{ready_project.id}/workflows",
        data={
            "provider_name": provider_name,
            "model_name": model_name,
            "requested_chapters": requested_chapters,
            "csrf_token": csrf(client, f"/projects/{ready_project.id}"),
        },
    )

    assert result.status_code == 422
    assert message in result.text
    session.expire_all()
    project = session.get(NovelProject, ready_project.id)
    assert project is not None
    assert project.active_workflow_id is None


def test_start_checks_registry_membership_without_constructing_provider(
    client: TestClient,
    ready_project: NovelProject,
) -> None:
    calls: list[str] = []

    def broken_factory():
        calls.append("constructed")
        raise RuntimeError("Authorization: Bearer create-secret")

    client.app.state.provider_registry = ProviderRegistry({"fake": broken_factory})
    result = client.post(
        f"/projects/{ready_project.id}/workflows",
        data={
            "provider_name": "fake",
            "model_name": "scripted",
            "requested_chapters": "1",
            "csrf_token": csrf(client, f"/projects/{ready_project.id}"),
        },
        follow_redirects=False,
    )

    assert result.status_code == 303
    assert calls == []


class BrokenCapabilitiesProvider:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def capabilities(self, model: str) -> ProviderCapabilities:
        raise RuntimeError(f"X-Api-Key: {self.secret}")

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("generate must not run after capability failure")

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        return ProviderDiagnostic(False, "not used", ())


class BrokenGenerateProvider:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128_000, 16_000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError(f"Authorization: Bearer {self.secret}")

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        return ProviderDiagnostic(True, "not used", ())


class MalformedResponseProvider:
    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128_000, 16_000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            structured={
                "chapters": [
                    {
                        "ordinal": 1,
                        "title": "机密响应",
                        "goal": "不应保存",
                        "ending_hook": "不应显示",
                    }
                ]
            },
            text="Authorization: Bearer malformed-web-secret",
            provider_response_id="X-Api-Key: malformed-web-secret",
            input_tokens=-1,
            output_tokens=5,
            latency_ms=1,
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        return ProviderDiagnostic(True, "not used", ())


def install_orchestrator_registry(
    client: TestClient, registry: ProviderRegistry
) -> None:
    client.app.state.provider_registry = registry
    orchestrator = WorkflowOrchestrator(
        client.app.state.session_factory,
        registry,
        AgentRunner(),
        worker_id="web-provider-failure",
    )
    client.app.state.orchestrator_factory = lambda: orchestrator


@pytest.mark.parametrize("boundary", ["factory", "capabilities"])
def test_pre_attempt_provider_failure_pauses_safely_and_clears_lease(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
    boundary: str,
) -> None:
    secret = f"pre-attempt-{boundary}-secret"

    def factory():
        if boundary == "factory":
            raise RuntimeError(f"Authorization: Bearer {secret}")
        return BrokenCapabilitiesProvider(secret)

    install_orchestrator_registry(client, ProviderRegistry({"fake": factory}))
    result = post_workflow_action(client, workflow.id, "run")
    assert result.status_code == 303

    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert persisted is not None and step is not None
    assert persisted.status == "PAUSED_PROVIDER"
    assert persisted.last_error_code == "provider_unavailable"
    assert persisted.last_error_detail == "provider is unavailable"
    assert step.status == "PAUSED"
    assert step.lease_owner is None and step.lease_expires_at is None
    assert session.scalar(
        select(func.count())
        .select_from(ModelAttempt)
        .where(ModelAttempt.step_id == step.id)
    ) == 0
    page = client.get(f"/workflows/{workflow.id}")
    assert secret not in page.text
    assert "Authorization" not in page.text
    assert "Bearer" not in page.text


def test_post_attempt_untyped_provider_failure_records_safe_attempt_and_pause(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    secret = "post-attempt-generate-secret"
    provider = BrokenGenerateProvider(secret)
    install_orchestrator_registry(
        client,
        ProviderRegistry({"fake": lambda: provider}),
    )
    result = post_workflow_action(client, workflow.id, "run")
    assert result.status_code == 303

    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    attempt = session.scalar(
        select(ModelAttempt).where(ModelAttempt.step_id == step.id)
    )
    assert persisted is not None and step is not None and attempt is not None
    assert persisted.status == "PAUSED_PROVIDER"
    assert persisted.last_error_code == "provider_unavailable"
    assert persisted.last_error_detail == "provider is unavailable"
    assert step.status == "PAUSED"
    assert step.lease_owner is None and step.lease_expires_at is None
    assert step.attempt_count == 1
    assert attempt.status == "FAILED"
    assert attempt.error_code == "provider_unavailable"
    assert attempt.error_detail == "provider is unavailable"
    page = client.get(f"/workflows/{workflow.id}")
    assert secret not in page.text
    assert "Authorization" not in page.text
    assert "Bearer" not in page.text


def test_malformed_provider_response_fails_attempts_and_clears_lease_safely(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    provider = MalformedResponseProvider()
    install_orchestrator_registry(
        client,
        ProviderRegistry({"fake": lambda: provider}),
    )

    result = post_workflow_action(client, workflow.id, "run")

    assert result.status_code == 303
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    attempts = session.scalars(
        select(ModelAttempt)
        .where(ModelAttempt.step_id == step.id)
        .order_by(ModelAttempt.attempt_number)
    ).all()
    assert persisted is not None and step is not None
    assert persisted.status == "PAUSED_ATTEMPTS"
    assert persisted.last_error_code == "provider_protocol"
    assert persisted.last_error_detail == "provider returned an invalid response"
    assert step.status == "PAUSED"
    assert step.lease_owner is None and step.lease_expires_at is None
    assert step.attempt_count == 2
    assert [attempt.status for attempt in attempts] == ["FAILED", "FAILED"]
    assert all(attempt.error_code == "provider_protocol" for attempt in attempts)
    assert all(attempt.error_detail == "provider returned an invalid response" for attempt in attempts)
    assert all(attempt.provider_response_id is None for attempt in attempts)
    assert all(attempt.input_tokens is None for attempt in attempts)
    assert all(attempt.output_tokens is None for attempt in attempts)
    assert all(attempt.latency_ms is None for attempt in attempts)

    page = client.get(f"/workflows/{workflow.id}")
    assert page.status_code == 200
    assert "malformed-web-secret" not in page.text
    assert "Authorization" not in page.text
    assert "Bearer" not in page.text
    assert "X-Api-Key" not in page.text


def test_every_workflow_mutation_rejects_missing_csrf(
    client: TestClient, workflow: GenerationWorkflow
) -> None:
    paths = [
        f"/projects/{workflow.project_id}/workflows",
        f"/workflows/{workflow.id}/run",
        f"/workflows/{workflow.id}/plan/approve",
        f"/workflows/{workflow.id}/plan/reject",
        f"/workflows/{workflow.id}/resume",
        f"/projects/{workflow.project_id}/providers/diagnose",
    ]
    for path in paths:
        assert client.post(path, data={}).status_code == 403, path


def test_untrusted_host_cannot_reach_workflow_page(
    client: TestClient, workflow: GenerationWorkflow
) -> None:
    result = client.get(
        f"/workflows/{workflow.id}", headers={"host": "attacker.example"}
    )
    assert result.status_code == 400


def test_project_page_has_csrf_on_workflow_and_diagnostic_forms_then_links_owner(
    client: TestClient, ready_project: NovelProject
) -> None:
    path = f"/projects/{ready_project.id}"
    initial = client.get(path)
    token = csrf(client, path)
    assert f'action="{path}/workflows"' in initial.text
    assert f'action="{path}/providers/diagnose"' in initial.text
    assert initial.text.count('name="csrf_token"') == initial.text.count("<form ")
    assert token in initial.text

    workflow_id = start_workflow(client, ready_project.id)
    active = client.get(path)
    assert f'href="/workflows/{workflow_id}"' in active.text
    assert f'action="{path}/workflows"' not in active.text
    assert active.text.count('name="csrf_token"') == active.text.count("<form ")


def test_run_renders_plan_snapshots_steps_attempts_budgets_and_usage(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
    fake_provider: FakeProvider,
) -> None:
    run = post_workflow_action(client, workflow.id, "run")
    assert run.status_code == 303
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == "AWAITING_PLAN_APPROVAL"
    assert [request.metadata["agent_role"] for request in fake_provider.requests] == [
        "batch_planner"
    ]

    page = client.get(f"/workflows/{workflow.id}")
    for text in (
        "AWAITING_PLAN_APPROVAL",
        "第一章",
        "推进第一章目标",
        "留下第一章悬念",
        "16000",
        "4000",
        "123",
        "45",
        "batch_planner",
        "chapter_writer",
        "PLANNING",
        "web-response-1",
        "batch_plan",
    ):
        assert text in page.text

    approve = re.search(
        rf'<form action="/workflows/{workflow.id}/plan/approve"[^>]*>', page.text
    )
    reject = re.search(
        rf'<form action="/workflows/{workflow.id}/plan/reject"[^>]*>', page.text
    )
    assert approve is not None and "data-confirm" in approve.group(0)
    assert reject is not None and "data-confirm" in reject.group(0)
    assert page.text.count('name="csrf_token"') == page.text.count("<form ")


def test_workflow_page_renders_complete_immutable_prompt_snapshot_safely(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    snapshot = session.scalar(
        select(WorkflowPromptSnapshot).where(
            WorkflowPromptSnapshot.workflow_id == workflow.id,
            WorkflowPromptSnapshot.role == "batch_planner",
        )
    )
    assert snapshot is not None
    snapshot.prompt_body = "<prompt-body-sentinel>"
    snapshot.output_schema = {"description": "output-schema-sentinel"}
    snapshot.parameters = {"marker": "parameters-sentinel"}
    session.commit()

    page = client.get(f"/workflows/{workflow.id}")

    assert page.status_code == 200
    assert "&lt;prompt-body-sentinel&gt;" in page.text
    assert "<prompt-body-sentinel>" not in page.text
    assert "output-schema-sentinel" in page.text
    assert "parameters-sentinel" in page.text


def test_run_and_resume_forms_do_not_require_browser_confirmation(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    initial = client.get(f"/workflows/{workflow.id}").text
    run_form = re.search(
        rf'<form action="/workflows/{workflow.id}/run"[^>]*>', initial
    )
    assert run_form is not None and "data-confirm" not in run_form.group(0)

    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert step is not None
    workflow.status = "PAUSED_PROVIDER"
    workflow.last_error_code = "provider_unavailable"
    workflow.last_error_detail = "provider is unavailable"
    step.status = "PAUSED"
    session.commit()

    paused = client.get(f"/workflows/{workflow.id}").text
    resume_form = re.search(
        rf'<form action="/workflows/{workflow.id}/resume"[^>]*>', paused
    )
    assert resume_form is not None and "data-confirm" not in resume_form.group(0)
    assert re.search(r"<details[^>]*open", paused) is not None
    assert "provider_unavailable" in paused
    assert "provider is unavailable" in paused
    assert paused.count('name="csrf_token"') == paused.count("<form ")


def test_plan_approval_runs_to_candidate_gate_and_links_batch(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
    fake_provider: FakeProvider,
) -> None:
    assert post_workflow_action(client, workflow.id, "run").status_code == 303
    approve = post_workflow_action(
        client,
        workflow.id,
        "plan/approve",
        project_id="attacker-controlled",
    )
    assert approve.status_code == 303
    generate = post_workflow_action(client, workflow.id, "run")
    assert generate.status_code == 303

    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.status == "AWAITING_CONTENT_APPROVAL"
    assert persisted.candidate_batch_id is not None
    assert [request.metadata["agent_role"] for request in fake_provider.requests] == [
        "batch_planner",
        "chapter_writer",
        "chapter_summarizer",
        "batch_reviewer",
    ]
    page = client.get(f"/workflows/{workflow.id}")
    assert f'href="/projects/{workflow.project_id}#batch-{persisted.candidate_batch_id}"' in page.text


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "REJECTED"])
def test_terminal_workflow_page_keeps_candidate_batch_link(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
    terminal_status: str,
) -> None:
    assert post_workflow_action(client, workflow.id, "run").status_code == 303
    assert (
        post_workflow_action(client, workflow.id, "plan/approve").status_code
        == 303
    )
    assert post_workflow_action(client, workflow.id, "run").status_code == 303
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    assert persisted is not None
    assert persisted.candidate_batch_id is not None
    candidate_batch_id = persisted.candidate_batch_id
    persisted.status = terminal_status
    session.commit()

    page = client.get(f"/workflows/{workflow.id}")

    assert page.status_code == 200
    assert (
        f'href="/projects/{workflow.project_id}#batch-{candidate_batch_id}"'
        in page.text
    )


def test_workflow_page_renders_persisted_review_issues(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert step is not None
    session.add(
        WorkflowArtifact(
            id=str(uuid4()),
            workflow_id=workflow.id,
            step_id=step.id,
            kind="batch_review",
            ordinal=None,
            text_content=None,
            payload={
                "passed": False,
                "issues": ["时间线存在严重冲突"],
                "evidence_queries": [],
            },
            visible_char_count=None,
            content_hash="a" * 64,
        )
    )
    session.commit()

    page = client.get(f"/workflows/{workflow.id}")
    assert page.status_code == 200
    assert "审核问题" in page.text
    assert "时间线存在严重冲突" in page.text


def test_plan_rejection_requires_reason_and_uses_database_workflow_owner(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    assert post_workflow_action(client, workflow.id, "run").status_code == 303
    blank = post_workflow_action(client, workflow.id, "plan/reject", reason=" ")
    assert blank.status_code == 422
    rejected = post_workflow_action(
        client,
        workflow.id,
        "plan/reject",
        reason="请重新规划",
        project_id="attacker-controlled",
    )
    assert rejected.status_code == 303
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    project = session.get(NovelProject, workflow.project_id)
    assert persisted is not None and persisted.status == "REJECTED"
    assert project is not None and project.active_workflow_id is None


def test_resume_delegates_strict_whitelist_to_workflow_service(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert step is not None
    workflow.status = "PAUSED_PROVIDER"
    workflow.last_error_code = "provider_unavailable"
    workflow.last_error_detail = "provider is unavailable"
    step.status = "PAUSED"
    session.commit()

    resumed = post_workflow_action(client, workflow.id, "resume")
    assert resumed.status_code == 303
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    assert persisted is not None and persisted.status == "PLANNING"

    persisted.status = "PAUSED_REVIEW"
    step = session.get(WorkflowStep, step.id)
    assert step is not None
    step.status = "PAUSED"
    session.commit()
    refused = client.post(
        f"/workflows/{workflow.id}/resume",
        data={"csrf_token": csrf(client, f"/projects/{workflow.project_id}")},
        follow_redirects=False,
    )
    assert refused.status_code == 422
    session.expire_all()
    persisted = session.get(GenerationWorkflow, workflow.id)
    assert persisted is not None and persisted.status == "PAUSED_REVIEW"


def test_exhausted_paused_provider_does_not_offer_resume(
    client: TestClient,
    session,
    workflow: GenerationWorkflow,
) -> None:
    step = session.scalar(
        select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id)
    )
    assert step is not None
    workflow.status = "PAUSED_PROVIDER"
    workflow.last_error_code = "provider_unavailable"
    workflow.last_error_detail = "provider is unavailable"
    step.status = "PAUSED"
    step.attempt_count = 2
    session.commit()

    page = client.get(f"/workflows/{workflow.id}")
    assert page.status_code == 200
    assert f'action="/workflows/{workflow.id}/resume"' not in page.text
    assert WorkflowService(session).can_resume(workflow.id) is False

    refused = client.post(
        f"/workflows/{workflow.id}/resume",
        data={"csrf_token": csrf(client, f"/projects/{workflow.project_id}")},
        follow_redirects=False,
    )
    assert refused.status_code == 422


class DiagnosticOnlyProvider:
    def __init__(self, diagnostic: ProviderDiagnostic | Exception) -> None:
        self.diagnostic = diagnostic
        self.diagnose_calls: list[str | None] = []

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(16_000, 4_000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("diagnostics must never generate or download a model")

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        self.diagnose_calls.append(model)
        if isinstance(self.diagnostic, Exception):
            raise self.diagnostic
        return self.diagnostic


def test_provider_diagnostic_calls_only_diagnose_and_renders_safe_success(
    client: TestClient, ready_project: NovelProject
) -> None:
    provider = DiagnosticOnlyProvider(
        ProviderDiagnostic(True, "internal provider detail", ("local-model",))
    )
    client.app.state.provider_registry = ProviderRegistry({"fake": lambda: provider})
    path = f"/projects/{ready_project.id}"
    result = client.post(
        f"{path}/providers/diagnose",
        data={
            "provider_name": "fake",
            "model_name": "local-model",
            "csrf_token": csrf(client, path),
        },
    )

    assert result.status_code == 200
    assert "诊断成功" in result.text
    assert "local-model" in result.text
    assert "internal provider detail" not in result.text
    assert provider.diagnose_calls == ["local-model"]


def test_provider_diagnostic_redacts_raw_exception_and_headers(
    client: TestClient, ready_project: NovelProject
) -> None:
    provider = DiagnosticOnlyProvider(
        RuntimeError("Authorization: Bearer top-secret-key")
    )
    client.app.state.provider_registry = ProviderRegistry({"fake": lambda: provider})
    path = f"/projects/{ready_project.id}"
    result = client.post(
        f"{path}/providers/diagnose",
        data={
            "provider_name": "fake",
            "model_name": "local-model",
            "csrf_token": csrf(client, path),
        },
    )

    assert result.status_code == 200
    assert "诊断失败" in result.text
    assert "top-secret-key" not in result.text
    assert "Authorization" not in result.text
    assert "Bearer" not in result.text


def test_provider_diagnostic_redacts_provider_factory_exception(
    client: TestClient, ready_project: NovelProject
) -> None:
    def broken_factory():
        raise RuntimeError("X-Api-Key: factory-top-secret")

    client.app.state.provider_registry = ProviderRegistry({"fake": broken_factory})
    path = f"/projects/{ready_project.id}"
    result = client.post(
        f"{path}/providers/diagnose",
        data={
            "provider_name": "fake",
            "model_name": "local-model",
            "csrf_token": csrf(client, path),
        },
    )

    assert result.status_code == 200
    assert "诊断失败" in result.text
    assert "factory-top-secret" not in result.text
    assert "X-Api-Key" not in result.text


def test_default_registry_is_lazy_and_uses_fresh_fake_instances(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ainovel.app as app_module

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("provider clients must be lazy at app creation")

    monkeypatch.setattr(app_module.httpx, "Client", forbidden_constructor)
    monkeypatch.setattr(app_module, "OpenAI", forbidden_constructor)
    app = app_module.create_app(database_url)
    registry = app.state.provider_registry
    assert registry.get("fake") is not registry.get("fake")


def test_default_registry_injects_conservative_provider_capability_ceilings(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ainovel.app import create_app

    monkeypatch.setenv("AINOVEL_ALLOW_REAL_OPENAI", "false")
    monkeypatch.delenv("AINOVEL_OPENAI_API_KEY", raising=False)
    app = create_app(database_url)
    ollama = app.state.provider_registry.get("ollama")
    openai = app.state.provider_registry.get("openai")
    for provider in (ollama, openai):
        capabilities = provider.capabilities("configured-model")
        assert capabilities.context_window == 16_000
        assert capabilities.max_output_tokens == 4_000
    assert openai.capabilities("configured-model").real_calls_allowed is False


def test_default_ollama_client_ignores_proxy_environment(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ainovel.app as app_module

    captured: dict[str, object] = {}

    def client_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://proxy.invalid:8080")
    monkeypatch.setattr(app_module.httpx, "Client", client_constructor)
    app = app_module.create_app(database_url)

    app.state.provider_registry.get("ollama")

    assert captured == {"timeout": 120.0, "trust_env": False}


def test_openai_default_provider_requires_both_opt_in_and_key(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ainovel.app import create_app

    monkeypatch.setenv("AINOVEL_ALLOW_REAL_OPENAI", "true")
    monkeypatch.delenv("AINOVEL_OPENAI_API_KEY", raising=False)
    disabled = create_app(database_url).state.provider_registry.get("openai")
    assert disabled.capabilities("configured-model").real_calls_allowed is False

    monkeypatch.setenv("AINOVEL_OPENAI_API_KEY", "test-secret-value")
    enabled = create_app(database_url).state.provider_registry.get("openai")
    assert enabled.capabilities("configured-model").real_calls_allowed is True


def test_app_rejects_non_loopback_ollama_configuration(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ainovel.app import create_app

    monkeypatch.setenv("AINOVEL_OLLAMA_BASE_URL", "https://provider.example")
    with pytest.raises(ValueError, match="loopback"):
        create_app(database_url)

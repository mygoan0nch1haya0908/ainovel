from __future__ import annotations

from html import unescape
import re

import pytest
from fastapi.testclient import TestClient

from ainovel.models.project import NovelProject
from ainovel.models.stage import StageRoadmapVersion, StageWorkflowNode, StoryStage
from ainovel.models.workflow import GenerationWorkflow
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.projects import ProjectService
from ainovel.services.stages import StageService


class RecordingDemoProvider(DemoFakeProvider):
    def __init__(self) -> None:
        self.roles: list[str | None] = []

    def generate(self, request):
        self.roles.append(request.metadata.get("agent_role"))
        return super().generate(request)


@pytest.fixture
def stage_provider() -> RecordingDemoProvider:
    return RecordingDemoProvider()


@pytest.fixture
def provider_registry(stage_provider: RecordingDemoProvider) -> ProviderRegistry:
    return ProviderRegistry({"fake": lambda: stage_provider})


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


def csrf(client: TestClient, path: str) -> str:
    page = client.get(path)
    assert page.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match is not None
    return unescape(match.group(1))


def post(
    client: TestClient,
    path: str,
    page_path: str,
    data: dict[str, str],
):
    return client.post(
        path,
        data={**data, "csrf_token": csrf(client, page_path)},
        follow_redirects=False,
    )


def test_author_creates_reviews_and_starts_five_chapter_stage_without_hidden_calls(
    client: TestClient,
    session,
    ready_project: NovelProject,
    stage_provider: RecordingDemoProvider,
) -> None:
    project_path = f"/projects/{ready_project.id}"
    setup = client.get(project_path)
    assert "剧情阶段总体架构" in setup.text

    created = post(
        client,
        f"{project_path}/stages",
        project_path,
        {
            "architecture": "主角入城，调查失踪案并找到幕后主使。",
            "author_confirm": "yes",
        },
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/stages/")
    stage_id = created.headers["location"].rsplit("/", 1)[-1]
    assert stage_provider.roles == []

    stage_path = f"/stages/{stage_id}"
    proposed = post(
        client,
        f"{stage_path}/roadmaps",
        stage_path,
        {
            "provider_name": "fake",
            "model_name": "demo",
            "author_confirm": "yes",
        },
    )
    assert proposed.status_code == 303
    assert stage_provider.roles == []

    session.expire_all()
    roadmap = session.query(StageRoadmapVersion).filter_by(stage_id=stage_id).one()
    assert roadmap.status == "PENDING"
    generated = post(
        client,
        f"{stage_path}/roadmaps/{roadmap.id}/generate",
        stage_path,
        {"model_call_confirm": "yes"},
    )
    assert generated.status_code == 303
    assert stage_provider.roles == ["stage_planner"]

    preview = client.get(stage_path)
    assert preview.status_code == 200
    assert "预计 7 章" in preview.text
    assert "演示第1章" in preview.text
    assert "阶段第 1 章" in preview.text
    assert "已确认 0 / 7 章" in preview.text
    assert stage_provider.roles == ["stage_planner"]

    approved = post(
        client,
        f"{stage_path}/roadmaps/{roadmap.id}/approve",
        stage_path,
        {"approval_confirm": "yes"},
    )
    assert approved.status_code == 303
    assert stage_provider.roles == ["stage_planner"]

    started = post(
        client,
        f"{stage_path}/batches",
        stage_path,
        {"requested_chapters": "5", "author_confirm": "yes"},
    )
    assert started.status_code == 303
    assert started.headers["location"].startswith("/workflows/")
    assert stage_provider.roles == ["stage_planner"]

    workflow_id = started.headers["location"].rsplit("/", 1)[-1]
    session.expire_all()
    workflow = session.get(GenerationWorkflow, workflow_id)
    stage = session.get(StoryStage, stage_id)
    nodes = (
        session.query(StageWorkflowNode)
        .filter_by(workflow_id=workflow_id)
        .order_by(StageWorkflowNode.ordinal)
        .all()
    )
    assert workflow is not None and workflow.generation_version == 2
    assert workflow.requested_chapters == 5
    assert stage is not None and stage.confirmed_chapters == 0
    assert [(node.stage_ordinal, node.book_ordinal) for node in nodes] == [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]

    blocked = client.get(stage_path)
    assert "不能提前启动下一批" in blocked.text
    assert f'action="{stage_path}/batches"' not in blocked.text

    planned = post(client, f"/workflows/{workflow_id}/run", f"/workflows/{workflow_id}", {})
    assert planned.status_code == 303
    plan_page = client.get(f"/workflows/{workflow_id}")
    assert "阶段第 1 章" in plan_page.text
    assert "全书第 1 章" in plan_page.text
    assert "场景 1" in plan_page.text
    assert "目标可见字符：5200" in plan_page.text
    assert "章节覆盖检查" in plan_page.text

    calls_before_approval = len(stage_provider.roles)
    plan_approved = post(
        client,
        f"/workflows/{workflow_id}/plan/approve",
        f"/workflows/{workflow_id}",
        {},
    )
    assert plan_approved.status_code == 303
    assert len(stage_provider.roles) == calls_before_approval

    generated_batch = post(
        client,
        f"/workflows/{workflow_id}/run",
        f"/workflows/{workflow_id}",
        {},
    )
    assert generated_batch.status_code == 303
    session.expire_all()
    workflow = session.get(GenerationWorkflow, workflow_id)
    assert workflow is not None and workflow.candidate_batch_id is not None
    batch_id = workflow.candidate_batch_id

    assert post(client, f"/batches/{batch_id}/approve", f"/batches/{batch_id}", {}).status_code == 303
    assert post(client, f"/workflows/{workflow_id}/reconcile", f"/workflows/{workflow_id}", {}).status_code == 303

    advanced = client.get(stage_path)
    assert "已确认 5 / 7 章" in advanced.text
    assert f'action="{stage_path}/batches"' in advanced.text
    second = post(
        client,
        f"{stage_path}/batches",
        stage_path,
        {"requested_chapters": "5", "author_confirm": "yes"},
    )
    assert second.status_code == 303
    second_id = second.headers["location"].rsplit("/", 1)[-1]
    session.expire_all()
    second_workflow = session.get(GenerationWorkflow, second_id)
    second_nodes = (
        session.query(StageWorkflowNode)
        .filter_by(workflow_id=second_id)
        .order_by(StageWorkflowNode.ordinal)
        .all()
    )
    assert second_workflow is not None and second_workflow.requested_chapters == 2
    assert [(node.stage_ordinal, node.book_ordinal) for node in second_nodes] == [(6, 6), (7, 7)]


def test_stage_mutations_require_csrf_and_explicit_confirmation(
    client: TestClient,
    session,
    ready_project: NovelProject,
    stage_provider: RecordingDemoProvider,
) -> None:
    path = f"/projects/{ready_project.id}"
    missing_csrf = client.post(
        f"{path}/stages",
        data={"architecture": "查案", "author_confirm": "yes"},
    )
    assert missing_csrf.status_code == 403
    missing_confirmation = post(
        client,
        f"{path}/stages",
        path,
        {"architecture": "查案"},
    )
    assert missing_confirmation.status_code == 422
    assert session.query(StoryStage).count() == 0
    assert stage_provider.roles == []


def test_stage_get_is_read_only_and_each_later_action_requires_confirmation(
    client: TestClient,
    session,
    ready_project: NovelProject,
    stage_provider: RecordingDemoProvider,
) -> None:
    service = StageService(session)
    stage = service.create(ready_project.id, "查案", "author")
    first = service.propose_roadmap(stage.id, "author", "fake", "demo")
    first = service.generate_roadmap(first.id, stage_provider)
    assert first.status == "PROPOSED"
    stage_provider.roles.clear()

    snapshot = (
        service.get(stage.id).revision,
        [(item.id, item.status, item.attempts_used) for item in service.list_roadmaps(stage.id)],
        session.query(GenerationWorkflow).count(),
    )
    page = client.get(f"/stages/{stage.id}")
    assert page.status_code == 200
    session.expire_all()
    assert (
        service.get(stage.id).revision,
        [(item.id, item.status, item.attempts_used) for item in service.list_roadmaps(stage.id)],
        session.query(GenerationWorkflow).count(),
    ) == snapshot
    assert stage_provider.roles == []

    stage_path = f"/stages/{stage.id}"
    no_approve = post(
        client,
        f"{stage_path}/roadmaps/{first.id}/approve",
        stage_path,
        {},
    )
    assert no_approve.status_code == 422
    session.expire_all()
    assert service.get(stage.id).approved_roadmap_id is None

    assert post(
        client,
        f"{stage_path}/roadmaps/{first.id}/approve",
        stage_path,
        {"approval_confirm": "yes"},
    ).status_code == 303
    no_batch = post(
        client,
        f"{stage_path}/batches",
        stage_path,
        {"requested_chapters": "5"},
    )
    assert no_batch.status_code == 422
    assert session.query(GenerationWorkflow).count() == 0

    second = StageService(session).propose_roadmap(
        stage.id, "author", "fake", "demo"
    )
    no_generation = post(
        client,
        f"{stage_path}/roadmaps/{second.id}/generate",
        stage_path,
        {},
    )
    assert no_generation.status_code == 422
    session.expire_all()
    assert StageService(session).roadmap(second.id).status == "PENDING"
    assert stage_provider.roles == []

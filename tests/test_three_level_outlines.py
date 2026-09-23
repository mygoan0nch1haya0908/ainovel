import json
import re
from copy import deepcopy

import pytest
from sqlalchemy import func, select

from test_chapter_test import (
    chapter_test_module, chapter_test_app, chapter_client, bounded_provider,
    form_tokens, valid_setup_data,
)
from test_stages import roadmap_payload
from test_orchestrator import response, clock, make_orchestrator
from ainovel.models import (
    NovelProject, OutlineNode, OutlineVersion, ConstitutionVersion,
    ModelAttempt, StoryStage, StageModelAttempt, WorkflowStep,
)
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.fake import FakeProvider
from ainovel.services.stages import StageService
from ainovel.services.workflows import WorkflowService
from ainovel.services.batches import BatchService


BOOK = "全书调查失踪王朝，最终公开真相"
STAGE = "第一阶段调查邮局，找到失踪名单"
HINT = "首章专属：先烧掉蓝色假信再进入邮局"


class RecordingDemoProvider(DemoFakeProvider):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


def hierarchy_data(client, hint=""):
    return {**valid_setup_data(client), "setup_mode": "hierarchical",
            "book_outline": BOOK, "stage_architecture": STAGE,
            "chapter_outline": hint, "chapter_title": "", "chapter_goal": "",
            "chapter_hook": ""}


def create_hierarchy(client, hint=""):
    result = client.post("/chapter-test", data=hierarchy_data(client, hint), follow_redirects=False)
    assert result.status_code == 303
    assert result.headers["location"].startswith("/stages/")
    return result.headers["location"].rsplit("/", 1)[-1]


def test_default_form_requires_book_and_stage_but_not_first_chapter(chapter_client):
    page = chapter_client.get("/chapter-test")
    selected = re.search(r'<input[^>]*value="hierarchical"[^>]*>', page.text)
    assert selected is not None and "checked" in selected.group()
    for name in ("book_outline", "stage_architecture"):
        field = re.search(rf'<textarea[^>]*name="{name}"[^>]*>', page.text)
        assert field is not None and "required" in field.group()
    chapter = re.search(r'<textarea[^>]*name="chapter_outline"[^>]*>', page.text)
    assert "required" not in chapter.group()
    assert "总剧情大纲" in page.text and "阶段大纲" in page.text
    assert "单章大纲（可选，当前阶段第一章）" in page.text


@pytest.mark.parametrize("hint", ["", HINT])
def test_setup_saves_distinct_hierarchy_without_model_calls(chapter_client, chapter_test_app, hint):
    stage_id = create_hierarchy(chapter_client, hint)
    with chapter_test_app.state.session_factory() as session:
        stage = session.get(StoryStage, stage_id)
        nodes = {n.stable_key: n for n in session.scalars(select(OutlineNode))}
        assert nodes["book"].payload == {"book_outline": BOOK}
        assert nodes["stage-1"].parent_key == "book"
        assert nodes["stage-1"].payload == {"stage_architecture": STAGE}
        assert nodes["provisional-ending"].parent_key == "book"
        assert stage.architecture == STAGE
        if hint:
            assert nodes["chapter-1"].parent_key == "stage-1"
            assert nodes["chapter-1"].payload == {"chapter_outline": HINT, "stage_ordinal": 1}
        else:
            assert "chapter-1" not in nodes
        for model in (ModelAttempt, StageModelAttempt):
            assert session.scalar(select(func.count()).select_from(model)) == 0
    page = chapter_client.get(f"/stages/{stage_id}")
    assert BOOK in page.text and STAGE in page.text
    if hint:
        assert HINT in page.text


@pytest.mark.parametrize("field,value", [("book_outline", " "), ("stage_architecture", " "), ("book_outline", "大" * 12001)], ids=["missing-book", "missing-stage", "oversize-book"])
def test_hierarchy_validation_is_authoritative(chapter_client, chapter_test_app, field, value):
    data = hierarchy_data(chapter_client)
    data[field] = value
    result = chapter_client.post("/chapter-test", data=data)
    assert result.status_code == 422
    expected = "不能超过" if len(value) > 12000 else "不能为空"
    assert expected in result.text
    with chapter_test_app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(NovelProject)) == 0


def test_hierarchy_reuse_preserves_levels_escaped_and_does_not_mutate(chapter_client, chapter_test_app):
    hint = "<script>首章专属</script>"
    stage_id = create_hierarchy(chapter_client, hint)
    with chapter_test_app.state.session_factory() as session:
        stage = session.get(StoryStage, stage_id)
        project_id = stage.project_id
        before = [(n.stable_key, deepcopy(n.payload)) for n in session.scalars(select(OutlineNode))]
    result = chapter_client.post(f"/chapter-test/reuse/{project_id}", data={
        "csrf_token": form_tokens(chapter_client)["csrf_token"], "reuse_confirm": "yes"})
    assert result.status_code == 200
    assert BOOK in result.text and STAGE in result.text
    assert "&lt;script&gt;首章专属&lt;/script&gt;" in result.text and hint not in result.text
    assert 'value="hierarchical" checked' in result.text
    with chapter_test_app.state.session_factory() as session:
        assert before == [(n.stable_key, n.payload) for n in session.scalars(select(OutlineNode))]
        assert session.scalar(select(func.count()).select_from(StoryStage)) == 1
        assert session.scalar(select(func.count()).select_from(StageModelAttempt)) == 0


def test_hierarchical_setup_rolls_back_after_stage_creation(chapter_client, chapter_test_app, monkeypatch):
    original = StageService.create
    def fail_after_create(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ValueError("injected late failure")
    monkeypatch.setattr(StageService, "create", fail_after_create)
    result = chapter_client.post("/chapter-test", data=hierarchy_data(chapter_client, HINT))
    assert result.status_code == 500
    with chapter_test_app.state.session_factory() as session:
        for model in (NovelProject, ConstitutionVersion, OutlineVersion, OutlineNode, StoryStage):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_frozen_hierarchy_reaches_planner_and_only_first_node_across_batches(chapter_client, chapter_test_app, clock):
    stage_id = create_hierarchy(chapter_client, HINT)
    factory = chapter_test_app.state.session_factory
    with factory() as session:
        service = StageService(session)
        stage = service.get(stage_id)
        version = service.propose_roadmap(stage_id, "author", "fake", "demo")
        frozen = deepcopy(version.input_snapshot)
        provider = FakeProvider([response(roadmap_payload(3), 1)])
        assert service.generate_roadmap(version.id, provider).status == "PROPOSED"
        assert provider.requests[0].input_payload["outline_hierarchy"]["book_outline"] == BOOK
        assert provider.requests[0].input_payload["outline_hierarchy"]["stage_architecture"] == STAGE
        assert provider.requests[0].input_payload["outline_hierarchy"]["first_chapter"]["chapter_outline"] == HINT
        service.approve_roadmap(stage_id, version.id, "author")
        batch_provider = RecordingDemoProvider()
        orchestrator = make_orchestrator(factory, batch_provider, clock)
        for count, first_ordinal in ((2, 1), (1, 3)):
            started = service.start_next_batch(stage_id, "author", "fake", "demo", count)
            step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == started.workflow.id))
            request = orchestrator._build_request(step)
            text = json.dumps(request.input_payload, ensure_ascii=False)
            assert BOOK in text and STAGE in text
            assert (HINT in text) == (first_ordinal == 1)
            nodes = request.input_payload["stage"]["nodes"]
            assert ("author_chapter_outline" in nodes[0]) == (first_ordinal == 1)
            if count == 2:
                assert "author_chapter_outline" not in nodes[1]
            assert orchestrator.advance(started.workflow.id).status == "AWAITING_PLAN_APPROVAL"
            WorkflowService(session, clock=clock).approve_plan(started.workflow.id, "author")
            batch_provider.requests.clear()
            result = orchestrator.run_until_blocked(started.workflow.id, max_steps=50)
            assert result.status == "AWAITING_CONTENT_APPROVAL"
            writers = [r for r in batch_provider.requests if r.metadata.get("agent_role") == "chapter_writer"]
            assert len(writers) == count
            for ordinal, request in enumerate(writers, 1):
                text = json.dumps(request.input_payload, ensure_ascii=False)
                assert BOOK in text and STAGE in text
                assert (HINT in text) == (first_ordinal == 1 and ordinal == 1)
            BatchService(session).approve(result.candidate_batch_id, stage.base_outline_version_id)
            WorkflowService(session, clock=clock).reconcile_batch_decision(started.workflow.id)
        assert service.roadmap(version.id).input_snapshot == frozen


def test_hierarchy_overflow_pauses_without_truncating_or_dispatching(chapter_client, chapter_test_app):
    data = hierarchy_data(chapter_client)
    data.update(book_outline="全" * 12000, stage_architecture="阶" * 12000)
    created = chapter_client.post("/chapter-test", data=data, follow_redirects=False)
    assert created.status_code == 303
    with chapter_test_app.state.session_factory() as session:
        service = StageService(session)
        version = service.propose_roadmap(created.headers["location"].rsplit("/", 1)[-1], "author", "fake", "demo")
        provider = FakeProvider([])
        result = service.generate_roadmap(version.id, provider)
        assert result.status == "PAUSED_CONTEXT_OVERFLOW"
        assert result.input_snapshot["outline_hierarchy"]["book_outline"] == "全" * 12000
        assert result.input_snapshot["outline_hierarchy"]["stage_architecture"] == "阶" * 12000
        assert result.attempts_used == 0 and provider.requests == []


def test_legacy_stage_reuse_does_not_invent_whole_book_outline(chapter_client, chapter_test_app):
    data = valid_setup_data(chapter_client)
    data.update(setup_mode="stage", stage_architecture=STAGE, chapter_outline="")
    created = chapter_client.post("/chapter-test", data=data, follow_redirects=False)
    assert created.status_code == 303
    with chapter_test_app.state.session_factory() as session:
        stage = session.get(StoryStage, created.headers["location"].rsplit("/", 1)[-1])
        project_id = stage.project_id
        assert StageService(session).outline_context(stage.id) is None
    result = chapter_client.post(f"/chapter-test/reuse/{project_id}", data={
        "csrf_token": form_tokens(chapter_client)["csrf_token"], "reuse_confirm": "yes"})
    assert result.status_code == 200
    assert re.search(r'<textarea[^>]*name="book_outline"[^>]*></textarea>', result.text)
    assert STAGE in result.text and 'value="stage" checked' in result.text

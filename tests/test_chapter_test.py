from __future__ import annotations

from html import unescape
import importlib.util
import logging
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

import ainovel.app as app_module
from ainovel.app import create_app
from ainovel.models import (
    Base,
    Chapter,
    ConstitutionVersion,
    GenerationWorkflow,
    ModelAttempt,
    NovelProject,
    OutlineNode,
    OutlineVersion,
    WritingBatch,
)
from ainovel.providers.contracts import ModelResponse, ProviderCapabilities
from ainovel.providers.fake import FakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.counting import count_visible_characters
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService
from ainovel.workflows.orchestrator import WorkflowOrchestrator


MODULE_NAME = "ainovel.chapter_test"
EXPECTED_DEFAULT_DATABASE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers"
    / "runtime"
    / "chapter-test"
    / "chapter-test.db"
)


def test_chapter_test_factory_module_exists() -> None:
    # A missing dedicated module would silently leave authors with only the
    # ordinary multi-chapter application.
    assert importlib.util.find_spec(MODULE_NAME) is not None


@pytest.fixture
def chapter_test_module():
    return pytest.importorskip(MODULE_NAME)


class BoundedFakeProvider(FakeProvider):
    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(32_000, 12_000, True, True, True, False)


def provider_script() -> list[ModelResponse]:
    goal = "主角必须当面交出密信"
    hook = "钟声响起时城门突然关闭"
    required = f"{goal}。{hook}。"
    unsafe_fragment = "<script>alert('candidate')</script>"
    prefix = required + unsafe_fragment
    body = prefix + "甲" * (4_500 - count_visible_characters(prefix))
    assert count_visible_characters(body) == 4_500
    return [
        ModelResponse(
            structured={
                "chapters": [
                    {
                        "ordinal": 1,
                        "title": "密信之夜",
                        "goal": goal,
                        "ending_hook": hook,
                    }
                ]
            },
            text=None,
            provider_response_id="chapter-test-plan",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1,
        ),
        ModelResponse(
            structured={"title": "密信之夜", "body": body},
            text=None,
            provider_response_id="chapter-test-draft",
            input_tokens=1_000,
            output_tokens=3_000,
            latency_ms=2,
        ),
        ModelResponse(
            structured={"summary": "密信已交付，城门关闭。", "state_delta": {"gate": "closed"}},
            text=None,
            provider_response_id="chapter-test-summary",
            input_tokens=500,
            output_tokens=100,
            latency_ms=3,
        ),
        ModelResponse(
            structured={"passed": True, "issues": [], "evidence_queries": []},
            text=None,
            provider_response_id="chapter-test-review",
            input_tokens=600,
            output_tokens=80,
            latency_ms=4,
        ),
    ]


@pytest.fixture
def bounded_provider() -> BoundedFakeProvider:
    return BoundedFakeProvider(provider_script())


def create_schema(app) -> None:
    Base.metadata.create_all(app.state.engine)
    with app.state.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS context_source_fts "
                "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
            )
        )


@pytest.fixture
def chapter_test_app(chapter_test_module, database_url: str, bounded_provider):
    registry = ProviderRegistry({"qwen": lambda: bounded_provider})
    app = chapter_test_module.create_chapter_test_app(
        database_url=database_url,
        provider_registry=registry,
    )
    create_schema(app)
    return app


@pytest.fixture
def chapter_client(chapter_test_app):
    with TestClient(chapter_test_app) as test_client:
        yield test_client


def form_tokens(client: TestClient, path: str = "/chapter-test") -> dict[str, str]:
    page = client.get(path)
    assert page.status_code == 200
    csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    submission_match = re.search(
        r'name="submission_token" value="([^"]+)"', page.text
    )
    assert csrf_match is not None
    assert submission_match is not None
    return {
        "csrf_token": unescape(csrf_match.group(1)),
        "submission_token": unescape(submission_match.group(1)),
    }


def valid_setup_data(client: TestClient) -> dict[str, str]:
    return {
        **form_tokens(client),
        "project_title": "雾城来信",
        "setting_style": "近未来山城；第三人称限知；克制、悬疑。",
        "provisional_ending": "主角最终公开密信，但失去原来的身份。",
        "chapter_outline": "主角潜入旧邮局取得密信，穿过封锁线交给记者。",
        "chapter_title": "密信之夜",
        "chapter_goal": "交出密信",
        "chapter_hook": "城门关闭",
        "model_name": "qwen-flash",
        "author_confirm": "yes",
    }


def post_action(client: TestClient, workflow_id: str, suffix: str):
    page = client.get(f"/workflows/{workflow_id}")
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert token is not None
    return client.post(
        f"/workflows/{workflow_id}/{suffix}",
        data={"csrf_token": unescape(token.group(1))},
        follow_redirects=False,
    )


def test_default_app_has_no_chapter_test_route(database_url: str) -> None:
    app = create_app(database_url)
    with TestClient(app) as client:
        assert client.get("/chapter-test").status_code == 404


def test_default_database_is_isolated_from_normal_database_setting(
    chapter_test_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    normal_database = tmp_path / "normal.db"
    monkeypatch.setenv(
        "AINOVEL_DATABASE_URL", f"sqlite+pysqlite:///{normal_database.as_posix()}"
    )

    app = chapter_test_module.create_chapter_test_app()

    assert Path(app.state.engine.url.database).resolve() == EXPECTED_DEFAULT_DATABASE.resolve()
    assert Path(app.state.engine.url.database).resolve() != normal_database.resolve()


def test_dedicated_qwen_has_larger_caps_while_ordinary_factory_is_unchanged(
    chapter_test_module,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            self.api_key = kwargs.get("api_key")
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setattr(app_module, "OpenAI", Client)
    monkeypatch.setattr(app_module.httpx, "Client", lambda **_kwargs: object())
    ordinary_registry = create_app(database_url).state.provider_registry
    dedicated_registry = chapter_test_module.create_chapter_test_app(
        database_url=database_url
    ).state.provider_registry
    ordinary = ordinary_registry.get("qwen")
    dedicated = dedicated_registry.get("qwen")

    assert ordinary.capabilities("qwen-flash").context_window == 16_000
    assert ordinary.capabilities("qwen-flash").max_output_tokens == 4_000
    assert dedicated.capabilities("qwen-flash").context_window == 32_000
    assert dedicated.capabilities("qwen-flash").max_output_tokens == 12_000
    for provider_name in ("ollama", "openai"):
        capabilities = dedicated_registry.get(provider_name).capabilities("configured")
        assert capabilities.context_window == 16_000
        assert capabilities.max_output_tokens == 4_000


@pytest.mark.parametrize("timeout", [True, 0, -1, 300, 301, float("inf")])
def test_orchestrator_rejects_request_timeout_outside_claim_lease(
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="request timeout"):
        WorkflowOrchestrator(
            lambda: pytest.fail("constructor must not open a session"),
            ProviderRegistry({}),
            SimpleNamespace(),
            request_timeout_seconds=timeout,
        )


def test_setup_to_candidate_flow_keeps_both_author_gates_and_readable_body(
    chapter_client: TestClient,
    chapter_test_app,
    bounded_provider: BoundedFakeProvider,
) -> None:
    setup_page = chapter_client.get("/chapter-test")
    assert "单章写作测试" in setup_page.text
    assert "可能产生 API 费用" in setup_page.text
    assert "qwen-flash" in setup_page.text
    assert bounded_provider.requests == []

    created = chapter_client.post(
        "/chapter-test", data=valid_setup_data(chapter_client), follow_redirects=False
    )

    assert created.status_code == 303, created.text
    assert created.headers["location"].startswith("/workflows/")
    assert bounded_provider.requests == []
    workflow_id = created.headers["location"].rsplit("/", 1)[-1]

    with chapter_test_app.state.session_factory() as session:
        project = session.scalar(select(NovelProject))
        workflow = session.get(GenerationWorkflow, workflow_id)
        constitution = session.scalar(select(ConstitutionVersion))
        outline = session.scalar(select(OutlineVersion))
        nodes = session.scalars(select(OutlineNode).order_by(OutlineNode.order)).all()
        assert project is not None and workflow is not None
        assert constitution is not None and outline is not None
        assert project.target_chars_min == 2_000_000
        assert project.target_chars_max == 5_000_000
        assert constitution.author_approved is True
        assert constitution.content["setting_style"] == "近未来山城；第三人称限知；克制、悬疑。"
        assert outline.status == "official"
        ending_nodes = [node for node in nodes if node.kind == "provisional_ending"]
        assert len(ending_nodes) == 1
        assert ending_nodes[0].payload["text"] == "主角最终公开密信，但失去原来的身份。"
        assert any(
            node.payload.get("chapter_outline")
            == "主角潜入旧邮局取得密信，穿过封锁线交给记者。"
            for node in nodes
        )
        assert workflow.provider_name == "qwen"
        assert workflow.model_name == "qwen-flash"
        assert workflow.requested_chapters == 1
        assert workflow.status == "PLANNING"
        assert workflow.planner_output_tokens == 4_000
        assert workflow.writer_output_tokens == 12_000
        assert workflow.summarizer_output_tokens == 4_000
        assert workflow.reviewer_output_tokens == 4_000

    confirmed = chapter_client.get(f"/chapter-test?project_id={project.id}")
    assert "近未来山城；第三人称限知；克制、悬疑。" in confirmed.text
    assert "主角最终公开密信，但失去原来的身份。" in confirmed.text
    assert "主角潜入旧邮局取得密信，穿过封锁线交给记者。" in confirmed.text

    assert post_action(chapter_client, workflow_id, "run").status_code == 303
    assert len(bounded_provider.requests) == 1
    plan_page = chapter_client.get(f"/workflows/{workflow_id}")
    assert "AWAITING_PLAN_APPROVAL" in plan_page.text
    assert "主角必须当面交出密信" in plan_page.text

    assert post_action(chapter_client, workflow_id, "plan/approve").status_code == 303
    assert len(bounded_provider.requests) == 1
    assert post_action(chapter_client, workflow_id, "run").status_code == 303
    assert len(bounded_provider.requests) == 4

    roles = [request.metadata["agent_role"] for request in bounded_provider.requests]
    assert roles == [
        "batch_planner",
        "chapter_writer",
        "chapter_summarizer",
        "batch_reviewer",
    ]
    planner_context = bounded_provider.requests[0].input_payload["context_packet"]
    assert any(
        item["source_type"] == "provisional_ending"
        and "主角最终公开密信，但失去原来的身份。" in item["text"]
        for item in planner_context["items"]
    )
    assert any(
        item["source_type"] == "current_stage_goal"
        and "主角潜入旧邮局取得密信，穿过封锁线交给记者。" in item["text"]
        for item in planner_context["items"]
    )
    assert [request.max_output_tokens for request in bounded_provider.requests] == [
        4_000,
        12_000,
        4_000,
        4_000,
    ]
    assert all(request.timeout_seconds == 180.0 for request in bounded_provider.requests)
    assert all(
        request.max_input_tokens + request.max_output_tokens + 1_024 <= 32_000
        for request in bounded_provider.requests
    )
    writer_request = bounded_provider.requests[1]
    assert writer_request.max_input_tokens == 18_976

    candidate_page = chapter_client.get(f"/workflows/{workflow_id}")
    assert "AWAITING_CONTENT_APPROVAL" in candidate_page.text
    assert "候选章节正文" in candidate_page.text
    assert "密信之夜" in candidate_page.text
    assert "可见字符：4500" in candidate_page.text
    assert "&lt;script&gt;alert(&#39;candidate&#39;)&lt;/script&gt;" in candidate_page.text
    assert "<script>alert('candidate')</script>" not in candidate_page.text
    assert "这是独立测试数据库中的候选内容，不是已发布正文" in candidate_page.text
    assert 'href="/chapter-test' in candidate_page.text

    with chapter_test_app.state.session_factory() as session:
        workflow = session.get(GenerationWorkflow, workflow_id)
        assert workflow is not None and workflow.candidate_batch_id is not None
        batch = session.get(WritingBatch, workflow.candidate_batch_id)
        chapters = session.scalars(
            select(Chapter).where(Chapter.batch_id == workflow.candidate_batch_id)
        ).all()
        assert batch is not None and batch.status == "ready_for_review"
        assert len(chapters) == 1
        assert chapters[0].status == "candidate"
        assert chapters[0].official_chapter_number is None
        assert session.scalar(
            select(func.count())
            .select_from(Chapter)
            .where(Chapter.official_chapter_number.is_not(None))
        ) == 0


@pytest.mark.parametrize(
    ("changes", "expected_message"),
    [
        ({"author_confirm": None}, "请勾选作者确认"),
        ({"provisional_ending": "   "}, "暂定结局不能为空"),
        ({"chapter_outline": "大" * 12_001}, "章节提纲不能超过"),
    ],
)
def test_invalid_setup_redisplays_escaped_input_without_writes_or_calls(
    chapter_client: TestClient,
    chapter_test_app,
    bounded_provider: BoundedFakeProvider,
    changes: dict[str, str | None],
    expected_message: str,
) -> None:
    data = valid_setup_data(chapter_client)
    data["project_title"] = "<img src=x onerror=alert('setup')>"
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value

    result = chapter_client.post("/chapter-test", data=data)

    assert result.status_code == 422
    assert expected_message in result.text
    assert "&lt;img src=x onerror=alert" in result.text
    assert "<img src=x onerror=alert('setup')>" not in result.text
    assert bounded_provider.requests == []
    with chapter_test_app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(NovelProject)) == 0
        assert session.scalar(select(func.count()).select_from(GenerationWorkflow)) == 0


def test_setup_rejects_missing_csrf_without_writes_or_calls(
    chapter_client: TestClient,
    chapter_test_app,
    bounded_provider: BoundedFakeProvider,
) -> None:
    result = chapter_client.post(
        "/chapter-test",
        data={
            "project_title": "无 CSRF",
            "setting_style": "设定",
            "provisional_ending": "结局",
            "chapter_outline": "提纲",
            "model_name": "qwen-flash",
            "author_confirm": "yes",
        },
    )

    assert result.status_code == 403
    assert bounded_provider.requests == []
    with chapter_test_app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(NovelProject)) == 0


def test_duplicate_valid_submission_creates_only_one_workflow(
    chapter_client: TestClient,
    chapter_test_app,
    bounded_provider: BoundedFakeProvider,
) -> None:
    data = valid_setup_data(chapter_client)

    first = chapter_client.post("/chapter-test", data=data, follow_redirects=False)
    duplicate = chapter_client.post("/chapter-test", data=data, follow_redirects=False)

    assert first.status_code == 303
    assert duplicate.status_code == 409
    assert bounded_provider.requests == []
    with chapter_test_app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(NovelProject)) == 1
        assert session.scalar(select(func.count()).select_from(GenerationWorkflow)) == 1


def test_workflow_start_failure_rolls_back_entire_setup(
    chapter_test_module,
    chapter_client: TestClient,
    chapter_test_app,
    bounded_provider: BoundedFakeProvider,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import ainovel.web.chapter_test_routes as route_module

    original_start = route_module.WorkflowService.start
    sensitive_input = "不得进入日志的作者私密设定"
    sensitive_error = "Authorization: Bearer diagnostic-secret"
    data = valid_setup_data(chapter_client)
    data["setting_style"] = sensitive_input

    def fail_after_start(self, *args, **kwargs):
        original_start(self, *args, **kwargs)
        raise ValueError(f"{sensitive_error}; input={sensitive_input}")

    monkeypatch.setattr(route_module.WorkflowService, "start", fail_after_start)
    monkeypatch.setattr(route_module, "token_urlsafe", lambda _size: "setup-event-123")

    with caplog.at_level(logging.ERROR, logger=route_module.__name__):
        result = chapter_client.post("/chapter-test", data=data)

    assert result.status_code == 500
    assert "测试项目创建失败" in result.text
    assert "setup-event-123" in result.text
    assert sensitive_error not in result.text
    assert sensitive_input not in result.text
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "chapter_test_setup_failed event_id=setup-event-123 "
        "stage=workflow_start exception_type=ValueError"
    ]
    assert "Traceback" not in caplog.text
    assert sensitive_error not in caplog.text
    assert sensitive_input not in caplog.text
    assert bounded_provider.requests == []
    with chapter_test_app.state.session_factory() as session:
        for model in (
            NovelProject,
            ConstitutionVersion,
            OutlineVersion,
            OutlineNode,
            GenerationWorkflow,
        ):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_busy_setup_releases_connection_without_partial_state(
    chapter_test_module,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "busy.db"
    database_url = (
        f"sqlite+pysqlite:///{database_path.as_posix()}?timeout=0.01"
    )
    provider = BoundedFakeProvider(provider_script())
    app = chapter_test_module.create_chapter_test_app(
        database_url=database_url,
        provider_registry=ProviderRegistry({"qwen": lambda: provider}),
    )
    create_schema(app)

    with TestClient(app) as client:
        data = valid_setup_data(client)
        locker = app.state.engine.connect()
        locker_transaction = locker.begin()
        locker.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            result = client.post("/chapter-test", data=data)
        finally:
            locker_transaction.rollback()
            locker.close()

        assert result.status_code == 500
        assert app.state.engine.pool.checkedout() == 0
        assert provider.requests == []
        with app.state.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(NovelProject)) == 0


def atomic_setup_values() -> dict[str, str]:
    return {
        "project_title": "cleanup test",
        "setting_style": "style",
        "provisional_ending": "ending",
        "chapter_outline": "outline",
        "chapter_title": "title",
        "chapter_goal": "goal",
        "chapter_hook": "hook",
        "model_name": "qwen-flash",
        "author_confirm": "yes",
    }


def install_atomic_setup_fakes(
    route_module,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workflow_error: Exception | None = None,
    cleanup_errors: set[str] | None = None,
) -> tuple[SimpleNamespace, list[str]]:
    events: list[str] = []
    failures = cleanup_errors or set()

    class FakeTransaction:
        is_active = True

        def commit(self) -> None:
            events.append("transaction_commit")
            self.is_active = False

        def rollback(self) -> None:
            events.append("transaction_rollback")
            self.is_active = False
            if "transaction_rollback" in failures:
                raise RuntimeError("rollback Authorization: Bearer cleanup-secret")

    transaction = FakeTransaction()

    class FakeConnection:
        dialect = SimpleNamespace(name="postgresql")

        def begin(self):
            events.append("transaction_begin")
            return transaction

        def close(self) -> None:
            events.append("connection_close")
            if "connection_close" in failures:
                raise RuntimeError("close X-Api-Key: cleanup-secret")

    connection = FakeConnection()

    class FakeEngine:
        def connect(self):
            events.append("connect")
            return connection

    class FakeSession:
        def close(self) -> None:
            events.append("session_close")
            if "session_close" in failures:
                raise RuntimeError("session author-private-input")

    class FakeProjectService:
        def __init__(self, _session) -> None:
            pass

        def create(self, *_args):
            return SimpleNamespace(id="project-id")

        def add_constitution(self, *_args, **_kwargs) -> None:
            pass

    class FakeOutlineService:
        def __init__(self, _session) -> None:
            pass

        def create_candidate(self, *_args, **_kwargs):
            return SimpleNamespace(id="outline-id")

        def approve(self, _outline_id: str) -> None:
            pass

    class FakeWorkflowService:
        def __init__(self, _session) -> None:
            pass

        def start(self, *_args):
            if workflow_error is not None:
                raise workflow_error
            return SimpleNamespace(id="workflow-id")

    monkeypatch.setattr(route_module, "Session", lambda **_kwargs: FakeSession())
    monkeypatch.setattr(route_module, "ProjectService", FakeProjectService)
    monkeypatch.setattr(route_module, "OutlineService", FakeOutlineService)
    monkeypatch.setattr(route_module, "WorkflowService", FakeWorkflowService)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine=FakeEngine(),
                workflow_budgets=route_module.WorkflowBudgets(),
            )
        )
    )
    return request, events


def test_cleanup_failures_do_not_replace_primary_stage_and_all_cleanup_is_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ainovel.web.chapter_test_routes as route_module

    request, events = install_atomic_setup_fakes(
        route_module,
        monkeypatch,
        workflow_error=ValueError("Authorization: Bearer primary-secret"),
        cleanup_errors={"session_close", "transaction_rollback", "connection_close"},
    )

    with pytest.raises(route_module.ChapterTestSetupFailure) as raised:
        route_module._create_workflow_atomically(request, atomic_setup_values())

    assert raised.value.stage == "workflow_start"
    assert raised.value.exception_type == "ValueError"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert events[-3:] == [
        "session_close",
        "transaction_rollback",
        "connection_close",
    ]
    assert "secret" not in str(raised.value)


def test_standalone_connection_cleanup_failure_uses_safe_fixed_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ainovel.web.chapter_test_routes as route_module

    request, events = install_atomic_setup_fakes(
        route_module,
        monkeypatch,
        cleanup_errors={"connection_close"},
    )

    with pytest.raises(route_module.ChapterTestSetupFailure) as raised:
        route_module._create_workflow_atomically(request, atomic_setup_values())

    assert raised.value.stage == "connection_cleanup"
    assert raised.value.exception_type == "RuntimeError"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert events[-3:] == ["session_close", "transaction_commit", "connection_close"]
    assert "cleanup-secret" not in str(raised.value)


def test_test_app_existing_workflow_route_rejects_more_than_one_chapter(
    chapter_client: TestClient,
    chapter_test_app,
) -> None:
    with chapter_test_app.state.session_factory() as session:
        project = ProjectService(session).create("第二项目", 2_000_000, 5_000_000)
        ProjectService(session).add_constitution(
            project.id, {"setting_style": "测试"}, author_approved=True
        )
        outline = OutlineService(session).create_candidate(
            project.id,
            [
                OutlineNodeInput(
                    key="book",
                    parent_key=None,
                    kind="book",
                    title="单章",
                    order=0,
                )
            ],
            reason="test",
        )
        OutlineService(session).approve(outline.id)
        project_id = project.id

    project_page = chapter_client.get(f"/projects/{project_id}")
    token = re.search(r'name="csrf_token" value="([^"]+)"', project_page.text)
    assert token is not None
    result = chapter_client.post(
        f"/projects/{project_id}/workflows",
        data={
            "provider_name": "qwen",
            "model_name": "qwen-flash",
            "requested_chapters": "2",
            "csrf_token": unescape(token.group(1)),
        },
    )

    assert result.status_code == 422
    assert "单章测试仅允许生成 1 章" in result.text
    with chapter_test_app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(GenerationWorkflow)) == 0


def test_setup_page_never_renders_qwen_key(
    chapter_test_module,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "chapter-test-secret-that-must-not-render"
    monkeypatch.setenv("AINOVEL_QWEN_API_KEY", secret)
    app = chapter_test_module.create_chapter_test_app(database_url=database_url)
    create_schema(app)

    with TestClient(app) as client:
        page = client.get("/chapter-test")

    assert page.status_code == 200
    assert secret not in page.text

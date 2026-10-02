from __future__ import annotations

from types import SimpleNamespace

import pytest

from ainovel.agents.runner import AgentRunner
import ainovel.app as app_module
from ainovel.config import Settings
from ainovel.models import GenerationWorkflow
from ainovel.providers.contracts import ModelResponse
from ainovel.providers.fake import FakeProvider
from ainovel.providers.qwen import QwenProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.projects import ProjectService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService
from ainovel.workflows.orchestrator import WorkflowOrchestrator


def test_settings_prefers_ainovel_qwen_key_and_falls_back_to_dashscope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-fallback")
    monkeypatch.delenv("AINOVEL_QWEN_API_KEY", raising=False)
    fallback = Settings()
    assert fallback.qwen_api_key is not None
    assert fallback.qwen_api_key.get_secret_value() == "dashscope-fallback"

    monkeypatch.setenv("AINOVEL_QWEN_API_KEY", "ainovel-preferred")
    preferred = Settings()
    assert preferred.qwen_api_key is not None
    assert preferred.qwen_api_key.get_secret_value() == "ainovel-preferred"

    explicit = Settings(qwen_api_key="programmatic")
    assert explicit.qwen_api_key is not None
    assert explicit.qwen_api_key.get_secret_value() == "programmatic"


def test_qwen_opt_in_is_independent_from_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AINOVEL_QWEN_API_KEY", "qwen-test-key")
    monkeypatch.setenv("AINOVEL_ALLOW_REAL_QWEN", "true")
    monkeypatch.setenv("AINOVEL_ALLOW_REAL_OPENAI", "false")

    settings = Settings()

    assert settings.allow_real_qwen is True
    assert settings.allow_real_openai is False
    assert settings.qwen_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_default_registry_constructs_qwen_lazily_with_bounded_sdk_options(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[dict[str, object]] = []

    class Client:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(dict(kwargs))
            self.api_key = kwargs.get("api_key")
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setenv("AINOVEL_QWEN_API_KEY", "qwen-test-key")
    monkeypatch.setenv("AINOVEL_ALLOW_REAL_QWEN", "true")
    monkeypatch.setattr(app_module, "OpenAI", Client)

    app = app_module.create_app(database_url)
    assert constructed == []
    assert app.state.provider_registry.contains("qwen") is True

    provider = app.state.provider_registry.get("qwen")

    assert isinstance(provider, QwenProvider)
    assert constructed == [
        {
            "api_key": "qwen-test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "timeout": 120.0,
            "max_retries": 0,
        }
    ]
    capabilities = provider.capabilities("qwen-flash")
    assert capabilities.context_window == 16_000
    assert capabilities.max_output_tokens == 4_000
    assert capabilities.real_calls_allowed is True


@pytest.mark.parametrize(
    ("allow_real", "api_key", "detail_fragment", "real_calls_allowed"),
    [
        (False, "qwen-test-key", "disabled", False),
        (True, None, "API key is not configured", False),
        (True, "qwen-test-key", "configuration is ready", True),
    ],
)
def test_default_qwen_diagnosis_preserves_opt_in_and_key_presence_states(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    allow_real: bool,
    api_key: str | None,
    detail_fragment: str,
    real_calls_allowed: bool,
) -> None:
    class Client:
        def __init__(self, **kwargs: object) -> None:
            self.api_key = kwargs.get("api_key")
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: pytest.fail(
                        "configuration diagnosis must not call the network"
                    )
                )
            )

    monkeypatch.setenv("AINOVEL_ALLOW_REAL_QWEN", str(allow_real).lower())
    monkeypatch.delenv("AINOVEL_QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    if api_key is not None:
        monkeypatch.setenv("AINOVEL_QWEN_API_KEY", api_key)
    monkeypatch.setattr(app_module, "OpenAI", Client)

    provider = app_module.create_app(database_url).state.provider_registry.get("qwen")
    diagnostic = provider.diagnose("qwen-flash")

    assert diagnostic.available is real_calls_allowed
    assert detail_fragment in diagnostic.detail
    assert "not an online connectivity check" in diagnostic.detail
    assert (
        provider.capabilities("qwen-flash").real_calls_allowed
        is real_calls_allowed
    )


def test_workflow_accepts_qwen_and_retains_author_plan_approval_gate(
    client, session, project, official_outline
) -> None:
    ProjectService(session).add_constitution(
        project.id,
        {"genre": "historical fantasy", "voice": "close third"},
        author_approved=True,
    )
    provider = FakeProvider(
        [
            ModelResponse(
                structured={
                    "chapters": [
                        {
                            "ordinal": 1,
                            "title": "入局",
                            "goal": "主角接下委托",
                            "ending_hook": "发现追踪者",
                        }
                    ]
                },
                text=None,
                provider_response_id="synthetic-qwen",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1,
            )
        ]
    )
    workflow = WorkflowService(session).start(
        project.id, "qwen", "qwen-flash", 1, DEFAULT_BUDGETS
    )
    orchestrator = WorkflowOrchestrator(
        client.app.state.session_factory,
        ProviderRegistry({"qwen": lambda: provider}),
        AgentRunner(),
    )

    planned = orchestrator.advance(workflow.id)
    unchanged = orchestrator.advance(workflow.id)

    assert planned.status == "AWAITING_PLAN_APPROVAL"
    assert unchanged.status == "AWAITING_PLAN_APPROVAL"
    assert len(provider.requests) == 1
    approved = WorkflowService(session).approve_plan(workflow.id, "author")
    assert approved.status == "GENERATING_CHAPTERS"


def test_qwen_is_exported_and_accepted_by_provider_name_type() -> None:
    from ainovel.providers import QwenProvider as ExportedQwenProvider
    from ainovel.providers.registry import ProviderName

    name: ProviderName = "qwen"
    assert name == "qwen"
    assert ExportedQwenProvider is QwenProvider

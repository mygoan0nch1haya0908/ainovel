from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx
import pytest

from ainovel.agents.contracts import ChapterSummaryDelta
from ainovel.agents.runner import AgentRunner
from ainovel.config import Settings
from ainovel.providers.contracts import ModelRequest
from ainovel.providers.ollama import OllamaProvider


def test_ollama_live_helper_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def client_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://proxy.invalid:8080")
    monkeypatch.setenv("AINOVEL_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(httpx, "Client", client_constructor)

    provider = ollama_provider_from_settings()

    assert isinstance(provider, OllamaProvider)
    assert captured["base_url"] == "http://127.0.0.1:11434"
    assert captured["trust_env"] is False


@pytest.mark.parametrize(
    ("run_live", "model", "message"),
    [
        (None, None, "AINOVEL_RUN_OLLAMA_TESTS=1"),
        ("1", None, "AINOVEL_OLLAMA_MODEL"),
    ],
)
def test_ollama_live_opt_in_gates_skip_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    run_live: str | None,
    model: str | None,
    message: str,
) -> None:
    def forbidden_client(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Ollama client must not be constructed before opt-in")

    if run_live is None:
        monkeypatch.delenv("AINOVEL_RUN_OLLAMA_TESTS", raising=False)
    else:
        monkeypatch.setenv("AINOVEL_RUN_OLLAMA_TESTS", run_live)
    if model is None:
        monkeypatch.delenv("AINOVEL_OLLAMA_MODEL", raising=False)
    else:
        monkeypatch.setenv("AINOVEL_OLLAMA_MODEL", model)
    monkeypatch.setattr(httpx, "Client", forbidden_client)

    with pytest.raises(pytest.skip.Exception, match=message):
        test_configured_ollama_model_returns_structured_output()


def ollama_provider_from_settings() -> OllamaProvider:
    settings = Settings()
    parsed = urlsplit(settings.ollama_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama live tests require an HTTP 127.0.0.1 loopback URL")
    client = httpx.Client(
        base_url=settings.ollama_base_url,
        timeout=settings.provider_timeout_seconds,
        trust_env=False,
    )
    return OllamaProvider(client, settings.ollama_base_url)


def live_summary_request(model: str) -> ModelRequest:
    return ModelRequest(
        model=model,
        system_prompt=(
            "Return only a structured chapter summary delta. "
            "Do not include hidden reasoning."
        ),
        input_payload={
            "chapter": {"title": "测试章", "body": "测试正文。"},
            "chapter_plan": {
                "ordinal": 1,
                "title": "测试章",
                "goal": "概括已发生的事件",
                "ending_hook": "记录当前状态",
            },
        },
        output_schema=ChapterSummaryDelta.model_json_schema(),
        max_input_tokens=10_976,
        max_output_tokens=4_000,
        timeout_seconds=Settings().provider_timeout_seconds,
        metadata={
            "agent_role": "chapter_summarizer",
            "schema_name": "chapter_summary",
        },
    )


@pytest.mark.local_model
def test_configured_ollama_model_returns_structured_output() -> None:
    if os.getenv("AINOVEL_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set AINOVEL_RUN_OLLAMA_TESTS=1 to run local model tests")
    model = os.getenv("AINOVEL_OLLAMA_MODEL")
    if not model:
        pytest.skip("set AINOVEL_OLLAMA_MODEL to an installed model")

    provider = ollama_provider_from_settings()
    diagnostic = provider.diagnose(model)
    assert diagnostic.available
    assert model in diagnostic.models
    result = AgentRunner().run(
        provider, live_summary_request(model), ChapterSummaryDelta
    )
    assert result.summary.strip()

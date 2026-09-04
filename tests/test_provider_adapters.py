from __future__ import annotations

import json
import traceback
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ainovel.agents.contracts import ChapterSummaryDelta
from ainovel.providers.contracts import (
    ModelRequest,
    ProviderAuthenticationError,
    ProviderProtocolError,
    ProviderTimeout,
)
from ainovel.providers.ollama import OllamaProvider
from ainovel.providers.openai import OpenAIProvider
from ainovel.providers.registry import ProviderRegistry


def make_summary_request() -> ModelRequest:
    return ModelRequest(
        model="test-model",
        system_prompt="Return a chapter summary.",
        input_payload={"chapter": "第一章"},
        output_schema=ChapterSummaryDelta.model_json_schema(),
        max_input_tokens=16_000,
        max_output_tokens=4_000,
        timeout_seconds=5.0,
        metadata={"schema_name": "chapter_summary"},
    )


def ollama_provider_with_transport(
    payload: dict[str, object],
) -> tuple[OllamaProvider, dict[str, object]]:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        if request.content:
            captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=payload, request=request)

    return OllamaProvider(httpx.Client(transport=httpx.MockTransport(handler)), "http://ollama.test"), captured


class FakeResponses:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeOpenAIClient:
    def __init__(self, result: object, api_key: str | None = "test-openai-secret") -> None:
        self.api_key = api_key
        self.responses = FakeResponses(result)


@pytest.fixture
def fake_openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient(
        SimpleNamespace(
            output_text='{"summary":"有效","state_delta":{"chapter":1}}',
            id="response-1",
            usage=SimpleNamespace(input_tokens=31, output_tokens=12),
        )
    )


def test_ollama_posts_schema_and_parses_usage() -> None:
    request = make_summary_request()
    provider, captured = ollama_provider_with_transport(
        {
            "message": {"content": '{"summary":"有效"}'},
            "prompt_eval_count": 21,
            "eval_count": 9,
        }
    )

    response = provider.generate(request)

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/chat"
    assert captured["format"] == request.output_schema
    assert response.structured == {"summary": "有效"}
    assert response.input_tokens == 21
    assert response.output_tokens == 9


def test_ollama_timeout_is_classified_without_leaking_response_secret(caplog: pytest.LogCaptureFixture) -> None:
    secret = "ollama-test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    provider = OllamaProvider(httpx.Client(transport=httpx.MockTransport(handler)), "http://ollama.test")

    with pytest.raises(ProviderTimeout) as error:
        provider.generate(make_summary_request())

    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert secret not in "".join(traceback.format_exception(error.type, error.value, error.tb))


def test_ollama_rejects_malformed_json_without_leaking_content(caplog: pytest.LogCaptureFixture) -> None:
    secret = "ollama-test-secret"
    provider, _ = ollama_provider_with_transport({"message": {"content": secret}})

    with pytest.raises(ProviderProtocolError) as error:
        provider.generate(make_summary_request())

    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_ollama_diagnose_lists_models_in_sorted_order() -> None:
    provider, _ = ollama_provider_with_transport({"models": [{"name": "zeta"}, {"name": "alpha"}]})

    diagnostic = provider.diagnose()

    assert diagnostic.available is True
    assert diagnostic.models == ("alpha", "zeta")


def test_openai_is_disabled_without_explicit_opt_in(fake_openai_client: FakeOpenAIClient) -> None:
    provider = OpenAIProvider(fake_openai_client, allow_real_calls=False)

    with pytest.raises(ProviderAuthenticationError, match="disabled"):
        provider.generate(make_summary_request())

    assert fake_openai_client.responses.calls == []


def test_openai_rejects_missing_api_key_without_making_a_call() -> None:
    client = FakeOpenAIClient(SimpleNamespace(), api_key=None)
    provider = OpenAIProvider(client, allow_real_calls=True)

    with pytest.raises(ProviderAuthenticationError, match="API key"):
        provider.generate(make_summary_request())

    assert client.responses.calls == []


def test_openai_parses_output_text_and_maps_usage(fake_openai_client: FakeOpenAIClient) -> None:
    request = make_summary_request()
    provider = OpenAIProvider(fake_openai_client, allow_real_calls=True)

    response = provider.generate(request)

    assert response.structured == {"summary": "有效", "state_delta": {"chapter": 1}}
    assert response.provider_response_id == "response-1"
    assert response.input_tokens == 31
    assert response.output_tokens == 12
    assert fake_openai_client.responses.calls == [
        {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": json.dumps(request.input_payload, ensure_ascii=False),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "chapter_summary",
                    "strict": True,
                    "schema": request.output_schema,
                }
            },
            "max_output_tokens": request.max_output_tokens,
        }
    ]


def test_openai_supports_protocol_fake_without_an_api_key_attribute() -> None:
    result = SimpleNamespace(
        output_text='{"summary":"有效","state_delta":{}}',
        id="response-3",
        usage=None,
    )
    client = SimpleNamespace(responses=FakeResponses(result))

    response = OpenAIProvider(client, allow_real_calls=True).generate(make_summary_request())

    assert response.structured == {"summary": "有效", "state_delta": {}}


def test_openai_rejects_malformed_output_without_leaking_secret(caplog: pytest.LogCaptureFixture) -> None:
    secret = "openai-test-secret"
    client = FakeOpenAIClient(SimpleNamespace(output_text=secret, id="response-2", usage=None))
    provider = OpenAIProvider(client, allow_real_calls=True)

    with pytest.raises(ProviderProtocolError) as error:
        provider.generate(make_summary_request())

    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_registry_rejects_unknown_provider_name(fake_openai_client: FakeOpenAIClient) -> None:
    registry = ProviderRegistry(
        {
            "fake": lambda: fake_openai_client,
            "ollama": lambda: fake_openai_client,
            "openai": lambda: fake_openai_client,
        }
    )

    with pytest.raises(ProviderProtocolError, match="unknown provider"):
        registry.get("not-a-provider")  # type: ignore[arg-type]

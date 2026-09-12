from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any

import httpx
import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from ainovel.agents.contracts import ChapterSummaryDelta
from ainovel.agents.runner import AgentRunner
from ainovel.context import ConservativeEstimator
from ainovel.providers.contracts import (
    ModelRequest,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
)
from ainovel.providers.qwen import QwenProvider


def summary_request() -> ModelRequest:
    return ModelRequest(
        model="qwen-flash",
        system_prompt="Return a chapter summary.",
        input_payload={"chapter": "第一章"},
        output_schema=ChapterSummaryDelta.model_json_schema(),
        max_input_tokens=10_976,
        max_output_tokens=256,
        timeout_seconds=5.0,
        metadata={"schema_name": "chapter_summary"},
    )


def chat_response(
    content: object = '{"summary":"有效","state_delta":{"chapter":1}}',
    *,
    finish_reason: object = "stop",
    refusal: object = None,
    choices: object | None = None,
) -> object:
    if choices is None:
        choices = [
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
    return SimpleNamespace(
        id="chatcmpl-qwen-1",
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=37, completion_tokens=14),
    )


class CapturingCompletions:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeQwenClient:
    def __init__(self, result: object, api_key: str | None = "test-qwen-secret") -> None:
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=CapturingCompletions(result))


def test_qwen_chat_wire_includes_json_object_mode_schema_and_bounded_input() -> None:
    request = summary_request()
    client = FakeQwenClient(chat_response())

    response = QwenProvider(client, allow_real_calls=True).generate(request)

    assert response.structured == {"summary": "有效", "state_delta": {"chapter": 1}}
    assert client.chat.completions.calls == [
        {
            "model": "qwen-flash",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return a chapter summary.\n\n"
                        "Return only one JSON object matching this schema:\n"
                        + json.dumps(request.output_schema, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        request.input_payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
            "max_tokens": 256,
            "timeout": 5.0,
        }
    ]
    estimator = ConservativeEstimator()
    wire_messages = json.dumps(
        client.chat.completions.calls[0]["messages"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    budgeted_request = json.dumps(
        asdict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert estimator.estimate(wire_messages) <= estimator.estimate(budgeted_request)


def test_qwen_maps_response_id_and_usage() -> None:
    response = QwenProvider(
        FakeQwenClient(chat_response()), allow_real_calls=True
    ).generate(summary_request())

    assert response.provider_response_id == "chatcmpl-qwen-1"
    assert response.input_tokens == 37
    assert response.output_tokens == 14
    assert isinstance(response.latency_ms, int) and response.latency_ms >= 0


def test_qwen_advertises_truthful_conservative_capabilities() -> None:
    provider = QwenProvider(
        FakeQwenClient(chat_response()),
        allow_real_calls=True,
        context_window_limit=16_000,
        max_output_tokens_limit=4_000,
    )

    assert provider.capabilities("qwen-flash") == ProviderCapabilities(
        context_window=16_000,
        max_output_tokens=4_000,
        strict_structured_output=False,
        token_counting=True,
        local=False,
        real_calls_allowed=True,
    )


@pytest.mark.parametrize(
    ("allow_real_calls", "api_key", "message"),
    [
        (False, "configured", "disabled"),
        (True, None, "API key"),
        (True, "", "API key"),
    ],
)
def test_qwen_disabled_or_missing_key_never_calls_sdk(
    allow_real_calls: bool, api_key: str | None, message: str
) -> None:
    client = FakeQwenClient(chat_response(), api_key=api_key)

    with pytest.raises(ProviderAuthenticationError, match=message):
        QwenProvider(client, allow_real_calls=allow_real_calls).generate(summary_request())

    assert client.chat.completions.calls == []


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(id="bad", choices=[], usage=None),
        SimpleNamespace(id="bad", choices=None, usage=None),
        SimpleNamespace(id="bad", choices=[SimpleNamespace()], usage=None),
        chat_response(content=""),
        chat_response(content="[]"),
        chat_response(content="not-json"),
        chat_response(finish_reason="length"),
        chat_response(finish_reason="content_filter"),
        chat_response(refusal="cannot comply"),
    ],
)
def test_qwen_rejects_truncated_refused_or_malformed_responses(response: object) -> None:
    with pytest.raises(ProviderProtocolError, match="Qwen returned malformed structured output"):
        QwenProvider(FakeQwenClient(response), allow_real_calls=True).generate(summary_request())


def test_qwen_malformed_response_never_leaks_content(caplog: pytest.LogCaptureFixture) -> None:
    secret = "qwen-response-secret"
    response = chat_response(content=secret)

    with pytest.raises(ProviderProtocolError) as error:
        QwenProvider(FakeQwenClient(response), allow_real_calls=True).generate(summary_request())

    assert secret not in str(error.value)
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            lambda secret: AuthenticationError(
                secret,
                response=httpx2.Response(
                    401, request=httpx2.Request("POST", "https://qwen.test")
                ),
                body=None,
            ),
            ProviderAuthenticationError,
        ),
        (
            lambda secret: APITimeoutError(
                httpx2.Request("POST", f"https://{secret}.test")
            ),
            ProviderTimeout,
        ),
        (
            lambda secret: APIConnectionError(
                message=secret,
                request=httpx2.Request("POST", "https://qwen.test"),
            ),
            ProviderUnavailable,
        ),
        (
            lambda secret: APIStatusError(
                secret,
                response=httpx2.Response(
                    500, request=httpx2.Request("POST", "https://qwen.test")
                ),
                body=None,
            ),
            ProviderProtocolError,
        ),
    ],
)
def test_qwen_normalizes_sdk_errors_without_leaking_secrets(
    caplog: pytest.LogCaptureFixture,
    sdk_error: Any,
    expected_error: type[Exception],
) -> None:
    secret = "qwen-sdk-sentinel"

    with pytest.raises(expected_error) as error:
        QwenProvider(
            FakeQwenClient(sdk_error(secret)), allow_real_calls=True
        ).generate(summary_request())

    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_qwen_valid_json_still_uses_agent_runner_schema_validation() -> None:
    provider = QwenProvider(
        FakeQwenClient(chat_response(content='{"summary":12,"state_delta":{}}')),
        allow_real_calls=True,
    )

    with pytest.raises(ProviderProtocolError, match="provider returned an invalid response"):
        AgentRunner().run(provider, summary_request(), ChapterSummaryDelta)


def test_qwen_diagnose_is_configuration_only_and_never_calls_sdk() -> None:
    client = FakeQwenClient(chat_response())
    provider = QwenProvider(client, allow_real_calls=True)

    diagnostic = provider.diagnose("qwen-flash")

    assert diagnostic.available is True
    assert diagnostic.models == ("qwen-flash",)
    assert "configuration" in diagnostic.detail.casefold()
    assert "not an online connectivity check" in diagnostic.detail.casefold()
    assert client.chat.completions.calls == []

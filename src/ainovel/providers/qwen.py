from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI

from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
)


JSON_INSTRUCTION = "Return only one JSON object matching this schema:"


class QwenProvider:
    def __init__(
        self,
        client: OpenAI,
        allow_real_calls: bool,
        context_window_limit: int = 16_000,
        max_output_tokens_limit: int = 4_000,
        model_capabilities: Mapping[str, ProviderCapabilities] | None = None,
    ) -> None:
        self._validate_limit(context_window_limit)
        self._validate_limit(max_output_tokens_limit)
        for capability in (model_capabilities or {}).values():
            self._validate_limit(capability.context_window)
            self._validate_limit(capability.max_output_tokens)
        self._client = client
        self._allow_real_calls = allow_real_calls
        self._context_window_limit = context_window_limit
        self._max_output_tokens_limit = max_output_tokens_limit
        self._model_capabilities = dict(model_capabilities or {})

    def capabilities(self, model: str) -> ProviderCapabilities:
        override = self._model_capabilities.get(model)
        if override is None:
            return ProviderCapabilities(
                self._context_window_limit,
                self._max_output_tokens_limit,
                False,
                True,
                False,
                self._allow_real_calls,
            )
        return ProviderCapabilities(
            min(self._context_window_limit, override.context_window),
            min(self._max_output_tokens_limit, override.max_output_tokens),
            False,
            override.token_counting,
            False,
            self._allow_real_calls and override.real_calls_allowed,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._allow_real_calls:
            raise ProviderAuthenticationError("Qwen calls are disabled")
        api_key = getattr(self._client, "api_key", object())
        if api_key is None or api_key == "":
            raise ProviderAuthenticationError("Qwen API key is not configured")

        schema = json.dumps(
            request.output_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_content = f"{request.system_prompt}\n\n{JSON_INSTRUCTION}\n{schema}"
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": json.dumps(
                            request.input_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
                max_tokens=request.max_output_tokens,
                timeout=request.timeout_seconds,
            )
        except AuthenticationError:
            raise ProviderAuthenticationError("Qwen authentication failed") from None
        except APITimeoutError:
            raise ProviderTimeout("Qwen request timed out") from None
        except APIConnectionError:
            raise ProviderUnavailable("Qwen service is unavailable") from None
        except APIStatusError:
            raise ProviderProtocolError("Qwen returned an unsuccessful response") from None

        try:
            choices = response.choices
            if not isinstance(choices, list) or not choices:
                raise ValueError("choices are missing")
            choice = choices[0]
            if choice.finish_reason != "stop":
                raise ValueError("response did not finish normally")
            message = choice.message
            if getattr(message, "refusal", None) is not None:
                raise ValueError("response was refused")
            content = message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content is missing")
            structured = json.loads(content)
            if not isinstance(structured, dict):
                raise ValueError("structured response is not an object")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            raise ProviderProtocolError("Qwen returned malformed structured output") from None

        usage = getattr(response, "usage", None)
        return ModelResponse(
            structured=structured,
            text=content,
            provider_response_id=self._optional_string(getattr(response, "id", None)),
            input_tokens=self._optional_int(getattr(usage, "prompt_tokens", None)),
            output_tokens=self._optional_int(getattr(usage, "completion_tokens", None)),
            latency_ms=round((perf_counter() - started) * 1000),
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        if not self._allow_real_calls:
            return ProviderDiagnostic(
                False,
                "Qwen calls are disabled; configuration diagnosis is not an online connectivity check",
                (),
            )
        api_key = getattr(self._client, "api_key", object())
        if api_key is None or api_key == "":
            return ProviderDiagnostic(
                False,
                "Qwen API key is not configured; configuration diagnosis is not an online connectivity check",
                (),
            )
        models = (model,) if model is not None else ()
        return ProviderDiagnostic(
            True,
            "Qwen configuration is ready; this is not an online connectivity check",
            models,
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _validate_limit(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("provider capability limits must be positive integers")

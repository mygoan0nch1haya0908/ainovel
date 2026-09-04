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


class OpenAIProvider:
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
                True,
                True,
                False,
                self._allow_real_calls,
            )
        return ProviderCapabilities(
            min(self._context_window_limit, override.context_window),
            min(self._max_output_tokens_limit, override.max_output_tokens),
            override.strict_structured_output,
            override.token_counting,
            False,
            self._allow_real_calls and override.real_calls_allowed,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._allow_real_calls:
            raise ProviderAuthenticationError("OpenAI calls are disabled")
        api_key = getattr(self._client, "api_key", object())
        if api_key is None or api_key == "":
            raise ProviderAuthenticationError("OpenAI API key is not configured")

        started = perf_counter()
        try:
            response = self._client.responses.create(
                model=request.model,
                instructions=request.system_prompt,
                input=json.dumps(request.input_payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.metadata["schema_name"],
                        "strict": True,
                        "schema": request.output_schema,
                    }
                },
                max_output_tokens=request.max_output_tokens,
                timeout=request.timeout_seconds,
            )
        except AuthenticationError as error:
            raise ProviderAuthenticationError("OpenAI authentication failed") from None
        except APITimeoutError as error:
            raise ProviderTimeout("OpenAI request timed out") from None
        except APIConnectionError as error:
            raise ProviderUnavailable("OpenAI service is unavailable") from None
        except APIStatusError as error:
            raise ProviderProtocolError("OpenAI returned an unsuccessful response") from None

        try:
            output_text = response.output_text
            if not isinstance(output_text, str):
                raise ValueError("output text is not a string")
            structured = json.loads(output_text)
            if not isinstance(structured, dict):
                raise ValueError("structured response is not an object")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderProtocolError("OpenAI returned malformed structured output") from None

        usage = getattr(response, "usage", None)
        return ModelResponse(
            structured=structured,
            text=output_text,
            provider_response_id=self._optional_string(getattr(response, "id", None)),
            input_tokens=self._optional_int(getattr(usage, "input_tokens", None)),
            output_tokens=self._optional_int(getattr(usage, "output_tokens", None)),
            latency_ms=round((perf_counter() - started) * 1000),
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        if not self._allow_real_calls:
            return ProviderDiagnostic(False, "OpenAI calls are disabled", ())
        api_key = getattr(self._client, "api_key", object())
        if api_key is None or api_key == "":
            return ProviderDiagnostic(False, "OpenAI API key is not configured", ())
        models = (model,) if model is not None else ()
        return ProviderDiagnostic(True, "OpenAI calls are enabled", models)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _validate_limit(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("provider capability limits must be positive integers")

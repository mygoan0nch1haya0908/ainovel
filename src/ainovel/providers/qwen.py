from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI
from ainovel.providers.diagnostics import FailureReason, ResponseFailure

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
        api_key_configured: bool | None = None,
    ) -> None:
        self._validate_limit(context_window_limit)
        self._validate_limit(max_output_tokens_limit)
        for capability in (model_capabilities or {}).values():
            self._validate_limit(capability.context_window)
            self._validate_limit(capability.max_output_tokens)
        self._client = client
        self._allow_real_calls = allow_real_calls
        if api_key_configured is None:
            api_key = getattr(client, "api_key", object())
            self._api_key_configured = api_key is not None and api_key != ""
        elif type(api_key_configured) is bool:
            self._api_key_configured = api_key_configured
        else:
            raise TypeError("api_key_configured must be a boolean")
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
                self._allow_real_calls and self._api_key_configured,
            )
        return ProviderCapabilities(
            min(self._context_window_limit, override.context_window),
            min(self._max_output_tokens_limit, override.max_output_tokens),
            False,
            override.token_counting,
            False,
            self._allow_real_calls
            and self._api_key_configured
            and override.real_calls_allowed,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._allow_real_calls:
            raise ProviderAuthenticationError("Qwen calls are disabled")
        if not self._api_key_configured:
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
            raise ResponseFailure(FailureReason.HTTP) from None

        # Validate metadata independently before parsing potentially invalid content.
        provider_response_id = getattr(response, "id", None)
        if not isinstance(provider_response_id, str) or not provider_response_id.strip():
            provider_response_id = None
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if type(input_tokens) is not int or input_tokens < 0:
            input_tokens = None
        if type(output_tokens) is not int or output_tokens < 0:
            output_tokens = None
        metadata = ModelResponse(
            structured=None, text=None, provider_response_id=provider_response_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=round((perf_counter() - started) * 1000),
        )
        try:
            choices = response.choices
            if not isinstance(choices, list) or not choices:
                raise ResponseFailure(FailureReason.ENVELOPE)
            choice = choices[0]
            if choice.finish_reason == "length":
                raise ResponseFailure(FailureReason.TRUNCATED)
            if choice.finish_reason == "content_filter":
                raise ResponseFailure(FailureReason.REFUSED)
            if choice.finish_reason != "stop":
                raise ResponseFailure(FailureReason.FINISH)
            message = choice.message
            if getattr(message, "refusal", None) is not None:
                raise ResponseFailure(FailureReason.REFUSED)
            content = message.content
            if not isinstance(content, str) or not content.strip():
                raise ResponseFailure(FailureReason.EMPTY)
            try:
                structured = json.loads(content)
            except ValueError:
                raise ResponseFailure(FailureReason.JSON) from None
            if not isinstance(structured, dict):
                raise ResponseFailure(FailureReason.JSON)
        except ResponseFailure as error:
            error.response = metadata
            raise
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            raise ResponseFailure(FailureReason.ENVELOPE, response=metadata) from None

        if provider_response_id is None or input_tokens is None or output_tokens is None:
            raise ResponseFailure(FailureReason.METADATA, response=metadata) from None

        return ModelResponse(
            structured=structured,
            text=content,
            provider_response_id=provider_response_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((perf_counter() - started) * 1000),
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        if not self._allow_real_calls:
            return ProviderDiagnostic(
                False,
                "Qwen calls are disabled; configuration diagnosis is not an online connectivity check",
                (),
            )
        if not self._api_key_configured:
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
    def _validate_limit(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("provider capability limits must be positive integers")

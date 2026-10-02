from __future__ import annotations

import json
from dataclasses import replace
from ainovel.providers.request_diagnostics import trace_request,note_sdk_response
from ainovel.providers.llm_response import attach_sdk_diagnostic
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI
from ainovel.providers.llm_response import parse_llm_json_response, raise_sdk_status_error, log_diagnostic, check_sdk_response_error

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
        try:
            if not self._allow_real_calls or getattr(self._client,'api_key',object()) in (None,''):
                return self._generate(request)
            messages=[{'role':'system','content':request.system_prompt},{'role':'user','content':json.dumps(request.input_payload,ensure_ascii=False)}]
            with trace_request(request,str(getattr(self._client,'base_url','')),getattr(self._client,'api_key',None),messages=messages,response_format='json_schema') as trace:
                response=self._generate(request)
            return replace(response,diagnostic=trace['diagnostic'])
        except Exception as error:
            log_diagnostic(model=request.model, base_url=getattr(self._client, 'base_url', ''),
                           api_key=getattr(self._client, 'api_key', None), exception_type=type(error).__name__)
            raise

    def _generate(self, request: ModelRequest) -> ModelResponse:
        if not self._allow_real_calls:
            raise ProviderAuthenticationError("OpenAI calls are disabled")
        api_key = getattr(self._client, "api_key", object())
        if api_key is None or api_key == "":
            raise ProviderAuthenticationError("OpenAI API key is not configured")

        started = perf_counter()
        try:
            from ainovel.providers.request_diagnostics import note_actual_send
            note_actual_send()
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
            raise attach_sdk_diagnostic(ProviderAuthenticationError("OpenAI authentication failed"),error,model=request.model,client=self._client) from None
        except APITimeoutError as error:
            raise attach_sdk_diagnostic(ProviderTimeout("OpenAI request timed out"),error,model=request.model,client=self._client) from None
        except APIConnectionError as error:
            raise ProviderUnavailable("OpenAI service is unavailable") from None
        except APIStatusError as error:
            raise_sdk_status_error(error, model=request.model, client=self._client)

        note_sdk_response(response,responses_api=True,api_key=getattr(self._client,'api_key',None))
        check_sdk_response_error(response)
        try:
            output_text = response.output_text
            if not isinstance(output_text, str):
                raise ValueError("output text is not a string")
            log_diagnostic(model=request.model, base_url=getattr(self._client, 'base_url', ''),
                           content=output_text, api_key=getattr(self._client, 'api_key', None))
            structured = parse_llm_json_response(output_text, api_key=getattr(self._client, 'api_key', None))
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

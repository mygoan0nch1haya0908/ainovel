from __future__ import annotations

import json
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
    def __init__(self, client: OpenAI, allow_real_calls: bool) -> None:
        self._client = client
        self._allow_real_calls = allow_real_calls

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128000, 16000, True, True, False, self._allow_real_calls)

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

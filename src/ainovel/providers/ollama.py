from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import httpx

from ainovel.context import effective_input_capacity
from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
)


class OllamaProvider:
    def __init__(
        self,
        client: httpx.Client,
        base_url: str,
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
        self._base_url = base_url.rstrip("/")
        self._context_window_limit = context_window_limit
        self._max_output_tokens_limit = max_output_tokens_limit
        self._model_capabilities = dict(model_capabilities or {})

    def capabilities(self, model: str) -> ProviderCapabilities:
        override = self._model_capabilities.get(model)
        if override is None:
            return ProviderCapabilities(
                self._context_window_limit, self._max_output_tokens_limit, True, True, True, True
            )
        return ProviderCapabilities(
            min(self._context_window_limit, override.context_window),
            min(self._max_output_tokens_limit, override.max_output_tokens),
            override.strict_structured_output,
            override.token_counting,
            override.local,
            override.real_calls_allowed,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        capability = self.capabilities(request.model)
        try:
            capacity = effective_input_capacity(
                request.max_input_tokens, capability.context_window,
                request.max_output_tokens,
            )
        except ValueError:
            raise ProviderProtocolError("Ollama request budget exceeds capability") from None
        if request.max_input_tokens > capacity or request.max_output_tokens > capability.max_output_tokens:
            raise ProviderProtocolError("Ollama request budget exceeds capability")
        payload = {
            "model": request.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": json.dumps(request.input_payload, ensure_ascii=False)},
            ],
            "format": request.output_schema,
            "options": {
                "num_ctx": capability.context_window,
                "num_predict": request.max_output_tokens,
            },
        }
        started = perf_counter()
        try:
            response = self._client.post(
                f"{self._base_url}/api/chat", json=payload, timeout=request.timeout_seconds
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise ProviderTimeout("Ollama request timed out") from None
        except httpx.ConnectError as error:
            raise ProviderUnavailable("Ollama service is unavailable") from None
        except httpx.HTTPStatusError as error:
            raise ProviderProtocolError("Ollama returned an unsuccessful response") from None
        except httpx.RequestError as error:
            raise ProviderUnavailable("Ollama request failed") from None

        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("response body is not an object")
            message = body["message"]
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("message content is not a string")
            structured = json.loads(content)
            if not isinstance(structured, dict):
                raise ValueError("structured response is not an object")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderProtocolError("Ollama returned malformed structured output") from None

        return ModelResponse(
            structured=structured,
            text=content,
            provider_response_id=self._optional_string(body.get("id")),
            input_tokens=self._optional_int(body.get("prompt_eval_count")),
            output_tokens=self._optional_int(body.get("eval_count")),
            latency_ms=round((perf_counter() - started) * 1000),
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("models"), list):
                raise ValueError("malformed model list")
            models = tuple(sorted(
                item["name"] for item in body["models"]
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ))
        except httpx.TimeoutException:
            return ProviderDiagnostic(False, "Ollama diagnosis timed out", ())
        except httpx.HTTPStatusError:
            return ProviderDiagnostic(False, "Ollama service is unavailable", ())
        except httpx.RequestError:
            return ProviderDiagnostic(False, "Ollama service is unavailable", ())
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return ProviderDiagnostic(False, "Ollama returned an invalid model list", ())

        if model is not None and model not in models:
            return ProviderDiagnostic(False, "requested Ollama model is not installed", models)
        return ProviderDiagnostic(True, "Ollama service is available", models)

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

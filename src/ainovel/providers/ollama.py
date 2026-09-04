from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx

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
    def __init__(self, client: httpx.Client, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128000, 16000, True, True, True, True)

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": request.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": json.dumps(request.input_payload, ensure_ascii=False)},
            ],
            "format": request.output_schema,
            "options": {
                "num_ctx": request.max_input_tokens,
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

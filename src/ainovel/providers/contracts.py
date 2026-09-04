from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderCapabilities:
    context_window: int
    max_output_tokens: int
    strict_structured_output: bool
    token_counting: bool
    local: bool
    real_calls_allowed: bool


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system_prompt: str
    input_payload: dict[str, object]
    output_schema: dict[str, object]
    max_input_tokens: int
    max_output_tokens: int
    timeout_seconds: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class ModelResponse:
    structured: dict[str, object] | None
    text: str | None
    provider_response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


@dataclass(frozen=True)
class ProviderDiagnostic:
    available: bool
    detail: str
    models: tuple[str, ...]


class ProviderError(Exception):
    """Base error for provider integration failures."""


class ProviderUnavailable(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderProtocolError(ProviderError):
    pass


@runtime_checkable
class ModelProvider(Protocol):
    def capabilities(self, model: str) -> ProviderCapabilities:
        raise NotImplementedError

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        raise NotImplementedError

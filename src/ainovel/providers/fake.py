from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from typing import Sequence

from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderProtocolError,
)


class FakeProvider:
    def __init__(self, script: Sequence[ModelResponse | Exception]) -> None:
        self._script = deque(script)
        self.requests: list[ModelRequest] = []

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128000, 16000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(
            replace(
                request,
                input_payload=deepcopy(request.input_payload),
                output_schema=deepcopy(request.output_schema),
                metadata=deepcopy(request.metadata),
            )
        )
        if not self._script:
            raise ProviderProtocolError("fake provider script exhausted")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        models = (model,) if model is not None else ("scripted",)
        return ProviderDiagnostic(True, "scripted fake provider available", models)

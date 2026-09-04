from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from ainovel.providers.contracts import ModelProvider, ProviderProtocolError


ProviderName = Literal["fake", "ollama", "openai"]


class ProviderRegistry:
    def __init__(self, factories: Mapping[ProviderName, Callable[[], ModelProvider]]) -> None:
        self._factories = dict(factories)

    def get(self, name: ProviderName) -> ModelProvider:
        factory = self._factories.get(name)
        if factory is None:
            raise ProviderProtocolError("unknown provider")
        return factory()

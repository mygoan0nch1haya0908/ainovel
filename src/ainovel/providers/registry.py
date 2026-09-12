from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from ainovel.providers.contracts import (
    ModelProvider,
    ProviderError,
    ProviderProtocolError,
    ProviderUnavailable,
)


ProviderName = Literal["fake", "ollama", "openai", "qwen"]


class ProviderRegistry:
    def __init__(self, factories: Mapping[ProviderName, Callable[[], ModelProvider]]) -> None:
        self._factories = dict(factories)

    def get(self, name: ProviderName) -> ModelProvider:
        factory = self._factories.get(name)
        if factory is None:
            raise ProviderProtocolError("unknown provider")
        try:
            return factory()
        except ProviderError:
            raise
        except Exception:
            raise ProviderUnavailable("provider is unavailable") from None

    def contains(self, name: str) -> bool:
        return name in self._factories

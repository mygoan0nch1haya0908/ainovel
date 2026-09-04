from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ainovel.providers.contracts import ModelProvider, ModelRequest, ProviderProtocolError

ResultType = TypeVar("ResultType", bound=BaseModel)


class AgentRunner:
    def run(
        self, provider: ModelProvider, request: ModelRequest, result_type: type[ResultType]
    ) -> ResultType:
        response = provider.generate(request)
        if response.structured is None:
            raise ProviderProtocolError("provider response did not include structured output")
        try:
            return result_type.model_validate(response.structured)
        except ValidationError as error:
            raise ProviderProtocolError("provider returned invalid structured output") from error

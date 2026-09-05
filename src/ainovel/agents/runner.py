from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ainovel.providers.contracts import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderProtocolError,
)

ResultType = TypeVar("ResultType", bound=BaseModel)


@dataclass(frozen=True)
class AgentRunResult(Generic[ResultType]):
    result: ResultType
    response: ModelResponse


class AgentRunner:
    def run(
        self, provider: ModelProvider, request: ModelRequest, result_type: type[ResultType]
    ) -> ResultType:
        return self.run_with_response(provider, request, result_type).result

    def run_with_response(
        self, provider: ModelProvider, request: ModelRequest, result_type: type[ResultType]
    ) -> AgentRunResult[ResultType]:
        response = provider.generate(request)
        if response.structured is None:
            raise ProviderProtocolError("provider response did not include structured output")
        try:
            result = result_type.model_validate(response.structured)
        except ValidationError as error:
            raise ProviderProtocolError("provider returned invalid structured output") from None
        return AgentRunResult(result=result, response=response)

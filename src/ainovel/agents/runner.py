from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError
from ainovel.agents.contracts import ChapterDraft
from ainovel.services.counting import count_visible_characters
from ainovel.providers.diagnostics import FailureReason, ResponseFailure

from ainovel.providers.contracts import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderProtocolError,
    ProviderUnavailable,
)

ResultType = TypeVar("ResultType", bound=BaseModel)
INVALID_PROVIDER_RESPONSE = "provider returned an invalid response"


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
        try:
            response = provider.generate(request)
        except ProviderError:
            raise
        except Exception:
            raise ProviderUnavailable("provider is unavailable") from None
        response = self._validate_provider_response(response)
        try:
            result = result_type.model_validate(response.structured)
        except ValidationError:
            reason = FailureReason.SCHEMA
            count = None
            if result_type is ChapterDraft:
                title = response.structured.get('title')
                body = response.structured.get('body')
                if isinstance(title, str) and isinstance(body, str):
                    if not title.strip() or not body.strip():
                        reason = FailureReason.CHAPTER_EMPTY
                    else:
                        count = count_visible_characters(body)
                        if count < 4500:
                            reason = FailureReason.TOO_SHORT
                        elif count > 6000:
                            reason = FailureReason.TOO_LONG
            raise ResponseFailure(
                reason, visible_count=count, response=response
            ) from None
        return AgentRunResult(result=result, response=response)

    @staticmethod
    def _validate_provider_response(response: object) -> ModelResponse:
        if not isinstance(response, ModelResponse):
            raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        if not isinstance(response.structured, dict) or not all(
            isinstance(key, str) for key in response.structured
        ):
            raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        if response.text is not None and not isinstance(response.text, str):
            raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        if response.provider_response_id is not None and not isinstance(
            response.provider_response_id, str
        ):
            raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        for value in (response.input_tokens, response.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        if type(response.latency_ms) is not int or response.latency_ms < 0:
            raise ProviderProtocolError(INVALID_PROVIDER_RESPONSE) from None
        return response

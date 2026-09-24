"""Generic Chat Completions JSON-object adapter for an explicitly chosen endpoint.

AgentRunner remains the local schema/domain validation boundary. Construction and
diagnosis do not authorize networking; callers must explicitly enable dispatch.
"""
from __future__ import annotations

import json
import math
import re
from time import perf_counter

from ainovel.providers.contracts import (
    ModelRequest, ModelResponse, ProviderAuthenticationError, ProviderCapabilities,
    ProviderDiagnostic, ProviderProtocolError, ProviderTimeout, ProviderUnavailable,
)
from ainovel.providers.diagnostics import FailureReason, ResponseFailure
from ainovel.providers.endpoint_policy import Endpoint, normalize_endpoint
from ainovel.providers.safe_transport import SafeTransport, parse_json_object


class CompatibleProvider:
    def __init__(self, endpoint: Endpoint, model_name: str, *, api_key: str | None = None,
                 transport=None, allow_real_calls: bool = False,
                 context_window_limit: int = 32000, max_output_tokens_limit: int = 12000):
        if (normalize_endpoint(endpoint.base_url, endpoint.kind) != endpoint
                or not isinstance(model_name, str) or not 1 <= len(model_name) <= 255
                or any(ord(c) < 32 or ord(c) == 127 for c in model_name)
                or type(context_window_limit) is not int or not 1 <= context_window_limit <= 32000
                or type(max_output_tokens_limit) is not int or not 1 <= max_output_tokens_limit <= 12000
                or max_output_tokens_limit > context_window_limit
                or type(allow_real_calls) is not bool
                or (api_key is not None and (not isinstance(api_key, str) or not 1 <= len(api_key) <= 8192
                    or any(ord(c) < 33 or ord(c) > 126 for c in api_key)))
                or (api_key and (api_key in model_name or api_key in endpoint.base_url))):
            raise ProviderProtocolError('invalid model configuration')
        self._endpoint = endpoint
        self._model = model_name
        self._api_key = api_key
        self._transport = transport if transport is not None else SafeTransport()
        self._allowed = allow_real_calls
        self._context = context_window_limit
        self._output = max_output_tokens_limit

    def _ready(self):
        return self._allowed and (bool(self._api_key) or self._endpoint.kind == 'loopback')

    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(self._context, self._output, False, True,
            self._endpoint.kind == 'loopback', self._ready() and model == self._model)

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        ready = self._ready() and model in (None, self._model)
        return ProviderDiagnostic(ready,
            'Model configuration is ready; this is not an online connectivity check' if ready else
            'Model configuration is unavailable; this is not an online connectivity check',
            (self._model,) if ready else ())

    def _reject_secret(self, value):
        if not self._api_key:
            return
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, (str, int, float)) and self._api_key in str(item):
                raise ProviderProtocolError('model returned an unsafe response')

    def _dispatch(self, method, path, *, payload=None, timeout_seconds=30.0, max_response_bytes):
        if not self._ready():
            raise ProviderAuthenticationError('model calls are disabled or credentials are missing')
        try:
            result = self._transport.request_json(self._endpoint, method, path, api_key=self._api_key,
                payload=payload, timeout_seconds=min(timeout_seconds, 180.0), max_response_bytes=max_response_bytes)
            self._reject_secret(result)
            if not isinstance(result, dict):
                raise ProviderProtocolError('model returned an invalid response')
            return result
        except ProviderAuthenticationError:
            raise ProviderAuthenticationError('model authentication failed') from None
        except ProviderTimeout:
            raise ProviderTimeout('model request timed out') from None
        except ProviderProtocolError:
            raise ProviderProtocolError('model returned an invalid or unsafe response') from None
        except Exception:
            raise ProviderUnavailable('model service is unavailable') from None

    def generate(self, request: ModelRequest) -> ModelResponse:
        return self._generate(request, max_response_bytes=2097152)

    def _generate(self, request: ModelRequest, *, max_response_bytes: int) -> ModelResponse:
        if (request.model != self._model
                or type(request.max_output_tokens) is not int or not 1 <= request.max_output_tokens <= self._output
                or type(request.max_input_tokens) is not int or not 1 <= request.max_input_tokens <= self._context
                or request.max_input_tokens + request.max_output_tokens > self._context
                or isinstance(request.timeout_seconds, bool) or not isinstance(request.timeout_seconds, (int, float))
                or not math.isfinite(request.timeout_seconds) or request.timeout_seconds <= 0):
            raise ProviderProtocolError('invalid model request budget or model')
        try:
            schema = json.dumps(request.output_schema, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            payload = {'model': self._model, 'messages': [
                {'role': 'system', 'content': f'{request.system_prompt}\n\nReturn only one JSON object matching this schema:\n{schema}'},
                {'role': 'user', 'content': json.dumps(request.input_payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}],
                'response_format': {'type': 'json_object'}, 'max_tokens': request.max_output_tokens}
            self._reject_secret(payload)
        except Exception:
            raise ProviderProtocolError('invalid model input') from None
        started = perf_counter()
        result = self._dispatch('POST', 'chat/completions', payload=payload,
            timeout_seconds=request.timeout_seconds, max_response_bytes=max_response_bytes)
        response_id = result.get('id')
        if not isinstance(response_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,255}', response_id):
            response_id = None
        usage = result.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        def count(name):
            value = usage.get(name)
            return value if type(value) is int and 0 <= value <= 1_000_000_000 else None
        metadata = ModelResponse(None, None, response_id, count('prompt_tokens'), count('completion_tokens'),
            max(0, round((perf_counter() - started) * 1000)))
        try:
            choices = result.get('choices')
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ResponseFailure(FailureReason.ENVELOPE)
            choice = choices[0]
            finish = choice.get('finish_reason')
            if finish == 'length':
                raise ResponseFailure(FailureReason.TRUNCATED)
            if finish == 'content_filter':
                raise ResponseFailure(FailureReason.REFUSED)
            if finish != 'stop':
                raise ResponseFailure(FailureReason.FINISH)
            message = choice.get('message')
            if not isinstance(message, dict):
                raise ResponseFailure(FailureReason.ENVELOPE)
            if message.get('refusal') is not None:
                raise ResponseFailure(FailureReason.REFUSED)
            content = message.get('content')
            if not isinstance(content, str) or not content.strip():
                raise ResponseFailure(FailureReason.EMPTY)
            try:
                structured = parse_json_object(content)
            except (ValueError, RecursionError):
                raise ResponseFailure(FailureReason.JSON) from None
            self._reject_secret(structured)
            if response_id is None or metadata.input_tokens is None or metadata.output_tokens is None:
                raise ResponseFailure(FailureReason.METADATA)
        except ResponseFailure as error:
            error.response = metadata
            # Keep shared accounting/reason contracts with a provider-neutral message.
            error.args = ('model returned an invalid response',)
            raise
        return ModelResponse(structured, content, response_id, metadata.input_tokens,
            metadata.output_tokens, metadata.latency_ms)

    def list_models(self) -> tuple[str, ...]:
        """Return exact bounded IDs; render only with autoescape or textContent."""
        result = self._dispatch('GET', 'models', max_response_bytes=1048576)
        entries = result.get('data')
        if not isinstance(entries, list):
            raise ProviderProtocolError('model list is invalid')
        models = []
        for entry in entries:
            identifier = entry.get('id') if isinstance(entry, dict) else None
            if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 255
                    or not identifier.strip() or any(ord(c) < 32 or ord(c) == 127 for c in identifier)):
                raise ProviderProtocolError('model list is invalid')
            if identifier not in models:
                models.append(identifier)
            if len(models) > 200:
                raise ProviderProtocolError('model list exceeds allowed size')
        return tuple(models)

    def test_connection(self) -> ProviderDiagnostic:
        output = min(128, self._output, self._context - 1)
        if output < 1:
            raise ProviderProtocolError('model context limit is too small for connection test')
        request = ModelRequest(self._model, 'Connection test. Return {"ok":true}.', {'test': True},
            {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False},
            self._context - output, output, 30.0, {})
        response = self._generate(request, max_response_bytes=65536)
        if (not isinstance(response.structured, dict) or set(response.structured) != {'ok'}
                or response.structured['ok'] is not True):
            raise ProviderProtocolError('model connection test returned an invalid response')
        return ProviderDiagnostic(True, 'Model connection test succeeded', (self._model,))

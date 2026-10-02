import json
import logging
from dataclasses import replace

import pytest

from ainovel.providers.compatible import CompatibleProvider
from ainovel.providers.contracts import ProviderTimeout
from ainovel.providers.diagnostics import safe_failure_code
from ainovel.providers.endpoint_policy import normalize_endpoint
from ainovel.providers.safe_transport import SafeTransport
from test_safe_transport import ScriptedSocket, Connector, response
from test_compatible_provider import Transport, reply, request


@pytest.fixture(autouse=True)
def enable_test_capture(monkeypatch):
    # Migration tests use fileConfig which disables existing named loggers.
    monkeypatch.setattr(logging.getLogger('ainovel.llm.telemetry'), 'disabled', False)


def test_application_logging_recovers_after_migration_configuration():
    from ainovel.providers.request_diagnostics import configure_telemetry
    logger = logging.getLogger('ainovel.llm.telemetry')
    logger.disabled = True
    configure_telemetry()
    assert logger.isEnabledFor(logging.INFO)


def test_local_response_timeout_has_phase_and_no_usage_or_content(caplog):
    class WaitingSocket(ScriptedSocket):
        def recv(self, size):
            raise TimeoutError('SECRET_NETWORK_ERROR')
    sock = WaitingSocket(b'')
    transport = SafeTransport(resolver=lambda *a: ['93.184.216.34'], connector=Connector(sock))
    provider = CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'),
        'model-a', api_key='SECRET_KEY', transport=transport, allow_real_calls=True)
    with caplog.at_level(logging.INFO, logger='ainovel.llm.telemetry'), pytest.raises(ProviderTimeout) as caught:
        provider.generate(replace(request(), input_payload={'text': 'PRIVATE_NOVEL'}))
    assert safe_failure_code(caught.value) == 'client_timeout_response_headers'
    assert caught.value.source == 'client'
    records = [json.loads(r.message.split('] ', 1)[1]) for r in caplog.records if r.name == 'ainovel.llm.telemetry']
    before, sent, after = records
    assert sent['llm_call_id'] == after['llm_call_id']
    assert sent['actual_process_concurrency'] >= 1
    assert before['message_count'] == 2 and before['tool_chars'] == 0
    assert before['assistant_chars'] == 0 and before['attempt'] == 1
    assert after['http_status'] is None and after['usage'] is None
    assert after['received_response'] is False and after['received_content'] is False
    assert after['phase'] == 'response_headers'
    assert not any(v in caplog.text for v in ['SECRET_KEY', 'PRIVATE_NOVEL', 'SECRET_NETWORK_ERROR'])
    assert sock.closed


@pytest.mark.parametrize('status,body,expected', [
    (b'200 OK', b'{"error":{"code":"provider_timeout","message":"SECRET"}}', 'upstream_timeout'),
    (b'504 Gateway Timeout', b'{"error":{"code":"provider_timeout"}}', 'upstream_timeout'),
    (b'500 Server Error', b'{"error":{"message":"SECRET"}}', 'provider_temporary'),
    (b'429 Too Many Requests', b'{"error":{"code":"insufficient_quota"}}', 'provider_quota'),
])
def test_http_errors_distinguish_upstream_from_client(status, body, expected, caplog):
    sock = ScriptedSocket(response(body, b'X-Request-ID: req-123\r\n', status))
    transport = SafeTransport(resolver=lambda *a: ['93.184.216.34'], connector=Connector(sock))
    with pytest.raises(Exception) as caught:
        transport.request_json(normalize_endpoint('https://api.example/v1', 'remote'),
            'POST', 'chat/completions', api_key='SECRET_KEY', payload={}, timeout_seconds=120, max_response_bytes=1024)
    assert safe_failure_code(caught.value) == expected
    assert 'SECRET' not in str(caught.value) + caplog.text


@pytest.mark.parametrize('content,code', [('{"status":"ok"}', None), ('hello', 'model_content_json')])
def test_response_diagnostics_keep_usage_even_for_format_error(content, code, caplog):
    provider = CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'),
        'model-a', api_key='SECRET_KEY', transport=Transport(reply(content)), allow_real_calls=True)
    with caplog.at_level(logging.INFO, logger='ainovel.llm.telemetry'):
        if code:
            with pytest.raises(Exception) as caught:
                provider.generate(request())
            assert safe_failure_code(caught.value) == code
        else:
            assert provider.generate(request()).structured == {'status': 'ok'}
    after = json.loads(caplog.records[-1].message.split('] ', 1)[1])
    assert after['usage'] == {'prompt_tokens': 37, 'completion_tokens': 14, 'total_tokens': 51}
    assert after['received_response'] and after['received_content']
    assert after['error_code'] == code


def test_health_retry_is_bounded_and_messages_unchanged(monkeypatch):
    import ainovel.llm_health as health
    from ainovel.providers.contracts import ModelResponse, ProviderCapabilities
    calls, waits = [], []
    class Provider:
        def capabilities(self, model):
            return ProviderCapabilities(16000, 1000, False, True, False, True)
        def generate(self, req):
            calls.append(req)
            if len(calls) < 3:
                raise ProviderTimeout('timeout')
            return ModelResponse({'status':'ok'}, '{"status":"ok"}', 'ok', 10, 5, 1)
    monkeypatch.setattr(health.time, 'sleep', waits.append)
    assert health.probe(Provider(), 'model', retries=2, synthetic_chars=100) == {'status':'ok'}
    assert len(calls) == 3 and len(waits) == 2
    assert 1 <= waits[0] < 2 and 2 <= waits[1] < 3
    assert calls[0].input_payload == calls[1].input_payload == calls[2].input_payload
    assert calls[0].metadata['attempt'] == '1' and calls[2].metadata['attempt'] == '3'


def test_health_retry_exhaustion_and_quota_never_retried(monkeypatch):
    import ainovel.llm_health as health
    from ainovel.providers.llm_response import LLMQuotaError
    from ainovel.providers.contracts import ProviderCapabilities
    monkeypatch.setattr(health.time, 'sleep', lambda _: None)
    for error, expected in [(ProviderTimeout('timeout'), 3), (LLMQuotaError(), 1)]:
        calls = []
        class Provider:
            def capabilities(self, model):
                return ProviderCapabilities(1000, 200, False, True, False, True)
            def generate(self, req):
                calls.append(req)
                raise error
        with pytest.raises(type(error)):
            health.probe(Provider(), 'model', retries=2)
        assert len(calls) == expected


@pytest.mark.parametrize('phase', ['dns', 'connect', 'write', 'response_body'])
def test_each_local_timeout_phase_is_preserved(phase):
    class PhaseSocket(ScriptedSocket):
        def sendall(self, data):
            if phase == 'write':
                raise TimeoutError()
            super().sendall(data)
        def recv(self, size):
            if phase == 'response_body' and not self.response:
                raise TimeoutError()
            return super().recv(size)
    class PhaseConnector(Connector):
        def connect(self, *a, **kw):
            if phase == 'connect':
                raise TimeoutError()
            return super().connect(*a, **kw)
    class PhaseTransport(SafeTransport):
        def _resolve(self, *a):
            if phase == 'dns':
                raise ProviderTimeout('timeout')
            return ['93.184.216.34']
    sock = PhaseSocket(b'HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n')
    with pytest.raises(ProviderTimeout) as caught:
        PhaseTransport(connector=PhaseConnector(sock)).request_json(
            normalize_endpoint('https://api.example/v1', 'remote'), 'POST', 'chat/completions',
            api_key='SECRET_KEY', payload={}, timeout_seconds=120, max_response_bytes=1024)
    assert safe_failure_code(caught.value) == 'client_timeout_' + phase


def test_response_trace_identifier_is_logged_without_body(caplog):
    wire = response(json.dumps(reply()).encode(), b'X-Request-ID: req-123\r\n')
    transport = SafeTransport(resolver=lambda *a: ['93.184.216.34'], connector=Connector(ScriptedSocket(wire)))
    provider = CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'),
        'model-a', api_key='SECRET_KEY', transport=transport, allow_real_calls=True)
    with caplog.at_level(logging.INFO, logger='ainovel.llm.telemetry'):
        provider.generate(request())
    after = json.loads(caplog.records[-1].message.split('] ', 1)[1])
    assert after['request_id'] == 'req-123' and after['http_status'] == 200
    assert 'state_delta' not in caplog.text


@pytest.mark.parametrize('value', [-1, 100001, True])
def test_probe_rejects_unbounded_synthetic_inputs_before_dispatch(value):
    from ainovel.llm_health import probe
    with pytest.raises(ValueError):
        probe(None, 'model', synthetic_chars=value)


def test_provider_error_envelope_is_recorded_as_received_response(caplog):
    provider = CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'),
        'model-a', api_key='SECRET_KEY',
        transport=Transport({'error': {'code': 'provider_timeout', 'message': 'PRIVATE_NOVEL'}}), allow_real_calls=True)
    with caplog.at_level(logging.INFO, logger='ainovel.llm.telemetry'), pytest.raises(ProviderTimeout):
        provider.generate(request())
    after = json.loads(caplog.records[-1].message.split('] ', 1)[1])
    assert after['received_response'] is True and after['received_content'] is False
    assert after['provider_code'] == 'provider_timeout'
    assert after['usage'] is None and 'PRIVATE_NOVEL' not in caplog.text


@pytest.mark.parametrize('changes', [
    {'timeout_seconds': 'SECRET_VALUE'}, {'timeout_seconds': None},
    {'input_payload': {'private': object()}}, {'output_schema': {'private': object()}},
])
def test_diagnostics_do_not_bypass_invalid_request_sanitization(changes):
    from ainovel.providers.contracts import ProviderProtocolError
    transport = Transport(reply())
    provider = CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'),
        'model-a', api_key='SECRET_KEY', transport=transport, allow_real_calls=True)
    with pytest.raises(ProviderProtocolError) as caught:
        provider.generate(replace(request(), **changes))
    assert transport.calls == [] and 'SECRET' not in str(caught.value)

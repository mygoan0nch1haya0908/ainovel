import importlib
import importlib.util
import json
import socket
import ssl
import threading
import time

import pytest

from ainovel.providers.contracts import ProviderError, ProviderTimeout
from ainovel.providers.endpoint_policy import normalize_endpoint

SENTINEL = 'FAKE_TRANSPORT_KEY_42'


@pytest.fixture(autouse=True)
def prohibit_real_dns(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('real DNS is prohibited'))


@pytest.fixture
def api():
    class API:
        def __getattr__(self, name):
            assert importlib.util.find_spec('ainovel.providers.safe_transport'), 'guarded transport is missing'
            return getattr(importlib.import_module('ainovel.providers.safe_transport'), name)
    return API()


class ScriptedSocket:
    def __init__(self, response):
        self.response = bytearray(response)
        self.sent = b''
        self.closed = False
        self.timeouts = []

    def recv(self, size):
        data = self.response[:size]
        del self.response[:size]
        return bytes(data)

    def sendall(self, data):
        self.sent += data

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def close(self):
        self.closed = True


class Connector:
    def __init__(self, sock):
        self.sock = sock
        self.calls = []

    def connect(self, address, port, **kwargs):
        self.calls.append((address, port, kwargs))
        return self.sock


def response(body=b'{"data":[]}', headers=b'', status=b'200 OK'):
    return b'HTTP/1.1 ' + status + b'\r\n' + headers + b'\r\n' + body


def call(api, wire, *, answers=None, endpoint=None, limit=1048576):
    sock = ScriptedSocket(wire)
    connector = Connector(sock)
    transport = api.SafeTransport(resolver=lambda h, p: answers or ['93.184.216.34'], connector=connector)
    result = transport.request_json(endpoint or normalize_endpoint('https://api.example/v1', 'remote'),
        'GET', 'models', api_key=SENTINEL, timeout_seconds=10, max_response_bytes=limit)
    return result, sock, connector


def test_pins_validated_address_preserves_host_and_sni_and_ignores_proxy_env(api, monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:6666')
    monkeypatch.setenv('ALL_PROXY', 'http://127.0.0.1:6666')
    resolutions = []
    def resolver(host, port):
        resolutions.append((host, port))
        return ['93.184.216.34'] if len(resolutions) == 1 else ['127.0.0.1']
    sock = ScriptedSocket(response())
    connector = Connector(sock)
    result = api.SafeTransport(resolver=resolver, connector=connector).request_json(
        normalize_endpoint('https://api.example:8443/v1', 'remote'), 'GET', 'models',
        api_key=SENTINEL, timeout_seconds=10, max_response_bytes=1048576)
    assert result == {'data': []}
    assert resolutions == [('api.example', 8443)]
    assert connector.calls[0][0:2] == ('93.184.216.34', 8443)
    assert connector.calls[0][2]['tls_server_name'] == 'api.example'
    assert b'GET /v1/models HTTP/1.1\r\n' in sock.sent
    assert b'Host: api.example:8443\r\n' in sock.sent
    assert b'Authorization: Bearer ' + SENTINEL.encode() in sock.sent
    assert sock.closed


def test_response_wait_uses_remaining_request_deadline_not_connect_cap(api, monkeypatch):
    import ainovel.providers.safe_transport as module
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    class SlowSocket(ScriptedSocket):
        def recv(self, size):
            if self.response:
                # A valid non-streaming model may take more than ten seconds.
                assert self.timeouts[-1] > 10
                clock[0] += 15
            return super().recv(size)
    sock = SlowSocket(response())
    connector = Connector(sock)
    result = api.SafeTransport(resolver=lambda *args: ['93.184.216.34'], connector=connector).request_json(
        normalize_endpoint('https://api.example/v1', 'remote'), 'POST', 'chat/completions',
        api_key=SENTINEL, payload={}, timeout_seconds=120, max_response_bytes=1024)
    assert result == {'data': []}
    assert connector.calls[0][2]['timeout_seconds'] <= 10
    assert sock.closed


def test_response_wait_still_obeys_total_deadline(api, monkeypatch):
    import ainovel.providers.safe_transport as module
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    class ExpiredSocket(ScriptedSocket):
        def recv(self, size):
            clock[0] += 121
            return super().recv(size)
    sock = ExpiredSocket(response())
    with pytest.raises(ProviderTimeout):
        api.SafeTransport(resolver=lambda *args: ['93.184.216.34'], connector=Connector(sock)).request_json(
            normalize_endpoint('https://api.example/v1', 'remote'), 'POST', 'chat/completions',
            api_key=SENTINEL, payload={}, timeout_seconds=120, max_response_bytes=1024)
    assert sock.closed


def test_http_failure_preserves_only_status_not_body(api):
    from ainovel.providers.diagnostics import ResponseFailure, safe_failure_detail
    with pytest.raises(ResponseFailure) as caught:
        call(api, response(SENTINEL.encode(), status=b'429 Too Many Requests'))
    assert caught.value.http_status == 429
    assert '429' in safe_failure_detail(caught.value)
    assert SENTINEL not in str(caught.value) + safe_failure_detail(caught.value)


def test_mixed_dns_rejection_never_connects_or_sends_key(api):
    sock = ScriptedSocket(response())
    connector = Connector(sock)
    with pytest.raises(ProviderError):
        api.SafeTransport(resolver=lambda h, p: ['93.184.216.34', '169.254.169.254'], connector=connector).request_json(
            normalize_endpoint('https://api.example', 'remote'), 'GET', 'models', api_key=SENTINEL,
            timeout_seconds=10, max_response_bytes=1024)
    assert connector.calls == [] and sock.sent == b''


@pytest.mark.parametrize('wire', [
    response(b'credential=' + SENTINEL.encode(), b'Location: https://other.example\r\n', b'302 Found'),
    response(b'credential=' + SENTINEL.encode(), status=b'401 Unauthorized'),
    response(b'credential=' + SENTINEL.encode(), status=b'500 Error'),
    response(b'bad json ' + SENTINEL.encode()), response(b'[]'), response(b'{"n":NaN}'),
    response(b'{}', b'Content-Encoding: gzip\r\n'),
    response(b'{}', b'Content-Length: 9999999\r\n'),
    response(b'{}', b'Content-Length: 20\r\n'),
    response(b'{}', b'Content-Length: 2\r\nContent-Length: 2\r\n'),
    response(b'{}', b'Transfer-Encoding: gzip\r\n'),
    response(b'0\r\n\r\n', b'Transfer-Encoding: chunked\r\nContent-Length: 5\r\n'),
    response(b'Z\r\nsecret\r\n', b'Transfer-Encoding: chunked\r\n'),
    response(b'4\r\n{}', b'Transfer-Encoding: chunked\r\n'),
])
def test_rejects_bad_http_and_json_and_closes_without_error_body(api, wire, caplog):
    sock = ScriptedSocket(wire)
    connector = Connector(sock)
    with pytest.raises(ProviderError) as caught:
        api.SafeTransport(resolver=lambda h, p: ['93.184.216.34'], connector=connector).request_json(
            normalize_endpoint('https://api.example', 'remote'), 'GET', 'models', api_key=SENTINEL,
            timeout_seconds=10, max_response_bytes=1024)
    assert sock.closed and len(connector.calls) == 1
    assert SENTINEL not in str(caught.value) + caplog.text


@pytest.mark.parametrize('headers,body', [
    (b'', b'{"value":"' + b'x' * 200 + b'"}'),
    (b'Transfer-Encoding: chunked\r\n', b'd2\r\n' + b'x' * 210 + b'\r\n0\r\n\r\n'),
])
def test_bounds_body_even_without_content_length(api, headers, body):
    with pytest.raises(ProviderError):
        call(api, response(body, headers), limit=128)


def test_valid_chunked_json_is_decoded_with_bounded_reader(api):
    result, sock, _ = call(api, response(b'b\r\n{"data":[]}\r\n0\r\n\r\n', b'Transfer-Encoding: chunked\r\n'))
    assert result == {'data': []} and sock.closed


def test_default_connector_never_resolves_and_enforces_verified_tls(api, monkeypatch):
    raw = ScriptedSocket(b'')
    connected = []
    raw.connect = connected.append
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('connector must not resolve'))
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: raw)
    contexts = []
    original = ssl.create_default_context
    def context():
        ctx = original()
        contexts.append((ctx.check_hostname, ctx.verify_mode))
        def wrap(sock, *, server_hostname):
            assert sock is raw and server_hostname == 'api.example'
            return sock
        ctx.wrap_socket = wrap
        return ctx
    monkeypatch.setattr(ssl, 'create_default_context', context)
    result = api.PinnedConnector().connect('93.184.216.34', 443, timeout_seconds=10, tls_server_name='api.example')
    assert result is raw and connected == [('93.184.216.34', 443)]
    assert contexts == [(True, ssl.CERT_REQUIRED)]


def test_certificate_failure_closes_socket_without_retry(api, monkeypatch):
    raw = ScriptedSocket(b'')
    raw.connect = lambda address: None
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: raw)
    class TLS:
        def wrap_socket(self, *a, **k):
            raise ssl.SSLCertVerificationError(SENTINEL)
    monkeypatch.setattr(ssl, 'create_default_context', lambda: TLS())
    connector = api.PinnedConnector()
    with pytest.raises(ProviderError) as caught:
        api.SafeTransport(resolver=lambda h, p: ['93.184.216.34'], connector=connector).request_json(
            normalize_endpoint('https://api.example', 'remote'), 'GET', 'models', api_key=SENTINEL,
            timeout_seconds=10, max_response_bytes=1024)
    assert raw.closed and not raw.sent and SENTINEL not in str(caught.value)


@pytest.mark.parametrize('phase', ['dns', 'headers', 'body'])
def test_total_deadline_bounds_dns_slow_headers_and_body(api, phase):
    if phase == 'dns':
        def resolver(h, p):
            time.sleep(0.3)
            return ['93.184.216.34']
        transport = api.SafeTransport(resolver=resolver)
        endpoint = normalize_endpoint('https://api.example', 'remote')
        cleanup = lambda: None
    else:
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        endpoint = normalize_endpoint(f'http://127.0.0.1:{listener.getsockname()[1]}', 'loopback')
        def serve():
            conn, _ = listener.accept()
            with conn:
                conn.recv(8192)
                wire = response(b'{"data":[]}', b'Content-Length: 11\r\n')
                if phase == 'body':
                    head, wire = wire.split(b'\r\n\r\n')
                    conn.sendall(head + b'\r\n\r\n')
                for byte in wire:
                    try:
                        conn.sendall(bytes([byte]))
                    except OSError:
                        break
                    time.sleep(0.03)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        cleanup = listener.close
        transport = api.SafeTransport(resolver=lambda h, p: ['127.0.0.1'])
    start = time.monotonic()
    try:
        with pytest.raises(ProviderTimeout):
            transport.request_json(endpoint, 'GET', 'models', api_key=None, timeout_seconds=0.08, max_response_bytes=1024)
    finally:
        cleanup()
    assert time.monotonic() - start < 0.25


@pytest.mark.parametrize('method,path,key', [('GET','../models',None), ('GET','https://evil.example',None), ('DELETE','models',None), ('GET','models','key\r\nX: y')])
def test_request_injection_is_rejected_before_network(api, method, path, key):
    transport = api.SafeTransport(resolver=lambda *a: pytest.fail('must not resolve'))
    with pytest.raises(ProviderError):
        transport.request_json(normalize_endpoint('https://api.example', 'remote'), method, path,
            api_key=key, timeout_seconds=10, max_response_bytes=1024)


@pytest.mark.parametrize('wire', [
    response(b'{}', b'Invalid header line\r\n'),
    response(b'{"value":1e999}'),
    response(b'2\r\n{}XX0\r\n\r\n', b'Transfer-Encoding: chunked\r\n'),
])
def test_rejects_malformed_headers_chunk_delimiters_and_numeric_overflow(api, wire):
    with pytest.raises(ProviderError):
        call(api, wire)


def test_cleanup_error_cannot_expose_secret_or_override_success(api):
    sock = ScriptedSocket(response())
    def close():
        sock.closed = True
        raise OSError(SENTINEL)
    sock.close = close
    result = api.SafeTransport(resolver=lambda h, p: ['93.184.216.34'], connector=Connector(sock)).request_json(
        normalize_endpoint('https://api.example', 'remote'), 'GET', 'models', api_key=SENTINEL,
        timeout_seconds=10, max_response_bytes=1024)
    assert result == {'data': []} and sock.closed


@pytest.mark.parametrize('body', [b'<html>gateway error</html>', b'```json\n{}\n```', b'[]'])
def test_invalid_outer_json_has_transport_specific_reason(api, body):
    from ainovel.providers.diagnostics import ResponseFailure
    with pytest.raises(ResponseFailure) as caught:
        call(api, response(body))
    assert caught.value.reason.value == 'transport_json'


@pytest.mark.parametrize('status,body,reason', [
    (b'429 Too Many Requests', b'{"error":{"code":"insufficient_quota"}}', 'provider_quota'),
    (b'429 Too Many Requests', b'{"error":{"code":"rate_limit_exceeded"}}', 'provider_rate_limit'),
    (b'200 OK', b'{"error":{"message":"Insufficient balance"}}', 'provider_quota'),
    (b'503 Unavailable', b'not json', 'provider_temporary'),
])
def test_transport_classifies_provider_error_before_content(api, status, body, reason):
    from ainovel.providers.diagnostics import ResponseFailure
    with pytest.raises(ResponseFailure) as caught:
        call(api, response(body, status=status))
    assert caught.value.reason.value == reason


def test_production_connector_shares_connect_tls_deadline(api, monkeypatch):
    raw = ScriptedSocket(b'')
    raw.connect = lambda address: time.sleep(0.06)
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: raw)
    tls_timeouts = []
    class TLS:
        def wrap_socket(self, sock, *, server_hostname):
            tls_timeouts.append(sock.timeouts[-1])
            raise TimeoutError('synthetic handshake timeout')
    monkeypatch.setattr(ssl, 'create_default_context', lambda: TLS())
    with pytest.raises(TimeoutError):
        api.PinnedConnector().connect('93.184.216.34', 443, timeout_seconds=0.15, tls_server_name='api.example')
    assert raw.closed and len(tls_timeouts) == 1 and 0 < tls_timeouts[0] < 0.12

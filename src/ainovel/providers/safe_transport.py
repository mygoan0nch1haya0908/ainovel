"""One-shot, proxy-free HTTP with pinned addresses and verified TLS.

Only the resolver and connector are injectable trust boundaries. Neither receives
credentials. Production uses numeric socket.connect, never create_connection.
"""
from __future__ import annotations

from contextlib import suppress
import http.client
import io
import ipaddress
import json
import math
import queue
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

from ainovel.providers.contracts import (
    ProviderAuthenticationError, ProviderError, ProviderProtocolError,
    ProviderTimeout, ProviderUnavailable,
)
from ainovel.providers.endpoint_policy import Endpoint, normalize_endpoint, resolve_endpoint

_DNS_SLOTS = threading.BoundedSemaphore(4)


def _remaining(deadline: float, cap: float = 10.0) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderTimeout('model request timed out')
    return min(cap, remaining)


def _reject_constant(value):
    raise ValueError('invalid JSON constant')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('invalid JSON number')
    return result


def parse_json_object(data: str | bytes) -> dict:
    result = json.loads(data, parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(result, dict):
        raise ValueError('JSON object required')
    return result


class PinnedConnector:
    def connect(self, address: str, port: int, *, timeout_seconds: float,
                tls_server_name: str | None):
        numeric = ipaddress.ip_address(address)
        deadline = time.monotonic() + min(timeout_seconds, 10.0)
        sock = socket.socket(socket.AF_INET6 if numeric.version == 6 else socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect((str(numeric), port))
            if tls_server_name is not None:
                context = ssl.create_default_context()
                sock.settimeout(_remaining(deadline))
                sock = context.wrap_socket(sock, server_hostname=tls_server_name)
            _remaining(deadline)
            return sock
        except BaseException:
            with suppress(Exception):
                sock.close()
            raise


class _DeadlineReader(io.RawIOBase):
    def __init__(self, sock, deadline):
        self._sock = sock
        self._deadline = deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        self._sock.settimeout(_remaining(self._deadline))
        data = self._sock.recv(len(buffer))
        _remaining(self._deadline)
        buffer[:len(data)] = data
        return len(data)


class _ResponseSocket:
    def __init__(self, sock, deadline):
        self._sock, self._deadline = sock, deadline

    def makefile(self, mode):
        return io.BufferedReader(_DeadlineReader(self._sock, self._deadline))


def _chunked_body(stream, deadline, limit):
    # HTTPResponse discards chunk delimiters without validating CRLF. Parse
    # framing explicitly so malformed chunks cannot be accepted as valid JSON.
    chunks = []
    size = 0
    while True:
        _remaining(deadline)
        line = stream.readline(8193)
        number = line.partition(b';')[0].removesuffix(b'\r\n')
        if (len(line) > 8192 or not line.endswith(b'\r\n') or not number
                or any(byte not in b'0123456789abcdefABCDEF' for byte in number)):
            raise ProviderProtocolError('model returned invalid chunk framing')
        length = int(number, 16)
        if length == 0:
            trailer_size = 0
            for _ in range(100):
                _remaining(deadline)
                trailer = stream.readline(8193)
                trailer_size += len(trailer)
                if len(trailer) > 8192 or trailer_size > 65536 or not trailer.endswith(b'\r\n'):
                    raise ProviderProtocolError('model returned invalid chunk framing')
                if trailer == b'\r\n':
                    return b''.join(chunks)
                if b':' not in trailer:
                    raise ProviderProtocolError('model returned invalid chunk framing')
            raise ProviderProtocolError('model returned excessive chunk trailers')
        if size + length > limit:
            raise ProviderProtocolError('model response exceeds allowed size')
        while length:
            _remaining(deadline)
            chunk = stream.read(min(65536, length))
            if not chunk:
                raise ProviderProtocolError('model response is incomplete')
            chunks.append(chunk)
            size += len(chunk)
            length -= len(chunk)
        if stream.read(2) != b'\r\n':
            raise ProviderProtocolError('model returned invalid chunk framing')


class SafeTransport:
    def __init__(self, *, resolver=None, connector=None):
        self._resolver = resolver
        self._connector = connector if connector is not None else PinnedConnector()

    def _resolve(self, endpoint, deadline):
        # OS DNS has no portable per-call cancellation. Bound waiting and the
        # number of daemon workers; workers hold no key/socket and never dispatch.
        if not _DNS_SLOTS.acquire(timeout=_remaining(deadline)):
            raise ProviderTimeout('model address resolution timed out')
        result = queue.Queue(maxsize=1)
        def resolve():
            try:
                result.put((True, resolve_endpoint(endpoint, resolver=self._resolver)))
            except Exception:
                result.put((False, None))
            finally:
                _DNS_SLOTS.release()
        try:
            worker = threading.Thread(target=resolve, daemon=True)
            worker.start()
        except Exception:
            _DNS_SLOTS.release()
            raise
        try:
            valid, addresses = result.get(timeout=_remaining(deadline))
        except queue.Empty:
            raise ProviderTimeout('model address resolution timed out') from None
        _remaining(deadline)
        if not valid:
            raise ProviderUnavailable('model endpoint is not allowed or unavailable')
        return addresses

    def request_json(self, endpoint: Endpoint, method: str, relative_path: str, *,
                     api_key: str | None, payload=None, timeout_seconds: float,
                     max_response_bytes: int) -> dict:
        sock = response = None
        try:
            if (not isinstance(endpoint, Endpoint)
                    or normalize_endpoint(endpoint.base_url, endpoint.kind) != endpoint
                    or (method, relative_path) not in {('GET', 'models'), ('POST', 'chat/completions')}
                    or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                    or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
                    or type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 2097152
                    or (api_key is not None and (not isinstance(api_key, str) or len(api_key) > 8192
                        or any(ord(c) < 33 or ord(c) > 126 for c in api_key)))):
                raise ProviderProtocolError('invalid model request configuration')
            deadline = time.monotonic() + min(timeout_seconds, 180.0)
            body = b'' if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
            authority = urlsplit(endpoint.base_url).netloc
            headers = [f'{method} {endpoint.path}/{relative_path} HTTP/1.1',
                       f'Host: {authority}', 'Accept: application/json', 'Accept-Encoding: identity', 'Connection: close']
            if api_key:
                headers.append(f'Authorization: Bearer {api_key}')
            if payload is not None:
                headers.extend(['Content-Type: application/json', f'Content-Length: {len(body)}'])
            wire = ('\r\n'.join(headers) + '\r\n\r\n').encode('ascii') + body
            addresses = self._resolve(endpoint, deadline)
            sock = self._connector.connect(addresses[0], endpoint.port, timeout_seconds=_remaining(deadline),
                tls_server_name=endpoint.host if endpoint.base_url.startswith('https:') else None)
            sock.settimeout(_remaining(deadline))
            sock.sendall(wire)
            _remaining(deadline)
            response = http.client.HTTPResponse(_ResponseSocket(sock, deadline), method=method)
            response.begin()
            _remaining(deadline)
            if response.status in (401, 403):
                raise ProviderAuthenticationError('model authentication failed')
            if not 200 <= response.status < 300:
                raise ProviderProtocolError('model returned an unsuccessful HTTP response')
            lengths = response.headers.get_all('Content-Length', [])
            encodings = response.headers.get_all('Transfer-Encoding', [])
            if (response.headers.defects or response.headers.get_all('Content-Encoding') or len(lengths) > 1 or len(encodings) > 1
                    or (lengths and encodings) or (encodings and encodings[0].lower() != 'chunked')):
                raise ProviderProtocolError('model returned unsupported HTTP framing')
            if lengths and (not lengths[0].isascii() or not lengths[0].isdigit()
                            or int(lengths[0]) > max_response_bytes):
                raise ProviderProtocolError('model response exceeds allowed size or has invalid framing')
            chunks = []
            size = 0
            if response.chunked:
                chunks.append(_chunked_body(response.fp, deadline, max_response_bytes))
            while not response.chunked:
                _remaining(deadline)
                chunk = response.read1(min(65536, max_response_bytes + 1 - size))
                if not chunk:
                    if response.length not in (None, 0):
                        raise ProviderProtocolError('model response is incomplete')
                    break
                size += len(chunk)
                if size > max_response_bytes:
                    raise ProviderProtocolError('model response exceeds allowed size')
                chunks.append(chunk)
            result = parse_json_object(b''.join(chunks).decode('utf-8'))
            _remaining(deadline)
            return result
        except ProviderError:
            raise
        except (TimeoutError, socket.timeout):
            raise ProviderTimeout('model request timed out') from None
        except (ValueError, UnicodeError, http.client.HTTPException, RecursionError):
            raise ProviderProtocolError('model returned an invalid response') from None
        except Exception:
            raise ProviderUnavailable('model service is unavailable') from None
        finally:
            if response is not None:
                with suppress(Exception):
                    response.close()
            if sock is not None:
                with suppress(Exception):
                    sock.close()

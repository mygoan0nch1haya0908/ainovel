import importlib

import pytest

from ainovel.providers.endpoint_policy import EndpointError, normalize_endpoint


def resolve(endpoint, answers):
    module = importlib.import_module('ainovel.providers.endpoint_policy')
    assert hasattr(module, 'resolve_endpoint'), 'validated dispatch resolution is missing'
    return module.resolve_endpoint(endpoint, resolver=lambda host, port: answers)


@pytest.mark.parametrize('address', [
    '127.0.0.1', '10.1.2.3', '172.16.1.1', '192.168.1.1', '169.254.169.254',
    '0.0.0.0', '100.100.100.200', '224.0.0.1', '240.0.0.1', '192.0.2.1',
    '::', '::1', 'fe80::1', 'fc00::1', 'ff02::1', '2001:db8::1',
    '::ffff:127.0.0.1', '::ffff:93.184.216.34', 'not-an-ip',
    '2002:7f00:1::', '64:ff9b::a9fe:a9fe', '64:ff9b:1::a9fe:a9fe',
])
def test_remote_rejects_every_unsafe_dns_answer_including_mixed(address):
    endpoint = normalize_endpoint('https://api.example/v1', 'remote')
    with pytest.raises(EndpointError):
        resolve(endpoint, ['93.184.216.34', address])


def test_public_answers_are_numeric_and_deduplicated():
    endpoint = normalize_endpoint('https://api.example/v1', 'remote')
    assert resolve(endpoint, ['93.184.216.34', '2606:4700:4700::1111', '93.184.216.34']) == (
        '93.184.216.34', '2606:4700:4700::1111')


@pytest.mark.parametrize('answers', [[], ['127.0.0.1', '10.0.0.1'], ['93.184.216.34']])
def test_loopback_never_accepts_other_networks(answers):
    with pytest.raises(EndpointError):
        resolve(normalize_endpoint('http://localhost:11434/v1', 'loopback'), answers)


def test_loopback_accepts_only_loopback_answers():
    assert resolve(normalize_endpoint('http://localhost:11434', 'loopback'), ['127.0.0.1', '::1']) == ('127.0.0.1', '::1')


@pytest.mark.parametrize('url', [
    'https://user:secret@example.com/v1', 'https://example.com?key=secret',
    'https://example.com#secret', 'https://example.com:0', 'https://example.com:65536',
    'https://example.com:', 'https://example.com:abc', 'https://example.com/%2e%2e',
])
def test_endpoint_syntax_rejected_without_dns(url):
    with pytest.raises(EndpointError):
        normalize_endpoint(url, 'remote')

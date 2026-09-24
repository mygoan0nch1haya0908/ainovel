from dataclasses import asdict, replace
import importlib
import importlib.util
import json

import pytest

from ainovel.agents.contracts import ChapterSummaryDelta
from ainovel.agents.runner import AgentRunner
from ainovel.providers.contracts import ModelRequest, ProviderError
from ainovel.providers.diagnostics import ResponseFailure
from ainovel.providers.endpoint_policy import normalize_endpoint

SENTINEL = 'FAKE_COMPATIBLE_KEY_42'


@pytest.fixture
def api():
    class API:
        def __getattr__(self, name):
            assert importlib.util.find_spec('ainovel.providers.compatible'), 'compatible provider is missing'
            return getattr(importlib.import_module('ainovel.providers.compatible'), name)
    return API()


class Transport:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def request_json(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def reply(content='{"summary":"valid","state_delta":{"chapter":1}}', finish='stop'):
    return {'id': 'chatcmpl-1', 'usage': {'prompt_tokens': 37, 'completion_tokens': 14},
        'choices': [{'finish_reason': finish, 'message': {'content': content}}]}


def provider(api, result, **kwargs):
    transport = Transport(result)
    return api.CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'), 'model-a',
        api_key=SENTINEL, transport=transport, allow_real_calls=True, **kwargs), transport


def request():
    return ModelRequest('model-a', 'Summarize.', {'chapter': 'synthetic'}, ChapterSummaryDelta.model_json_schema(),
        1000, 256, 5.0, {})


def test_local_diagnosis_and_capabilities_never_dispatch(api):
    p, t = provider(api, RuntimeError('network forbidden'))
    assert p.diagnose().available
    assert p.diagnose().models == ('model-a',)
    caps = p.capabilities('model-a')
    assert caps.context_window == 32000 and caps.max_output_tokens == 12000
    assert not caps.strict_structured_output and caps.real_calls_allowed
    assert t.calls == []


def test_generic_wire_uses_schema_json_object_no_vendor_options_and_local_validation(api):
    p, t = provider(api, reply())
    result = AgentRunner().run_with_response(p, request(), ChapterSummaryDelta)
    assert result.result.summary == 'valid'
    args, kwargs = t.calls[0]
    assert args[1:] == ('POST', 'chat/completions')
    payload = kwargs['payload']
    assert set(payload) == {'model', 'messages', 'response_format', 'max_tokens'}
    assert payload['response_format'] == {'type': 'json_object'} and payload['max_tokens'] == 256
    assert json.dumps(request().output_schema, ensure_ascii=False, separators=(',', ':')) in payload['messages'][0]['content']
    assert kwargs['max_response_bytes'] == 2097152 and kwargs['timeout_seconds'] == 5
    assert SENTINEL not in json.dumps(payload)
    assert result.response.input_tokens == 37 and result.response.output_tokens == 14
    p, _ = provider(api, reply('{"summary":12,"state_delta":{}}'))
    with pytest.raises(ResponseFailure) as caught:
        AgentRunner().run(p, request(), ChapterSummaryDelta)
    assert caught.value.reason.value == 'schema_mismatch'


@pytest.mark.parametrize('content,finish,reason', [('not JSON','stop','response_json'), ('{}','length','response_truncated'), ('{}','content_filter','response_refused')])
def test_failures_keep_only_safe_usage(api, content, finish, reason):
    p, _ = provider(api, reply(content, finish))
    with pytest.raises(ResponseFailure) as caught:
        p.generate(request())
    assert caught.value.reason.value == reason
    assert caught.value.response.input_tokens == 37 and caught.value.response.output_tokens == 14
    assert caught.value.response.text is None and caught.value.response.structured is None


@pytest.mark.parametrize('location', ['id','content','escaped_content','nested','usage','refusal','unknown'])
def test_echoed_key_is_rejected_before_result_or_error_metadata_can_be_saved(api, location, caplog):
    data = reply()
    if location == 'id':
        data['id'] = 'id-' + SENTINEL
    elif location == 'usage':
        data['usage']['prompt_tokens'] = SENTINEL
    elif location == 'refusal':
        data['choices'][0]['message']['refusal'] = SENTINEL
    elif location == 'unknown':
        data['unexpected'] = {'value': SENTINEL}
    else:
        value = json.dumps({'summary': SENTINEL, 'state_delta': {'nested': SENTINEL}})
        if location == 'escaped_content':
            value = value.replace('F', '\\u0046')
        data['choices'][0]['message']['content'] = value
    p, _ = provider(api, data)
    with pytest.raises(ProviderError) as caught:
        p.generate(request())
    assert SENTINEL not in str(caught.value) + repr(getattr(caught.value, 'response', None)) + caplog.text


@pytest.mark.parametrize('field,value', [('id','<script>oops</script>'), ('id','x'*256), ('prompt_tokens',True), ('completion_tokens',-1), ('prompt_tokens',10**50)])
def test_invalid_metadata_is_not_persistable(api, field, value):
    data = reply()
    if field == 'id':
        data['id'] = value
    else:
        data['usage'][field] = value
    p, _ = provider(api, data)
    with pytest.raises(ResponseFailure) as caught:
        p.generate(request())
    safe = asdict(caught.value.response)
    assert value not in [safe['provider_response_id'], safe['input_tokens'], safe['output_tokens']]


def test_list_is_bounded_deduplicated_exact_ids_only(api):
    p, t = provider(api, {'data': [{'id': 'a<script>', 'owner': SENTINEL + '-not-displayed'}, {'id':'same'}, {'id':'same'}]})
    # Even ignored fields containing credentials must fail closed.
    with pytest.raises(ProviderError):
        p.list_models()
    t.result = {'data': [{'id':'a<script>', 'owner':'untrusted'}, {'id':'same'}, {'id':'same'}]}
    assert p.list_models() == ('a<script>', 'same')
    assert t.calls[-1][1]['max_response_bytes'] == 1048576
    for data in ([{'id':str(i)} for i in range(201)], [{'id':'x'*256}], [{'id':SENTINEL}], [{'id':None}]):
        t.result = {'data':data}
        with pytest.raises(ProviderError):
            p.list_models()


def test_model_identifier_round_trip_preserves_protocol_value(api):
    p, t = provider(api, {'data': [{'id': 'model<a>'}]})
    model = p.list_models()[0]
    assert model == 'model<a>'
    t.result = reply()
    chosen = api.CompatibleProvider(normalize_endpoint('https://api.example/v1', 'remote'), model,
        api_key=SENTINEL, transport=t, allow_real_calls=True)
    assert chosen.diagnose().models == ('model<a>',)
    chosen.generate(replace(request(), model=model))
    assert t.calls[-1][1]['payload']['model'] == 'model<a>'
    t.result = reply('{"ok":true}')
    assert chosen.test_connection().models == ('model<a>',)


def test_connection_test_is_short_synthetic_and_does_not_accept_novel_context(api):
    p, t = provider(api, reply('{"ok":true}'))
    assert p.test_connection().available
    args, kwargs = t.calls[0]
    assert args[1:] == ('POST', 'chat/completions')
    assert kwargs['payload']['max_tokens'] <= 128
    assert kwargs['max_response_bytes'] == 65536
    assert SENTINEL not in json.dumps(kwargs['payload'])


@pytest.mark.parametrize('value', [1, 1.0, 'true', '1'])
def test_connection_test_requires_literal_boolean_true(api, value):
    p, _ = provider(api, reply(json.dumps({'ok': value})))
    with pytest.raises(ProviderError):
        p.test_connection()


@pytest.mark.parametrize('changes', [{'model':'other'}, {'max_output_tokens':12001}, {'max_input_tokens':32001}, {'timeout_seconds':0}])
def test_invalid_or_unbound_requests_do_not_dispatch(api, changes):
    p, t = provider(api, reply())
    with pytest.raises(ProviderError):
        p.generate(replace(request(), **changes))
    assert t.calls == []


def test_disabled_and_keyless_remote_provider_cannot_dispatch(api):
    t = Transport(reply())
    for key, allowed in [(None, True), (SENTINEL, False)]:
        p = api.CompatibleProvider(normalize_endpoint('https://api.example', 'remote'), 'model-a',
            api_key=key, allow_real_calls=allowed, transport=t)
        assert not p.diagnose().available
        with pytest.raises(ProviderError):
            p.generate(request())
        with pytest.raises(ProviderError):
            p.list_models()
    assert t.calls == []


def test_loopback_allows_keyless_and_errors_are_fixed(api, caplog):
    t = Transport(RuntimeError(SENTINEL))
    p = api.CompatibleProvider(normalize_endpoint('http://localhost:11434/v1', 'loopback'), 'model-a',
        transport=t, allow_real_calls=True)
    assert p.capabilities('model-a').local and p.diagnose().available
    with pytest.raises(ProviderError) as caught:
        p.generate(request())
    assert SENTINEL not in str(caught.value) + caplog.text

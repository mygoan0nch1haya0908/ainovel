import pytest


def test_real_send_window_and_operation_count_survive_sixty_seconds():
    from ainovel.providers import request_diagnostics as telemetry
    assert hasattr(telemetry, 'SendMetrics'), 'Missing actual-send accounting'
    m = telemetry.SendMetrics()
    a = m.start(('url', 'model'), 'op', 10, now=0)
    completed=m.finish(a['llm_call_id'], {'prompt_tokens': 8, 'completion_tokens': 2})
    assert completed['actual_tokens_last_60s']==10
    b = m.start(('url', 'model'), 'op', 20, now=70)
    assert b['actual_requests_last_60s'] == 1
    assert b['actual_requests_last_300s'] == 2
    assert b['user_request_actual_calls'] == 2
    assert b['actual_known_tokens_last_300s'] == 10
    assert b['actual_tokens_last_300s'] is None
    m.finish(b['llm_call_id'], None)


def test_send_budget_prevents_extra_call_and_tracks_cross_model_concurrency():
    from ainovel.providers import request_diagnostics as telemetry
    assert hasattr(telemetry, 'SendMetrics'), 'Missing actual-send accounting'
    m = telemetry.SendMetrics()
    m.start(('url', 'a'), 'op', 10, now=0, budget=2)
    b = m.start(('url', 'b'), 'op', 10, now=1, budget=2)
    assert b['actual_process_concurrency'] == 2
    assert b['actual_model_concurrency'] == 1
    assert b['actual_operation_requests_last_10s']==2
    with pytest.raises(Exception, match='call budget'):
        m.start(('url', 'b'), 'op', 10, now=2, budget=2)


def test_standard_numeric_rate_headers_are_kept_without_secrets():
    from ainovel.providers.llm_diagnostic import safe_headers
    d = safe_headers({'X-RateLimit-Limit':'20','X-RateLimit-Remaining':'0',
        'X-RateLimit-Reset':'12','Authorization':'SECRET'}, api_key='SECRET')
    assert d['rate_limits']['x-ratelimit-limit'] == 20
    assert d['rate_limits']['x-ratelimit-remaining'] == 0
    assert 'SECRET' not in str(d)


def test_probe_suite_stops_on_first_failure():
    import importlib.util
    assert importlib.util.find_spec('ainovel.llm_rate_probe') is not None
    from ainovel.llm_rate_probe import run_suite
    calls=[]; waits=[]
    def probe(size):
        calls.append(size)
        return 1
    assert run_suite(probe, sleep=waits.append)==1
    assert calls==[0] and sum(waits)==70


def test_probe_suite_uses_eight_serial_calls_with_cooldowns():
    import importlib.util
    assert importlib.util.find_spec('ainovel.llm_rate_probe') is not None
    from ainovel.llm_rate_probe import run_suite
    calls=[]; waits=[]
    assert run_suite(lambda size: calls.append(size) or 0, sleep=waits.append)==0
    assert calls==[0,0,0,0,0,0,0,10000]
    assert sum(waits)==285
    assert max(waits)<=30


def test_observed_provider_limitation_keeps_code_type_and_trace():
    from ainovel.providers.request_diagnostics import trace_request, note_response
    from ainovel.providers.llm_diagnostic import diagnose
    from test_compatible_provider import request
    with trace_request(request(),'https://api.example/v1','SECRET') as state:
        note_response({'error':{'code':'429','type':'limitation','param':'',
            'message':'Model model-a rate limited by provider – contact support to request higher concurrency or try again later. (tid: 2026092809225551249056999227535)'}})
        state['http_status']=429
    d=state['diagnostic']
    assert d['provider_error_code']=='429'
    assert d['provider_error_type']=='limitation'
    assert d['trace_id']=='2026092809225551249056999227535'
    assert d['error_layer']=='upstream_provider'
    assert d['confidence']=='medium'
    assert d['error_subtype']=='concurrency_limit'


def test_provider_hint_requires_complete_recognized_envelope():
    from ainovel.providers.request_diagnostics import trace_request, note_response
    from test_compatible_provider import request
    with trace_request(request(),'https://api.example/v1','SECRET') as state:
        note_response({'error':{'message':'PRIVATE NOVEL says contact support to request higher concurrency'}})
        state['http_status']=429
    assert state['diagnostic']['error_layer']=='remote_endpoint'
    assert 'PRIVATE' not in str(state)


def test_actual_send_budget_blocks_before_socket_write(monkeypatch):
    import json
    from ainovel.providers import request_diagnostics as telemetry
    from ainovel.providers.compatible import CompatibleProvider
    from ainovel.providers.safe_transport import SafeTransport
    from ainovel.providers.endpoint_policy import normalize_endpoint
    from test_safe_transport import ScriptedSocket, Connector, response
    from test_compatible_provider import request, reply
    monkeypatch.setattr(telemetry,'send_metrics',telemetry.SendMetrics())
    monkeypatch.setenv('MAX_LLM_CALLS_PER_USER_REQUEST','1')
    token=telemetry.operation_context.set('11111111-1111-1111-1111-111111111111')
    sockets=[]
    class Socket(ScriptedSocket):
        sent_count=0
        def sendall(self,data):
            self.sent_count+=1
            return super().sendall(data)
    def provider():
        sock=Socket(response(json.dumps(reply()).encode()))
        sockets.append(sock)
        return CompatibleProvider(normalize_endpoint('https://api.example/v1','remote'),'model-a',api_key='SECRET',
            transport=SafeTransport(resolver=lambda *a:['93.184.216.34'],connector=Connector(sock)),allow_real_calls=True)
    try:
        provider().generate(request())
        with pytest.raises(Exception) as caught:
            provider().generate(request())
        assert caught.value.diagnostic['error_layer']=='agent'
        assert [s.sent_count for s in sockets]==[1,0]
    finally:
        telemetry.operation_context.reset(token)

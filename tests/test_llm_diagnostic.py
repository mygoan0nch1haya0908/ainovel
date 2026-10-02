import pytest

@pytest.mark.parametrize('status,category', [(401,'authentication'),(403,'permission'),(404,'model_not_found'),(408,'timeout'),(413,'invalid_request'),(422,'invalid_request'),(429,'rate_limit'),(500,'provider'),(502,'provider'),(503,'provider'),(504,'timeout')])
def test_http_categories(status,category):
    from ainovel.providers.llm_diagnostic import diagnose
    assert diagnose({}, {'http_status':status})['error_category'] == category

def test_rate_evidence_not_large_input_alone():
    from ainovel.providers.llm_diagnostic import diagnose
    for state,category,subtype,confidence in [
        ({'http_status':429},'rate_limit','unknown_rate_limit','low'),
        ({'http_status':429,'rate_limits':{'x-ratelimit-remaining-tokens':0}},'rate_limit','tpm_limit','medium'),
        ({'http_status':429,'rate_limits':{'x-ratelimit-remaining-requests':0}},'rate_limit','rpm_limit','medium'),
        ({'http_status':429,'provider_code':'insufficient_quota'},'quota','quota_exhausted','high'),
        ({'provider_code':'insufficient_balance'},'quota','quota_exhausted','high'),
        ({'error_code':'client_timeout_connect'},'timeout','connect_timeout','high'),
        ({'provider_code':'provider_timeout'},'timeout','provider_timeout','high'),
        ({'error_code':'model_content_json'},'format','invalid_json','high'),
        ({'error_code':'schema_mismatch'},'json_schema','schema_mismatch','high'),
        ({'provider_code':'context_length_exceeded'},'context_length','context_length_exceeded','high'),
        ({'error_code':'provider_unavailable'},'connection','connection_failed','low'),
    ]:
        d=diagnose({'estimated_input_tokens':47000},state)
        assert (d['error_category'],d['error_subtype'],d['confidence']) == (category,subtype,confidence)

def test_headers_are_safe_and_retry_date_supported():
    from ainovel.providers.llm_diagnostic import safe_headers
    d=safe_headers({'Retry-After':'12','X-RateLimit-Remaining-Tokens':'0','X-Request-ID':'req-1','x-trace-id':'trace-1','cf-ray':'abc-HKG','authorization':'SECRET','server-timing':'model;dur=12;desc="PRIVATE"'},now=0,api_key='SECRET')
    assert d['retry_after']==12 and d['request_id']=='req-1' and d['trace_id']=='trace-1'
    assert 'PRIVATE' not in str(d) and 'SECRET' not in str(d)
    assert safe_headers({'Retry-After':'Thu, 01 Jan 1970 00:00:12 GMT'},now=0)['retry_after']==12
    assert safe_headers({'Retry-After':'NaN'},now=0)['retry_after'] is None

def test_metrics_concurrency_growth_unknown_usage_expiration():
    from ainovel.providers.llm_metrics import RequestMetrics
    m=RequestMetrics()
    a=m.start('target','op',{'total_chars':100,'estimated_input_tokens':50},now=0)
    b=m.start('target','op',{'total_chars':3000,'estimated_input_tokens':1000},now=1)
    assert b['concurrency']==2 and b['requests_last_60s']==2
    assert b['estimated_tokens_last_60s']==1050 and b['tokens_last_60s'] is None and b['context_growth']
    m.finish(a['metric_id'],{'prompt_tokens':40,'completion_tokens':5},now=2)
    m.finish(b['metric_id'],None,now=3)
    c=m.start('target','op',{'total_chars':100,'estimated_input_tokens':50},now=62)
    assert c['concurrency']==1 and c['requests_last_60s']==1

def test_retry_wait_never_shortens_server_delay():
    from ainovel.providers.llm_diagnostic import retry_delay
    assert retry_delay({'retry_after':12},1,jitter=0.5)==12
    assert retry_delay({},2,jitter=0.5)==2.5
    assert retry_delay({'retry_after':120},1,jitter=0.5) is None

def test_transport_failure_attaches_diagnostic_and_preserves_headers():
    from ainovel.providers.compatible import CompatibleProvider
    from ainovel.providers.safe_transport import SafeTransport
    from ainovel.providers.endpoint_policy import normalize_endpoint
    from test_safe_transport import ScriptedSocket, Connector, response
    from test_compatible_provider import request
    transport=SafeTransport(resolver=lambda *a:['93.184.216.34'],connector=Connector(ScriptedSocket(response(
        b'{"error":{"code":"rate_limit_exceeded","message":"PRIVATE"}}',
        b'Retry-After: 12\r\nX-RateLimit-Remaining-Tokens: 0\r\nX-Request-ID: req-123\r\n',b'429 Too Many Requests'))))
    p=CompatibleProvider(normalize_endpoint('https://api.example/v1','remote'),'model-a',api_key='SECRET',transport=transport,allow_real_calls=True)
    with pytest.raises(Exception) as caught: p.generate(request())
    d=caught.value.diagnostic
    assert d['error_subtype']=='tpm_limit' and d['retry_after']==12 and d['request_id']=='req-123'
    assert d['usage_known'] is False and d['concurrency']>=1
    assert 'PRIVATE' not in str(d) and 'SECRET' not in str(d)

def test_schema_error_keeps_safe_request_diagnostic():
    from ainovel.agents.runner import AgentRunner
    from ainovel.agents.contracts import ChapterSummaryDelta
    from ainovel.providers.compatible import CompatibleProvider
    from ainovel.providers.endpoint_policy import normalize_endpoint
    from test_compatible_provider import request,reply,Transport
    p=CompatibleProvider(normalize_endpoint('https://api.example/v1','remote'),'model-a',api_key='SECRET',transport=Transport(reply('{}')),allow_real_calls=True)
    with pytest.raises(Exception) as caught: AgentRunner().run(p,request(),ChapterSummaryDelta)
    assert caught.value.diagnostic['error_category']=='json_schema'
    assert caught.value.diagnostic['actual_prompt_tokens']==37

def test_health_rate_retry_obeys_retry_after(monkeypatch):
    import ainovel.llm_health as health
    from ainovel.providers.contracts import ModelResponse,ProviderCapabilities
    from ainovel.providers.llm_response import LLMProviderError
    from ainovel.providers.diagnostics import FailureReason
    waits=[]
    monkeypatch.setattr(health.time,'sleep',waits.append)
    class Provider:
        calls=0
        def capabilities(self,m): return ProviderCapabilities(2000,128,False,True,False,True)
        def generate(self,r):
            self.calls+=1
            if self.calls==1:
                e=LLMProviderError(FailureReason.RATE_LIMIT,http_status=429)
                e.diagnostic={'retry_after':12,'http_status':429}
                raise e
            return ModelResponse({'status':'ok'},'{"status":"ok"}','r',10,5,1)
    p=Provider()
    assert health.probe(p,'model',retries=1)=={'status':'ok'}
    assert waits==[12] and p.calls==2

def test_health_long_retry_after_pauses_without_early_retry(monkeypatch):
    import ainovel.llm_health as health
    from ainovel.providers.contracts import ProviderTimeout,ProviderCapabilities
    class Provider:
        calls=0
        def capabilities(self,m): return ProviderCapabilities(2000,128,False,True,False,True)
        def generate(self,r):
            self.calls+=1
            e=ProviderTimeout('timeout');e.diagnostic={'retry_after':120};raise e
    monkeypatch.setattr(health.time,'sleep',lambda _:pytest.fail('must pause'))
    p=Provider()
    with pytest.raises(ProviderTimeout): health.probe(p,'model',retries=2)
    assert p.calls==1

def test_sdk_status_preserves_safe_headers_and_permission():
    from types import SimpleNamespace
    import httpx2
    from openai import APIStatusError
    from ainovel.providers.llm_response import raise_sdk_status_error
    from ainovel.providers.contracts import ProviderAuthenticationError
    error=APIStatusError('PRIVATE',response=httpx2.Response(403,headers={'x-request-id':'req-403'},request=httpx2.Request('POST','https://example/v1')),body={'message':'PRIVATE'})
    with pytest.raises(ProviderAuthenticationError) as caught:
        raise_sdk_status_error(error,model='model',client=SimpleNamespace(api_key='SECRET',base_url='https://example/v1'))
    assert caught.value.diagnostic['error_category']=='permission'
    assert caught.value.diagnostic['request_id']=='req-403'
    assert 'PRIVATE' not in str(caught.value.diagnostic)

def test_quota_from_text_is_not_high_confidence_provider_fact():
    from ainovel.providers.llm_diagnostic import diagnose
    d=diagnose({}, {'error_code':'provider_quota'})
    assert d['confidence'] != 'high'
    assert any('文本' in e for e in d['evidence'])

def test_metrics_completion_updates_usage_and_exposes_truncation():
    from ainovel.providers.llm_metrics import RequestMetrics
    from ainovel.providers.llm_diagnostic import diagnose
    m=RequestMetrics()
    a=m.start('target','op',{'total_chars':100,'estimated_input_tokens':50},now=0)
    updated=m.finish(a['metric_id'],{'prompt_tokens':40,'completion_tokens':5},now=1)
    assert updated['known_tokens_last_60s']==45 and updated['unknown_usage_requests']==0
    assert diagnose({'metrics_truncated':True},{})['metrics_truncated'] is True

@pytest.mark.parametrize('kind',['openai','qwen'])
def test_sdk_success_and_schema_error_keep_usage_and_context(kind,monkeypatch):
    import ainovel.providers.request_diagnostics as telemetry
    from ainovel.providers.llm_metrics import RequestMetrics
    monkeypatch.setattr(telemetry,'metrics',RequestMetrics())
    from types import SimpleNamespace as NS
    from ainovel.providers.openai import OpenAIProvider
    from ainovel.providers.qwen import QwenProvider
    from ainovel.agents.runner import AgentRunner
    from ainovel.agents.contracts import ChapterSummaryDelta
    from test_compatible_provider import request
    result=NS(id='resp-1',error=None,output_text='{}',usage=NS(input_tokens=37,output_tokens=14,prompt_tokens=37,completion_tokens=14),
        choices=[NS(finish_reason='stop',message=NS(content='{}',refusal=None))])
    client=NS(api_key='SECRET',base_url='https://example/v1',responses=NS(create=lambda **kw:result),chat=NS(completions=NS(create=lambda **kw:result)))
    provider=(OpenAIProvider if kind=='openai' else QwenProvider)(client,allow_real_calls=True)
    from dataclasses import replace
    req=replace(request(),metadata={'schema_name':'summary'})
    with pytest.raises(Exception) as caught: AgentRunner().run(provider,req,ChapterSummaryDelta)
    d=caught.value.diagnostic
    assert d['error_category']=='json_schema' and d['actual_prompt_tokens']==37 and d['message_count']==2
    assert d['unknown_usage_requests']==0

def test_sdk_endpoint_diagnostic_removes_url_credentials_and_query():
    from types import SimpleNamespace
    from ainovel.providers.llm_response import attach_sdk_diagnostic
    from ainovel.providers.contracts import ProviderTimeout
    error=ProviderTimeout('timeout')
    result=attach_sdk_diagnostic(error,error,model='model',client=SimpleNamespace(api_key='KEY',base_url='https://user:SECRET@example/v1?token=PRIVATE'))
    assert result.diagnostic['base_url']=='https://example/v1'

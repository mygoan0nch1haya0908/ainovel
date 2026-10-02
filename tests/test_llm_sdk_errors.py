from types import SimpleNamespace
import httpx2
import pytest
from openai import APIStatusError
from ainovel.providers.openai import OpenAIProvider
from ainovel.providers.qwen import QwenProvider
from ainovel.providers.contracts import ModelRequest, ProviderAuthenticationError
from ainovel.providers.diagnostics import ResponseFailure


@pytest.mark.parametrize('provider_type', [OpenAIProvider, QwenProvider])
def test_sdk_success_status_error_envelope_is_not_model_format(provider_type):
    from ainovel.providers.llm_response import LLMQuotaError
    result = SimpleNamespace(error=SimpleNamespace(code='insufficient_quota', message='SECRET', type=None), output_text='', choices=[])
    client = SimpleNamespace(api_key='SECRET', responses=SimpleNamespace(create=lambda **kwargs: result),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: result)))
    with pytest.raises(LLMQuotaError):
        provider_type(client, allow_real_calls=True).generate(ModelRequest('model','test',{}, {},100,128,5,{'schema_name':'test'}))


def test_production_sdk_does_not_retry_quota_http(monkeypatch):
    import ainovel.app as app_module
    from openai import OpenAI
    from ainovel.config import Settings
    from ainovel.providers.llm_response import LLMQuotaError
    calls = []
    def handle(request):
        calls.append(request)
        return httpx2.Response(429, json={'error':{'type':'insufficient_quota','message':'exhausted'}})
    with httpx2.Client(transport=httpx2.MockTransport(handle)) as http_client:
        monkeypatch.setattr(app_module, 'OpenAI', lambda **kwargs: OpenAI(http_client=http_client, **kwargs))
        settings = Settings(_env_file=None, openai_api_key='SYNTHETIC', openai_base_url='https://example/v1', allow_real_openai=True)
        provider = app_module._default_provider_registry(settings).get('openai')
        with pytest.raises(LLMQuotaError):
            provider.generate(ModelRequest('model','test',{}, {},100,128,5,{'schema_name':'test'}))
    assert len(calls) == 1


@pytest.mark.parametrize('provider_type', [OpenAIProvider, QwenProvider])
@pytest.mark.parametrize('status,body,reason', [
    (429, {'code': 'insufficient_quota'}, 'provider_quota'),
    (429, {'code': 'rate_limit_exceeded'}, 'provider_rate_limit'),
    (503, {}, 'provider_temporary'),
    (400, {'message':'bad request'}, 'provider_api'),
    (403, {}, 'auth'),
])
def test_sdk_status_classification(provider_type, status, body, reason):
    calls = []
    error = APIStatusError('PRIVATE', response=httpx2.Response(status, request=httpx2.Request('POST','https://example/v1')), body=body)
    def create(**kwargs):
        calls.append(kwargs)
        raise error
    client = SimpleNamespace(api_key='SECRET', responses=SimpleNamespace(create=create), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = provider_type(client, allow_real_calls=True)
    req = ModelRequest('model', 'test', {}, {}, 100, 128, 5, {'schema_name':'test'})
    with pytest.raises(ProviderAuthenticationError if reason == 'auth' else ResponseFailure) as caught:
        provider.generate(req)
    if reason != 'auth':
        assert caught.value.reason.value == reason
    assert 'PRIVATE' not in str(caught.value) and len(calls) == 1

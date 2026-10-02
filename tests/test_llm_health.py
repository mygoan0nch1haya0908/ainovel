import importlib
from types import SimpleNamespace
import pytest
from ainovel.providers.contracts import ModelResponse, ProviderCapabilities


def test_probe_sends_only_synthetic_data_and_calls_once():
    health = importlib.import_module('ainovel.llm_health')
    calls = []
    def generate(request):
        calls.append(request)
        return ModelResponse({'status': 'ok'}, '{"status":"ok"}', 'test', 10, 5, 1)
    provider = SimpleNamespace(generate=generate, capabilities=lambda model: ProviderCapabilities(1000, 200, False, True, False, True))
    assert health.probe(provider, 'chosen-free') == {'status': 'ok'}
    assert len(calls) == 1 and calls[0].model == 'chosen-free'
    assert calls[0].max_output_tokens <= 128
    assert calls[0].input_payload == {'test': 'synthetic'}


def test_probe_does_not_retry_quota():
    from ainovel.providers.llm_response import LLMQuotaError
    calls = []
    def generate(request):
        calls.append(request)
        raise LLMQuotaError()
    provider = SimpleNamespace(generate=generate, capabilities=lambda model: ProviderCapabilities(1000,200,False,True,False,True))
    with pytest.raises(LLMQuotaError):
        importlib.import_module('ainovel.llm_health').probe(provider, 'chosen-free')
    assert len(calls) == 1


def test_cli_requires_explicit_run(capsys):
    assert importlib.import_module('ainovel.llm_health').main([]) == 0
    assert '--run' in capsys.readouterr().out


def test_probe_accepts_small_explicit_output_cap():
    calls=[]
    def generate(request):
        calls.append(request)
        return ModelResponse({'status':'ok'}, None, None, 10, 5, 1)
    provider=SimpleNamespace(generate=generate, capabilities=lambda m: ProviderCapabilities(1000,200,False,True,False,True))
    assert importlib.import_module('ainovel.llm_health').probe(provider,'model',max_output_tokens=32)=={'status':'ok'}
    assert calls[0].max_output_tokens==32


def test_raw_debug_setting_is_restored_after_probe(monkeypatch):
    import logging, os
    health=importlib.import_module('ainovel.llm_health')
    monkeypatch.setattr(health,'HealthSettings',lambda: SimpleNamespace(model='',base_url='',api_key=None))
    monkeypatch.delenv('AINOVEL_LLM_DEBUG_CONTENT',raising=False)
    logger=logging.getLogger('ainovel.llm')
    before=(logger.level, list(logger.handlers))
    assert health.main(['--run','--show-content'])==2
    assert 'AINOVEL_LLM_DEBUG_CONTENT' not in os.environ
    assert (logger.level,logger.handlers)==before

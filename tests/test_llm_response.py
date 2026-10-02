import importlib
import json
import logging

import pytest


def api():
    return importlib.import_module('ainovel.providers.llm_response')


@pytest.mark.parametrize('content', ['{"status":"ok"}', '```json\n{"status":"ok"}\n```', '\n {"status":"ok"} \t'])
def test_plain_and_fenced_json(content):
    assert api().parse_llm_json_response(content) == {'status': 'ok'}


@pytest.mark.parametrize('content', ['', 'hello world', '[]', '{} {}', '```json\n{}', '{"x":NaN}', '{"x":1e999}'])
def test_format_errors_are_typed(content):
    with pytest.raises(api().LLMFormatError):
        api().parse_llm_json_response(content)


@pytest.mark.parametrize('content', ['Your trial quota has been exhausted', 'Insufficient balance', 'quota exceeded', 'billing required', '账户余额不足，请充值', '试用额度已用完'])
def test_explicit_quota_not_json_error(content):
    with pytest.raises(api().LLMQuotaError):
        api().parse_llm_json_response(content)


@pytest.mark.parametrize('content', ['Discuss balance and credit in the story', 'The hero said insufficient balance at the shop', 'quota', 'trial'])
def test_mentions_are_not_provider_errors(content):
    with pytest.raises(api().LLMFormatError):
        api().parse_llm_json_response(content)


def test_valid_story_json_not_classified_as_quota():
    assert api().parse_llm_json_response('{"body":"insufficient balance"}') == {'body': 'insufficient balance'}


def test_actual_unfunded_trial_notice_is_quota_without_network():
    content = ('Sorry, to prevent abuse of free resources, accounts that have not been recharged can only try 10 times. '
               'You can increase the free quota after recharging; https://console.aihubmix.com/topup')
    with pytest.raises(api().LLMQuotaError):
        api().parse_llm_json_response(content)
    assert api().parse_llm_json_response(json.dumps({'body':content})) == {'body':content}


@pytest.mark.parametrize('status,body,reason', [
    (429, {'error': {'type': 'insufficient_quota', 'message': 'secret'}}, 'provider_quota'),
    (429, {'error': {'code': 'rate_limit_exceeded'}}, 'provider_rate_limit'),
    (200, {'error': {'message': 'Insufficient balance'}}, 'provider_quota'),
    (200, {'error': {'message': 'arbitrary vendor failure'}}, 'provider_api'),
    (503, {'error': {'message': 'insufficient balance'}}, 'provider_temporary'),
])
def test_provider_error_precedence(status, body, reason):
    with pytest.raises(api().LLMProviderError) as caught:
        api().check_provider_error(body, status=status)
    assert caught.value.reason.value == reason
    assert 'secret' not in str(caught.value)


def test_content_debug_is_opt_in_redacted_and_bounded(monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger('ainovel.llm'), 'disabled', False)
    caplog.set_level(logging.DEBUG, logger='ainovel.llm')
    monkeypatch.delenv('AINOVEL_LLM_DEBUG_CONTENT', raising=False)
    api().log_diagnostic(model='model-free', base_url='https://example/v1', content='PRIVATE', api_key='KEYVALUE')
    assert 'PRIVATE' not in caplog.text
    monkeypatch.setenv('AINOVEL_LLM_DEBUG_CONTENT', '1')
    api().log_diagnostic(model='model-free', base_url='https://example/v1', content='KEYVALUE sk-abcdef123456 ' + 'x'*2000, api_key='KEYVALUE')
    assert 'KEYVALUE' not in caplog.text and 'sk-abcdef123456' not in caplog.text
    assert '[REDACTED]' in caplog.text and 'truncated' in caplog.text


def test_workflow_quota_is_not_retryable():
    from ainovel.services.workflows import WorkflowService
    error = api().LLMQuotaError()
    code, detail, retry = WorkflowService._provider_failure(error)
    assert code == 'provider_quota' and not retry
    assert '额度' in detail

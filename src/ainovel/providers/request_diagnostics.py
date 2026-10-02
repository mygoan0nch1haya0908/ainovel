"""Content-free, per-call diagnostics; ContextVar isolates concurrent requests."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import re
import os
import threading
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from ainovel.context import ConservativeEstimator
from ainovel.providers.diagnostics import safe_failure_code, failure_code_detail
from ainovel.providers.llm_response import safe_preview,safe_endpoint_url
from ainovel.providers.contracts import ProviderProtocolError
from ainovel.providers.llm_diagnostic import diagnose, safe_headers, SAFE_CODES
from ainovel.providers.llm_metrics import metrics
from ainovel.providers.send_metrics import SendMetrics

send_metrics = SendMetrics()


def note_actual_send():
    state = _current.get()
    if state is None:
        return
    before = state.get('_send_before', {})
    raw_budget = os.getenv('MAX_LLM_CALLS_PER_USER_REQUEST', '128')
    budget = int(raw_budget) if raw_budget.isascii() and raw_budget.isdigit() and 1 <= int(raw_budget) <= 10000 else 128
    sent = send_metrics.start((before.get('base_url'), before.get('model')),
        before.get('operation_id'), before.get('estimated_input_tokens', 0), budget=budget)
    state.update(sent)
    state.update(user_request_id=before.get('operation_id'), pid=os.getpid(),
                 thread_id=str(threading.get_ident()), send_timestamp=datetime.now(timezone.utc).isoformat())
    logger.info('[LLM ACTUAL SEND] %s', json.dumps({**before, **sent,
        'user_request_id':state['user_request_id'], 'pid':state['pid'],
        'thread_id':state['thread_id'], 'send_timestamp':state['send_timestamp']}, ensure_ascii=False))

logger = logging.getLogger('ainovel.llm.telemetry')
logger.setLevel(logging.INFO)
_current = ContextVar('llm_diagnostic', default=None)
operation_context = ContextVar('llm_user_operation', default=None)


def configure_telemetry():
    """Ensure the local application's stderr receives content-free events."""
    logger.disabled = False
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())


def messages_for(request):
    try:
        schema = json.dumps(request.output_schema, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return [{'role': 'system', 'content': f'{request.system_prompt}\n\nReturn only one JSON object matching this schema:\n{schema}'},
                {'role': 'user', 'content': json.dumps(request.input_payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}]
    except Exception:
        raise ProviderProtocolError('invalid model input') from None


def note_transport(phase=None, status=None, request_id=None, api_key=None, headers=None):
    state = _current.get()
    if state is None:
        return
    if headers is not None:
        state.update(safe_headers(headers, api_key=api_key))
    if phase in {'dns', 'connect', 'write', 'response_headers', 'response_body'}:
        state['phase'] = phase
    if type(status) is int and 100 <= status <= 599:
        state.update(http_status=status, received_response=True)
    # Only an opaque, bounded identifier; never arbitrary header/error text.
    if (isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', request_id)
            and not (api_key and api_key in request_id) and not request_id.lower().startswith('sk-')):
        state['request_id'] = request_id


def note_response(result):
    state = _current.get()
    if state is None or not isinstance(result, dict):
        return
    state['received_response'] = True
    error = result.get('error')
    if isinstance(error, dict):
        allowed = {'provider_timeout', 'upstream_timeout', 'gateway_timeout', 'request_timeout',
                   'insufficient_quota', 'insufficient_balance', 'billing_required', 'trial_quota_exhausted',
                   'credit_exhausted', 'rate_limit_exceeded', 'rate_limit_error', 'too_many_requests'}
        code = error.get('code') or error.get('type')
        state['provider_code'] = code if isinstance(code, str) and code in SAFE_CODES else 'unrecognized_redacted'
        state['provider_error_type'] = error.get('type') if isinstance(error.get('type'),str) and error['type'] in SAFE_CODES else None
        message = error.get('message')
        hint = re.fullmatch(r'Model [A-Za-z0-9_.:/-]{1,255} rate limited by provider [–-] contact support to request higher concurrency or try again later\. \(tid: ([0-9]{10,64})\)', message) if isinstance(message,str) else None
        if hint and str(code)=='429' and error.get('type')=='limitation':
            state['provider_reported_concurrency']=True
            state['trace_id']=hint.group(1)
    choices = result.get('choices')
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get('message')
    state['received_content'] = isinstance(message, dict) and isinstance(message.get('content'), str) and bool(message['content'])
    usage = result.get('usage')
    if isinstance(usage, dict):
        def count(k):
            v = usage.get(k)
            return v if type(v) is int and 0 <= v <= 1_000_000_000 else None
        prompt, completion, total = count('prompt_tokens'), count('completion_tokens'), count('total_tokens')
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        state['usage'] = {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': total}


@contextmanager
def trace_request(request, base_url, api_key, *, messages=None, response_format='json_object'):
    messages = messages_for(request) if messages is None else messages
    chars = {role: sum(len(m['content']) for m in messages if m['role'] == role) for role in ('system', 'user', 'assistant', 'tool')}
    def number(key, default):
        value = request.metadata.get(key, '') if isinstance(request.metadata, dict) else ''
        return int(value) if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 9 else default
    sequence = uuid4().hex
    before = {'sequence': sequence, 'round': number('round', None), 'attempt': number('attempt', 1),
              'model': safe_preview(request.model, api_key=api_key), 'base_url': safe_endpoint_url(base_url, api_key=api_key),
              'message_count': len(messages), **{k + '_chars': v for k, v in chars.items()},
              'total_chars': sum(chars.values()), 'estimated_input_tokens': ConservativeEstimator().estimate(''.join(m['content'] for m in messages)),
              'token_count_kind': 'conservative_estimate_not_tokenizer',
              'max_output_tokens': request.max_output_tokens, 'timeout_seconds': min(request.timeout_seconds, 180),
              'temperature': None, 'tools_enabled': False, 'tool_count': 0, 'response_format': response_format}
    operation = operation_context.get() or request.metadata.get('workflow_id') or request.metadata.get('roadmap_id')
    before['operation_id'] = operation if isinstance(operation,str) and re.fullmatch(r'[a-f0-9-]{36}',operation) else None
    before.update(metrics.start((before['base_url'],before['model']),before['operation_id'],before))
    state = {'sequence': sequence, 'round': before['round'], 'attempt': before['attempt'],
             'http_status': None, 'request_id': None, 'received_response': False,
             'received_content': False, 'usage': None, 'phase': 'unknown', 'exception_type': None,
             'provider_code': None, 'error_code': None, 'error_summary': None}
    token = _current.set(state)
    state['_send_before'] = before
    started = perf_counter()
    failure = None
    logger.info('[LLM REQUEST] %s', json.dumps(before, ensure_ascii=False))
    try:
        yield state
    except Exception as error:
        failure = error
        existing=getattr(error,'diagnostic',None)
        if isinstance(existing,dict):
            for key in ('http_status','request_id','trace_id','cf_ray','retry_after','rate_limits'):
                if existing.get(key) is not None: state[key]=existing[key]
            state['provider_code']=existing.get('provider_error_code') or state.get('provider_code')
        state['exception_type'] = type(error).__name__
        state['error_code'] = safe_failure_code(error)
        from ainovel.providers.send_metrics import CallBudgetExceeded
        if isinstance(error, CallBudgetExceeded):
            state['error_code'] = 'local_call_budget'
        state['error_summary'] = failure_code_detail(state['error_code'])
        state['http_status'] = state['http_status'] or getattr(error, 'http_status', None)
        if isinstance(existing,dict) and existing.get('error_code')=='sdk_client_timeout':
            state['error_code']='sdk_client_timeout'
        if isinstance(existing,dict) and existing.get('exception_type'):
            state['exception_type']=existing['exception_type']
        # Vendor messages may echo prompt/key: deliberately omitted, not merely key-redacted.
        raise
    finally:
        state.pop('_send_before', None)
        if state.get('llm_call_id'):
            state.update(send_metrics.finish(state['llm_call_id'], state.get('usage')))
        state['elapsed_seconds'] = round(perf_counter() - started, 3)
        state.update(metrics.finish(before['metric_id'],state.get('usage')))
        state['diagnostic'] = diagnose(before,state)
        if failure is not None:
            failure.diagnostic = state['diagnostic']
        logger.info('[LLM RESPONSE] %s', json.dumps(state, ensure_ascii=False))
        _current.reset(token)


def note_sdk_response(response, *, responses_api=False, api_key=None):
    usage=getattr(response,'usage',None)
    choices=getattr(response,'choices',None)
    content=getattr(response,'output_text',None) if responses_api else getattr(getattr(choices[0],'message',None),'content',None) if isinstance(choices,list) and choices else None
    error=getattr(response,'error',None)
    if error is not None and not isinstance(error,dict):
        error={k:getattr(error,k,None) for k in ('code','type')}
    note_response({'error':error,'choices':[{'message':{'content':content}}],
        'usage':{'prompt_tokens':getattr(usage,'input_tokens' if responses_api else 'prompt_tokens',None),
                 'completion_tokens':getattr(usage,'output_tokens' if responses_api else 'completion_tokens',None),
                 'total_tokens':getattr(usage,'total_tokens',None)}})
    note_transport(request_id=getattr(response,'_request_id',None),api_key=api_key)

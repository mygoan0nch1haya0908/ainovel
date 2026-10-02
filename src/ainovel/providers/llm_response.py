"""Shared parsing and conservative provider-error recognition; no network retries.

Raw diagnostics are opt-in and never become persisted/public exception messages.
"""
import json
import logging
import math
import os
import re
from urllib.parse import urlsplit, urlunsplit

from ainovel.providers.contracts import ProviderAuthenticationError, ProviderError, ProviderTimeout
from ainovel.providers.diagnostics import FailureReason, ResponseFailure

logger = logging.getLogger('ainovel.llm')


class LLMProviderError(ResponseFailure):
    def __init__(self, reason=FailureReason.PROVIDER_API, *, http_status=None):
        super().__init__(reason, http_status=http_status)
        self.args = ('LLM provider error: ' + self.reason.value,)


class LLMQuotaError(LLMProviderError):
    def __init__(self, *, http_status=None):
        super().__init__(FailureReason.QUOTA, http_status=http_status)
        self.args = ('LLM provider quota/balance or trial restriction',)


class LLMFormatError(ResponseFailure):
    def __init__(self, content='', *, api_key=None):
        super().__init__(FailureReason.EMPTY if not content.strip() else FailureReason.CONTENT_JSON)
        # Debug-only preview, not args: existing persistence uses fixed messages.
        self.diagnostic_preview = safe_preview(content, api_key=api_key) if debug_content_enabled() else None


def debug_content_enabled():
    return os.environ.get('AINOVEL_LLM_DEBUG_CONTENT') == '1'


def safe_preview(content, *, api_key=None, limit=500):
    text = content if isinstance(content, str) else ''
    # Normalize escaped key characters before redaction (never print decoded keys).
    text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
    if isinstance(api_key, str) and api_key:
        text = text.replace(api_key, '[REDACTED]')
        text = text.replace(json.dumps(api_key)[1:-1], '[REDACTED]')
    text = re.sub(r'(?i)\b(?:sk|sk-proj)-[a-z0-9_-]+', '[REDACTED]', text)
    text = re.sub(r'(?i)(bearer\s+)[^\s"\x27,}]+', r'\1[REDACTED]', text)
    return text[:limit] + ('...[truncated]' if len(text) > limit else '')


def log_diagnostic(*, model, base_url, content=None, api_key=None, exception_type=None):
    if not logger.isEnabledFor(logging.DEBUG):
        return
    url = urlsplit(str(base_url))
    safe_url = urlunsplit((url.scheme, url.hostname or '', url.path, '', ''))
    logger.debug('LLM model=%r base_url=%r exception_type=%r',
                 safe_preview(model, api_key=api_key), safe_preview(safe_url, api_key=api_key), exception_type)
    if debug_content_enabled() and content is not None:
        logger.debug('Raw LLM content (redacted/bounded)=%r', safe_preview(content, api_key=api_key))


def safe_endpoint_url(base_url, *, api_key=None):
    try:
        url=urlsplit(str(base_url))
        host=url.hostname or ''
        if ':' in host: host='['+host+']'
        if url.port: host+=':'+str(url.port)
        return safe_preview(urlunsplit((url.scheme,host,url.path,'','')),api_key=api_key)
    except ValueError:
        return '[invalid endpoint]'


# Match error phrases, not individual financial/story vocabulary.
_QUOTA = re.compile(
    r'(?i)(?:insufficient[ _-]+(?:balance|quota|credits?|funds)|'
    r'(?:trial[ _-]+)?quota(?:\s+has\s+been|\s+is)?[ _-]+(?:exceeded|exhausted|depleted)|'
    r'(?:balance|credits?)(?:\s+is|\s+has\s+been)?[ _-]+(?:insufficient|exhausted|depleted)|'
    r'billing[ _-]+(?:required|not[ _-]+enabled)|'
    r'(?:试用)?(?:额度|余额|配额)(?:已|已经)?(?:不足|耗尽|用完)|'
    r'试用(?:期)?(?:已)?(?:到期|结束)|请(?:先)?充值)')
_RATE = re.compile(r'(?i)(?:rate[ _-]+limit(?:[ _-]+exceeded)?|too many requests|请求(?:过于频繁|频率过高)|触发限流)')
_PREFIX = re.compile(r'(?i)^(?:(?:error|warning|sorry|your|account|api|provider|the|free|trial)\b[\s:：,.-]*|(?:错误|提示|账户|账号|您的|你的|当前)[：:\s]*)*')
_UNFUNDED_TRIAL = re.compile(
    r'(?i)^(?:sorry,?\s*)?to prevent abuse of free resources,\s*'
    r'accounts that have not been recharged can only try \d+ times[.!]\s*'
    r'you can increase the free quota after recharging[.;]?\s*(?:https?://\S+)?$')


def _message_reason(message, *, envelope=False):
    if not isinstance(message, str) or len(message) > 2000:
        return None
    text = message.strip()
    if _UNFUNDED_TRIAL.fullmatch(text):
        return FailureReason.QUOTA
    if not envelope:
        text = _PREFIX.sub('', text)
    for pattern, reason in ((_QUOTA, FailureReason.QUOTA), (_RATE, FailureReason.RATE_LIMIT)):
        match = pattern.search(text) if envelope else pattern.match(text)
        if match:
            return reason
    return None


def check_provider_error(payload, *, status=200):
    if status in (401, 403):
        raise ProviderAuthenticationError('model authentication or permission failed')
    if status == 504:
        raise ProviderTimeout('provider or gateway returned a timeout', source='upstream', http_status=status)
    if 500 <= status <= 599:
        raise LLMProviderError(FailureReason.TEMPORARY, http_status=status)
    error = payload.get('error') if isinstance(payload, dict) else None
    reason = None
    if error is not None:
        if isinstance(error, dict):
            codes = [error.get('type'), error.get('code')]
            if any(c in ('insufficient_quota', 'insufficient_balance', 'billing_required', 'trial_quota_exhausted', 'credit_exhausted') for c in codes if isinstance(c, str)):
                reason = FailureReason.QUOTA
            elif any(c in ('rate_limit_exceeded', 'rate_limit_error', 'too_many_requests') for c in codes if isinstance(c, str)):
                reason = FailureReason.RATE_LIMIT
            reason = reason or _message_reason(error.get('message'), envelope=True)
            if reason is None and any(c in ('provider_timeout', 'upstream_timeout', 'gateway_timeout', 'request_timeout') for c in codes if isinstance(c, str)):
                raise ProviderTimeout('provider returned a timeout', source='upstream', http_status=status)
        elif isinstance(error, str):
            reason = _message_reason(error, envelope=True)
        if reason == FailureReason.QUOTA:
            raise LLMQuotaError(http_status=status)
        raise LLMProviderError(reason or FailureReason.PROVIDER_API, http_status=status)
    if not 200 <= status < 300:
        raise LLMProviderError(FailureReason.HTTP, http_status=status)


def parse_json_object(data):
    def reject(value):
        raise ValueError('invalid JSON number')
    def finite(value):
        number = float(value)
        return number if math.isfinite(number) else reject(value)
    result = json.loads(data, parse_constant=reject, parse_float=finite)
    if not isinstance(result, dict):
        raise ValueError('JSON object required')
    return result


def raise_sdk_status_error(error, *, model, client):
    body = getattr(error, 'body', None)
    log_diagnostic(model=model, base_url=getattr(client, 'base_url', ''),
                   api_key=getattr(client, 'api_key', None), exception_type=type(error).__name__,
                   content=json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, str)) else None)
    payload = body if isinstance(body, dict) and 'error' in body else {'error': body} if body else {}
    try:
        check_provider_error(payload, status=error.status_code)
    except ProviderError as classified:
        attach_sdk_diagnostic(classified,error,model=model,client=client)
        raise classified from None


def attach_sdk_diagnostic(classified,error,*,model,client):
    from ainovel.providers.llm_diagnostic import diagnose,safe_headers,SAFE_CODES
    from ainovel.providers.diagnostics import safe_failure_code
    key=getattr(client,'api_key',None)
    body=getattr(error,'body',None)
    body=body.get('error',body) if isinstance(body,dict) else {}
    code=body.get('code') or body.get('type') if isinstance(body,dict) else None
    state=safe_headers(getattr(getattr(error,'response',None),'headers',{}),api_key=key)
    state.update(http_status=getattr(error,'status_code',None),error_code=safe_failure_code(classified),
                 provider_code=code if isinstance(code,str) and code in SAFE_CODES else None,exception_type=type(error).__name__)
    if type(error).__name__=='APITimeoutError':
        # SDK does not expose a reliable connect/read phase here; do not invent it.
        state['error_code']='sdk_client_timeout'
    classified.diagnostic=diagnose({'model':safe_preview(model,api_key=key),
        'base_url':safe_endpoint_url(getattr(client,'base_url',''),api_key=key)},state)
    return classified


def check_sdk_response_error(response):
    error = getattr(response, 'error', None)
    if error is not None:
        if not isinstance(error, (dict, str)):
            error = {field: getattr(error, field, None) for field in ('type', 'code', 'message')}
        check_provider_error({'error': error})


def parse_llm_json_response(content: str, *, api_key=None):
    if not isinstance(content, str) or not content.strip():
        raise LLMFormatError('', api_key=api_key)
    normalized = content.strip()
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', normalized, re.DOTALL | re.IGNORECASE)
    if fence:
        normalized = fence.group(1)
    # A JSON story mentioning quota is not an API error; only bare error prose
    # is checked here. Explicit top-level error objects are checked below.
    if not normalized.startswith(('{', '[')):
        reason = _message_reason(normalized)
        if reason == FailureReason.QUOTA:
            raise LLMQuotaError()
        if reason == FailureReason.RATE_LIMIT:
            raise LLMProviderError(reason)
    try:
        result = parse_json_object(normalized)
    except (ValueError, RecursionError):
        raise LLMFormatError(content, api_key=api_key) from None
    check_provider_error(result)
    return result

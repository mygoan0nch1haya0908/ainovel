"""Structured facts and conservative inferences. No arbitrary vendor prose retained."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite, ceil
import random
import re

RATE_CODES={'rpm_limit','tpm_limit','concurrency_limit','account_rate_limit','model_rate_limit','provider_capacity'}
QUOTA_CODES={'insufficient_quota','insufficient_balance','billing_required','trial_quota_exhausted','credit_exhausted'}
TIMEOUT_CODES={'provider_timeout','upstream_timeout','gateway_timeout','request_timeout'}
SAFE_CODES=RATE_CODES|QUOTA_CODES|TIMEOUT_CODES|{'rate_limit_exceeded','rate_limit_error','too_many_requests',
    'context_length_exceeded','model_not_found','invalid_request_error','provider_overloaded','overloaded','internal_server_error'}
SAFE_CODES |= {'429', 'limitation'}
RATE_HEADERS={'x-ratelimit-limit-requests','x-ratelimit-remaining-requests','x-ratelimit-limit-tokens','x-ratelimit-remaining-tokens'}
RATE_HEADERS |= {'x-ratelimit-limit','x-ratelimit-remaining','x-ratelimit-reset',
                 'ratelimit-limit','ratelimit-remaining','ratelimit-reset'}


def number(value):
    return value if type(value) in (int,float) and isfinite(value) and 0<=value<=1_000_000_000 else None


def safe_headers(headers, *, now=None, api_key=None):
    values={str(k).lower():str(v) for k,v in headers.items()}
    result={'retry_after':None,'request_id':None,'trace_id':None,'cf_ray':None,'rate_limits':{}}
    for target, names in [('request_id',('x-request-id','request-id')),('trace_id',('x-trace-id','trace-id')),('cf_ray',('cf-ray',))]:
        v=next((values[k] for k in names if k in values),None)
        if v and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',v) and not v.lower().startswith('sk-') and not (api_key and api_key in v):
            result[target]=v
    value=values.get('retry-after','').strip()
    if value.isascii() and value.isdigit() and len(value)<=10:
        result['retry_after']=number(int(value))
    elif value:
        try:
            dt=parsedate_to_datetime(value)
            if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
            result['retry_after']=number(max(0,ceil(dt.timestamp()-(datetime.now(timezone.utc).timestamp() if now is None else now))))
        except (ValueError,TypeError,OverflowError):
            pass
    for key in RATE_HEADERS:
        value=values.get(key,'')
        if value.isascii() and value.isdigit() and len(value)<=10 and number(int(value)) is not None:
            result['rate_limits'][key]=int(value)
    return result


def retry_delay(diagnostic, attempt, *, jitter=None):
    retry=number(diagnostic.get('retry_after')) if isinstance(diagnostic,dict) else None
    delay=retry if retry is not None else 2**min(max(attempt-1,0),4)+(random.random() if jitter is None else jitter)
    return delay if delay<=15 else None  # Pause, never shorten a server wait or occupy a lease for minutes.


def diagnose(before, state):
    status=state.get('http_status')
    code=state.get('provider_code')
    code=code if isinstance(code,str) and code in SAFE_CODES else None
    error=state.get('error_code','') or ''
    if not isinstance(error,str): error=''
    category,subtype,layer,confidence='unknown','unknown','unknown','low'
    evidence=[]
    if type(status) is int and 100<=status<=599:
        evidence.append(f'收到 HTTP {status}。')
        mapping={401:('authentication','authentication_failed'),403:('permission','permission_denied'),
            404:('model_not_found','model_or_endpoint_not_found'),408:('timeout','gateway_timeout'),
            413:('invalid_request','payload_too_large'),422:('invalid_request','request_validation'),
            429:('rate_limit','unknown_rate_limit'),500:('provider','provider_internal_error'),
            502:('provider','provider_5xx'),503:('provider','provider_unavailable'),504:('timeout','gateway_timeout')}
        if status in mapping:
            category,subtype=mapping[status];layer='remote_endpoint';confidence='high'
    if error=='local_call_budget':
        category,subtype,layer,confidence='rate_limit','local_call_budget','agent','high'
        evidence.append('本地调用预算已用尽，本次没有发送 API 请求。')
    elif error=='sdk_client_timeout':
        category,subtype,layer,confidence='timeout','client_timeout','local_sdk','high'
    elif error.startswith('client_timeout_'):
        category,layer,confidence='timeout','local_http_client','high'
        phase=error[len('client_timeout_'):]
        subtype='connect_timeout' if phase in {'connect','dns'} else 'read_timeout' if phase in {'response_headers','response_body'} else 'client_timeout'
        evidence.append('本地传输计时器在已记录阶段触发，并非仅按错误文本猜测。')
    elif status not in (401,403):
        if code in QUOTA_CODES or error=='provider_quota':
            category,subtype,layer,confidence='quota','quota_exhausted','remote_endpoint','high'
            if code not in QUOTA_CODES:
                layer,confidence='unverified_response','medium'
                evidence.append('仅依据错误文本或适配器粗分类推断额度问题，没有结构化额度错误码佐证。')
        elif code in TIMEOUT_CODES or error=='upstream_timeout':
            category,subtype,layer,confidence='timeout','gateway_timeout' if status in (408,504) else 'provider_timeout','remote_endpoint','high'
        elif code=='context_length_exceeded':
            category,subtype,layer,confidence='context_length',code,'remote_endpoint','high'
        elif code=='model_not_found':
            category,subtype,layer,confidence='model_not_found',code,'remote_endpoint','high'
        elif code=='invalid_request_error':
            category,subtype,layer,confidence='invalid_request','request_validation','remote_endpoint','high'
        elif code in RATE_CODES or code in {'rate_limit_exceeded','rate_limit_error','too_many_requests'} or error=='provider_rate_limit':
            category,subtype,layer='rate_limit',code if code in RATE_CODES else 'unknown_rate_limit','remote_endpoint'
            confidence='high' if code in RATE_CODES else 'low'
        elif code in {'provider_overloaded','overloaded'}:
            category,subtype,layer,confidence='provider','provider_overloaded','remote_endpoint','high'
        elif error in {'model_content_json','response_empty','transport_json'}:
            category,subtype,layer,confidence='format','invalid_json','local_parser','high'
        elif error=='schema_mismatch':
            category,subtype,layer,confidence='json_schema',error,'local_validator','high'
        elif error=='provider_unavailable':
            category,subtype,layer,confidence='connection','connection_failed','client_or_network','low'
        elif error=='provider_timeout':
            category,subtype='timeout','unknown_timeout'
        elif error=='provider_auth':
            category,subtype,layer='authentication','authentication_failed','client_or_remote'
    limits=state.get('rate_limits',{})
    reported_hint = state.get('provider_reported_concurrency') is True
    if category=='rate_limit' and reported_hint:
        subtype,layer,confidence='concurrency_limit','upstream_provider','medium'
        evidence.append('供应商结构化错误报告模型被上游限流，并建议提高并发额度；疑似上游并发或共享容量限制，不证明本地并发超标。')
    if not isinstance(limits,dict): limits={}
    if category=='rate_limit' and subtype=='unknown_rate_limit':
        confidence='low'
        zero=[key for key in ('requests','tokens') if limits.get('x-ratelimit-remaining-'+key)==0]
        if len(zero)==1:
            subtype='rpm_limit' if zero[0]=='requests' else 'tpm_limit';confidence='medium'
            evidence.append('供应商对应 remaining 响应头为 0；疑似该维度限制，窗口是否为每分钟仍需核对。')
        else:
            evidence.append('无法确定 RPM、TPM、并发或共享容量；没有足够的区分证据。')
    if code: evidence.append('供应商报告错误码：'+code+'；其内部根因仍需服务商核对。')
    if before.get('context_growth'):
        evidence.append('同一操作近 60 秒内输入超过早期值两倍且增加超过 1024 字符：存在上下文膨胀迹象，来源尚无法确定。')
    actions=['结合请求编号核对当前接口服务商日志；不要据此自动更换模型。']
    if category=='rate_limit': actions=['遵守 Retry-After；没有明确维度证据时，检查供应商限额说明，不直接认定余额不足。']+actions
    if category=='quota': actions=['核对当前供应商余额、试用或账户额度；不自动充值或改配额。']+actions
    if category=='timeout': actions=['核对超时阶段与耗时；用量未知不代表未计费，重试可能重复计费。']+actions
    if category in {'format','json_schema'}: actions=['检查返回格式、字段类型及输出是否截断；不要将其当成账户错误。']+actions
    usage=state.get('usage') if isinstance(state.get('usage'),dict) else {}
    d={'error_category':category,'error_subtype':subtype,'error_layer':layer,'confidence':confidence,
       'provider_error_code':code,'provider_message':None,'raw_error_excerpt':None,
       'evidence':evidence,'recommended_actions':actions,'http_status':status if type(status) is int else None,
       'usage_known':all(number(usage.get(k)) is not None for k in ('prompt_tokens','completion_tokens')),
       'actual_prompt_tokens':number(usage.get('prompt_tokens')),'completion_tokens':number(usage.get('completion_tokens')),
       'total_tokens':number(usage.get('total_tokens')),'metrics_scope':'本进程、同一接口与模型；发送时的60秒窗口（含当前请求），完成时回填已知用量；非账户全局指标，重启清空。',
       'rate_limits':{k:number(v) for k,v in limits.items() if k in RATE_HEADERS and number(v) is not None},
       'context_growth':bool(before.get('context_growth'))}
    d['metrics_truncated']=bool(state.get('metrics_truncated',before.get('metrics_truncated')))
    if d['metrics_truncated']: evidence.append('本地统计达到保留上限，窗口数据可能不完整。')
    from ainovel.providers.diagnostics import FailureReason
    known={r.value for r in FailureReason}|{'provider_timeout','provider_unavailable','provider_auth','provider_protocol','upstream_timeout','sdk_client_timeout','local_call_budget'}
    known|={'client_timeout_'+p for p in ('dns','connect','write','response_headers','response_body')}
    d['error_code']=error if error in known else None
    for key in ('attempt','round','elapsed_seconds','message_count','total_chars','estimated_input_tokens','concurrency',
                'requests_last_60s','tokens_last_60s','known_tokens_last_60s','estimated_tokens_last_60s','unknown_usage_requests',
                'operation_calls_last_60s','last_request_interval','retry_after','timeout_seconds','system_chars','user_chars','assistant_chars','tool_chars'):
        d[key]=number(state.get(key,before.get(key)))
    for key in ('model','base_url','sequence','operation_id','request_id','trace_id','cf_ray','exception_type'):
        value=state.get(key,before.get(key))
        d[key]=value if isinstance(value,str) and len(value)<=500 and not any(ord(c)<32 for c in value) else None
    d['provider_reported_concurrency']=reported_hint
    if reported_hint:
        d['provider_message']='Model rate limited by provider; contact support to request higher concurrency or try again later.'
    d['provider_error_type']=state.get('provider_error_type') if state.get('provider_error_type') in SAFE_CODES else None
    for key, value in {**before, **state}.items():
        if key.startswith('actual_') and key not in d or key in {'user_request_actual_calls','pid','max_output_tokens'}:
            d[key] = number(value)
        elif key in {'llm_call_id','user_request_id','thread_id','send_timestamp'}:
            d[key] = value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:+-]{1,128}',value) else None
    if (d.get('user_request_actual_calls') or 0)>5:
        evidence.append(f"本次用户操作已触发 {d['user_request_actual_calls']} 次实际发送，可能存在请求放大；请按调用编号核对。")
    return d


def diagnostic_for_error(error):
    from ainovel.providers.diagnostics import safe_failure_code
    existing=getattr(error,'diagnostic',None)
    if not isinstance(existing,dict):
        existing=getattr(getattr(error,'response',None),'diagnostic',None)
    existing=existing if isinstance(existing,dict) else {}
    state={**existing,'provider_code':existing.get('provider_error_code'),
           'error_code':existing.get('error_code') if existing.get('error_code') in {'sdk_client_timeout','local_call_budget'} else safe_failure_code(error),'usage':{
               'prompt_tokens':existing.get('actual_prompt_tokens'),
               'completion_tokens':existing.get('completion_tokens'),'total_tokens':existing.get('total_tokens')}}
    state['http_status']=existing.get('http_status') or getattr(error,'http_status',None)
    return diagnose(existing,state)


def public_diagnostic(value):
    if not isinstance(value,dict): return None
    return diagnose(value,{**value,'provider_code':value.get('provider_error_code'),
        'usage':{'prompt_tokens':value.get('actual_prompt_tokens'),'completion_tokens':value.get('completion_tokens'),'total_tokens':value.get('total_tokens')}})

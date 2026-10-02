"""Allowlisted response diagnostics; never serialize SDK or validation messages."""
from enum import Enum

from ainovel.providers.contracts import ModelResponse, ProviderProtocolError, ProviderTimeout, ProviderAuthenticationError, ProviderUnavailable


class FailureReason(str, Enum):
    UNKNOWN = 'unknown'
    HTTP = 'response_http'
    PROVIDER_API = 'provider_api'
    QUOTA = 'provider_quota'
    RATE_LIMIT = 'provider_rate_limit'
    TEMPORARY = 'provider_temporary'
    TRUNCATED = 'response_truncated'
    REFUSED = 'response_refused'
    FINISH = 'response_finish'
    EMPTY = 'response_empty'
    JSON = 'response_json'
    TRANSPORT_JSON = 'transport_json'
    CONTENT_JSON = 'model_content_json'
    ENVELOPE = 'response_envelope'
    METADATA = 'response_metadata'
    SCHEMA = 'schema_mismatch'
    CHAPTER_EMPTY = 'chapter_empty'
    TOO_SHORT = 'chapter_too_short'
    TOO_LONG = 'chapter_too_long'
    PLAN = 'plan_mismatch'
    GOAL = 'goal_missing'
    HOOK = 'hook_missing'
    REPEATED = 'repeated_blocks'
    ORDINAL = 'ordinal_gap'


_DETAILS = {
    FailureReason.UNKNOWN: '未记录可识别的细分原因，不能据此判断具体故障。',
    FailureReason.HTTP: '模型接口返回非成功 HTTP 状态；请核对接口配置、账户状态和服务可用性。',
    FailureReason.PROVIDER_API: '模型供应商返回 API 错误；请检查服务商配置和控制台，不是小说 JSON 格式错误。',
    FailureReason.QUOTA: '模型供应商额度／余额不足或免费试用受限（quota/balance/billing）；已停止自动重试，请检查当前服务商账户、试用次数及充值条件。',
    FailureReason.RATE_LIMIT: '模型供应商请求限流（rate limit）；请稍后再试，不等同于余额不足。',
    FailureReason.TEMPORARY: '模型供应商暂时不可用（HTTP 5xx）；请稍后再试。',
    FailureReason.TRUNCATED: '输出因长度限制被截断，未得到完整响应；请检查输出预算与模型限制。',
    FailureReason.REFUSED: '模型拒绝输出或触发内容过滤；请由作者检查输入内容。',
    FailureReason.FINISH: '模型未正常结束响应；没有保存不完整结果。',
    FailureReason.EMPTY: '模型没有返回可用的文本内容。',
    FailureReason.JSON: '返回内容不是合法的 JSON 对象；请检查结构化输出要求。',
    FailureReason.TRANSPORT_JSON: '接口响应本身不是合法的 JSON 对象；请检查接口地址及网关兼容性，不必修改小说大纲。',
    FailureReason.CONTENT_JSON: 'AI 消息内容不是完整的 JSON 对象；完整代码块已兼容，请检查模型的结构化输出能力。未自动补造内容或重试。',
    FailureReason.ENVELOPE: '响应缺少有效的候选消息结构。',
    FailureReason.METADATA: '响应编号或 Token 用量元数据缺失或无效；需检查接口兼容性。',
    FailureReason.SCHEMA: '返回字段或类型不符合约定的输出结构；未保存不合格结果。',
    FailureReason.CHAPTER_EMPTY: '章节标题或正文为空。',
    FailureReason.TOO_SHORT: '正文不足 4500 个可见字符；不会当成完整章节保存。',
    FailureReason.TOO_LONG: '正文超过 6000 个可见字符；不会自动截断或批准。',
    FailureReason.PLAN: '返回的计划章节数或章节序号与已确认任务不匹配。',
    FailureReason.GOAL: '正文未通过获批目标的逐字覆盖校验（忽略空白）；这不是语义质量判断。',
    FailureReason.HOOK: '正文未通过获批章末钩子的逐字覆盖校验（忽略空白）；这不是语义质量判断。',
    FailureReason.REPEATED: '正文存在完全相同的长段落，触发重复内容校验。',
    FailureReason.ORDINAL: '前序章节尚未完成，章节连续性校验失败。',
}


class ResponseFailure(ProviderProtocolError):
    def __init__(
        self,
        reason: FailureReason,
        *,
        visible_count: int | None = None,
        response: ModelResponse | None = None,
        http_status: int | None = None,
    ):
        self.reason = reason if type(reason) is FailureReason else FailureReason.UNKNOWN
        self.visible_count = visible_count
        self.response = response
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        # Preserve the public exception family and legacy safe SDK messages.
        if self.reason == FailureReason.HTTP:
            message = 'Qwen returned an unsuccessful response'
        elif self.reason == FailureReason.METADATA:
            message = 'Qwen returned malformed response metadata'
        elif self.reason in {FailureReason.TRUNCATED, FailureReason.REFUSED, FailureReason.FINISH,
                             FailureReason.EMPTY, FailureReason.JSON, FailureReason.ENVELOPE}:
            message = 'Qwen returned malformed structured output'
        else:
            message = 'provider returned an invalid response'
        super().__init__(message)


def safe_failure_detail(error: ResponseFailure) -> str:
    # Revalidate at the persistence boundary, even if exception attributes were changed.
    reason = error.reason if type(error.reason) is FailureReason else FailureReason.UNKNOWN
    detail = f'[{reason.value}] {_DETAILS[reason]}'
    status = getattr(error, 'http_status', None)
    if reason == FailureReason.HTTP and type(status) is int and 100 <= status <= 599:
        detail += f' HTTP {status}。'
    count = error.visible_count
    if reason in {FailureReason.TOO_SHORT, FailureReason.TOO_LONG} and type(count) is int and 0 <= count <= 1_000_000_000:
        detail += f' 实际可见字数：{count}；要求：4500–6000。'
    return detail


def safe_failure_code(error: Exception) -> str:
    if isinstance(error, ProviderTimeout):
        if error.source == 'upstream':
            return 'upstream_timeout'
        if error.source == 'client' and error.phase in {'dns', 'connect', 'write', 'response_headers', 'response_body'}:
            return 'client_timeout_' + error.phase
    if isinstance(error, ResponseFailure):
        reason = error.reason if type(error.reason) is FailureReason else FailureReason.UNKNOWN
        status = getattr(error, 'http_status', None)
        if reason == FailureReason.HTTP and type(status) is int and 100 <= status <= 599:
            return f'{reason.value}_{status}'
        return reason.value
    for kind, code in ((ProviderTimeout, 'provider_timeout'), (ProviderAuthenticationError, 'provider_auth'),
                       (ProviderProtocolError, 'provider_protocol'), (ProviderUnavailable, 'provider_unavailable')):
        if isinstance(error, kind):
            return code
    return FailureReason.UNKNOWN.value


def failure_code_detail(code: str | None) -> str:
    phases = {'dns': '解析域名', 'connect': '建立连接或 TLS 握手', 'write': '发送请求',
              'response_headers': '等待响应头', 'response_body': '读取响应正文'}
    if isinstance(code, str) and code.startswith('client_timeout_') and code[15:] in phases:
        return '本地 HTTP 客户端在' + phases[code[15:]] + '时达到超时限制；不代表供应商返回了超时错误。用量未知时不会记为 0，请核对用量后再重试。'
    fixed = {
        'upstream_timeout': '服务端返回了超时错误（或 HTTP 504）；不能仅凭此区分网关与上游模型。请核对用量和服务商请求记录后重试。',
        'provider_timeout': '等待模型响应超时；没有自动重试，服务商可能已处理请求，请先核对用量。',
        'provider_auth': '接口认证失败或调用未启用，请检查密钥、权限及模型配置。',
        'provider_protocol': '接口响应不符合协议，可能涉及响应格式、大小或安全检查；未保存响应正文。',
        'provider_unavailable': '模型服务不可用，请检查地址、网络及服务状态。',
        'paused_budget': '本次任务的用量预算或尝试次数已用完。',
        'paused_stale_version': '输入版本已变化，请核对当前设定后重新规划。',
    }
    if code in fixed:
        return fixed[code]
    if isinstance(code, str) and code.startswith('response_http_'):
        number = code[len('response_http_'):]
        if len(number) == 3 and number.isascii() and number.isdigit() and 100 <= int(number) <= 599:
            return safe_failure_detail(ResponseFailure(FailureReason.HTTP, http_status=int(number)))
    try:
        return _DETAILS[FailureReason(code)]
    except (ValueError, TypeError):
        return '旧记录未保存具体原因，无法还原当时的错误；后续尝试会记录脱敏分类。'

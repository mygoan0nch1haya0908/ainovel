"""Allowlisted response diagnostics; never serialize SDK or validation messages."""
from enum import Enum

from ainovel.providers.contracts import ProviderProtocolError


class FailureReason(str, Enum):
    UNKNOWN = 'unknown'
    HTTP = 'response_http'
    TRUNCATED = 'response_truncated'
    REFUSED = 'response_refused'
    FINISH = 'response_finish'
    EMPTY = 'response_empty'
    JSON = 'response_json'
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
    FailureReason.TRUNCATED: '输出因长度限制被截断，未得到完整响应；请检查输出预算与模型限制。',
    FailureReason.REFUSED: '模型拒绝输出或触发内容过滤；请由作者检查输入内容。',
    FailureReason.FINISH: '模型未正常结束响应；没有保存不完整结果。',
    FailureReason.EMPTY: '模型没有返回可用的文本内容。',
    FailureReason.JSON: '返回内容不是合法的 JSON 对象；请检查结构化输出要求。',
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
    def __init__(self, reason: FailureReason, *, visible_count: int | None = None):
        self.reason = reason if type(reason) is FailureReason else FailureReason.UNKNOWN
        self.visible_count = visible_count
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
    count = error.visible_count
    if reason in {FailureReason.TOO_SHORT, FailureReason.TOO_LONG} and type(count) is int and 0 <= count <= 1_000_000_000:
        detail += f' 实际可见字数：{count}；要求：4500–6000。'
    return detail

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from ainovel.agents.contracts import (
    BatchPlanDraft,
    BatchReview,
    ChapterDraft,
    ChapterSummaryDelta,
)


BUILTIN_PROMPTS: Mapping[str, str] = {
    "batch_planner": """你是批次规划代理。接受的任务是：依据已确认的官方大纲、作者约束和已发布章节上下文，为本批次列出连续章节的候选写作计划。

只把官方大纲、已批准的人物设定和已发布章节当作权威事实；候选章节、候选计划和未经批准的建议都不是官方事实。不得改写官方资料、不得宣称候选内容已经批准、不得编造任务未提供的事实，也不得输出小说正文、评论、解释、Markdown 或任何未请求的散文。

输出必须且只能是提供的 BatchPlanDraft JSON Schema 所要求的 JSON 对象。每个章节计划必须服务于官方大纲，序号从 1 起连续，并清楚给出标题、目标和章末钩子。不要提供隐藏推理、思维链或推理过程；只提供符合 Schema 的最终结构化结果。""",
    "chapter_writer": """你是章节主笔代理。接受的任务是：在已批准的本章计划、官方大纲、已发布章节和按顺序提供的前文上下文之内，写出一章候选小说正文。

官方大纲、已批准计划、已发布章节和作者锁定约束才是权威；候选内容不是官方事实。必须忠实执行已批准的本章计划，只使用任务提供的顺序上下文，不得假设、总结或泄露未来章节内容，不得篡改官方资料，也不得把候选章节说成已经批准。

正文必须包含 4,500–6,000 个可见中文字符，形成完整的当前章节而非提纲或片段。正文必须逐字包含已批准计划中的 goal 和 ending_hook（空白差异不计），以便进行确定性覆盖验证。输出必须且只能是提供的 ChapterDraft JSON Schema 所要求的 JSON 对象；除 title 和 body 外不得输出解释、评论、Markdown、前后缀或未请求的散文。不要提供隐藏推理、思维链或推理过程；只提供符合 Schema 的最终结构化结果。""",
    "chapter_summarizer": """你是章节状态摘要代理。接受的任务是：根据指定候选章节及其给定上下文，提炼可供后续顺序写作使用的章节摘要与状态变化。

只有官方大纲、已批准设定和已发布章节是权威；候选章节与候选状态变化仍须等待作者批准。不得把候选内容升格为官方事实，不得修改正文、不得写新小说内容、不得臆测未来章节，也不得输出解释、评论、Markdown 或其他未请求的散文。

输出必须且只能是提供的 ChapterSummaryDelta JSON Schema 所要求的 JSON 对象。summary 只记录任务证据支持的本章结果，state_delta 只记录明确发生的状态变化。不要提供隐藏推理、思维链或推理过程；只提供符合 Schema 的最终结构化结果。""",
    "batch_reviewer": """你是批次审阅代理。接受的任务是：审阅候选批次是否符合官方大纲、已批准计划、章节报告与状态摘要，并决定是否通过以及列出问题。

官方大纲、已批准计划和已发布章节是权威；候选章节、报告和摘要不能自行成为官方内容。默认依据章节报告和摘要审阅，不得改写小说、不得批准或发布内容、不得虚构证据、不得输出解释、评论、Markdown 或其他未请求的散文。只有在无法判断的具体争议上，才可通过 evidence_queries 请求精确摘录；不得以其他方式索取全文或泛泛证据。

输出必须且只能是提供的 BatchReview JSON Schema 所要求的 JSON 对象。issues 应可执行且以给定证据为依据；evidence_queries 仅包含所需的精确摘录请求。不要提供隐藏推理、思维链或推理过程；只提供符合 Schema 的最终结构化结果。""",
}


AGENT_SCHEMAS: Mapping[str, type[BaseModel]] = {
    "batch_planner": BatchPlanDraft,
    "chapter_writer": ChapterDraft,
    "chapter_summarizer": ChapterSummaryDelta,
    "batch_reviewer": BatchReview,
}


AGENT_PARAMETERS: Mapping[str, dict[str, object]] = {
    "batch_planner": {"max_input_tokens": 16_000, "max_output_tokens": 4_000},
    "chapter_writer": {"max_input_tokens": 32_000, "max_output_tokens": 12_000},
    "chapter_summarizer": {"max_input_tokens": 16_000, "max_output_tokens": 4_000},
    "batch_reviewer": {"max_input_tokens": 32_000, "max_output_tokens": 6_000},
}

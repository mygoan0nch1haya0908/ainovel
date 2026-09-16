"""Read-only author-facing labels; no workflow transitions live here."""

WORKFLOW_LABELS = {
    "PREPARING": "准备中", "PLANNING": "等待生成计划",
    "AWAITING_PLAN_APPROVAL": "等待确认计划", "GENERATING_CHAPTERS": "等待生成正文",
    "REVIEWING_BATCH": "等待内容检查", "CREATING_CANDIDATE_BATCH": "等待整理候选正文",
    "AWAITING_CONTENT_APPROVAL": "等待审阅正文", "COMPLETED": "已完成",
    "REJECTED": "已驳回", "CANCELLED": "已取消", "FAILED": "生成失败",
    "PAUSED_PROVIDER": "模型调用暂停", "PAUSED_CONTEXT_OVERFLOW": "上下文超限暂停",
    "PAUSED_ATTEMPTS": "重试次数已用完", "PAUSED_REVIEW": "内容检查未通过",
    "PAUSED_STALE_VERSION": "设定版本变化暂停",
}

STAGE_ROADMAP_LABELS = {
    "PENDING": "等待生成",
    "RUNNING": "生成中（若进程中断，请勿重新提交）",
    "PROPOSED": "等待作者批准",
    "APPROVED": "已批准",
    "PAUSED_PROVIDER": "Provider 暂停，可显式重试",
    "PAUSED_INVALID": "返回格式无效，可显式重试",
    "PAUSED_BUDGET": "预算或次数已用完",
    "PAUSED_CONTEXT_OVERFLOW": "上下文超限",
    "PAUSED_STALE_VERSION": "输入版本已变化",
}

RUN_LABELS = {
    "PLANNING": "生成章节计划", "GENERATING_CHAPTERS": "生成正文",
    "REVIEWING_BATCH": "继续检查正文", "CREATING_CANDIDATE_BATCH": "整理候选正文",
}

STEP_LABELS = {
    "PLANNING": "生成计划",
    "WRITING": "生成正文",
    "VALIDATING_CHAPTER": "章节覆盖检查",
    "SUMMARIZING": "生成摘要",
    "REVIEWING": "检查批次正文",
}

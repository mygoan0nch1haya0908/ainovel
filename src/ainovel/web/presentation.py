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

RUN_LABELS = {
    "PLANNING": "生成章节计划", "GENERATING_CHAPTERS": "生成正文",
    "REVIEWING_BATCH": "继续检查正文", "CREATING_CANDIDATE_BATCH": "整理候选正文",
}

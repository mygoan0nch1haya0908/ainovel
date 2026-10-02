# Scoped Story Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留完整资料，通过本地可追溯记忆检索构建当前剧情点任务包，替代工作流重复发送全文。

**Architecture:** 复用 ContextSource/ContextPacket 和官方/候选隔离，引入版本化约束卡、记忆关联和工作流上下文策略。确定性关联优先、关键词补充；模型调用入口保持单一，预览和策略转换不调用模型。

**Tech Stack:** Python、FastAPI、SQLAlchemy、SQLite/FTS5、Alembic、Pydantic、pytest；不增加外部服务。

**Spec:** `docs/superpowers/specs/2026-10-01-scoped-story-memory-design.md`

## Global Constraints

- 每批最多五章，作者批准后才推进正式进度。
- 此次不引入向量数据库、不新增每步模型检索调用、不提高默认输入预算掩盖重复输入。
- 必需项溢出不得截断或降级为可选；不悄悄回退到发送全书全文。
- 完整资料、旧提案、已发布正文不改写；索引是可重建派生数据。
- 检索限定项目与允许的工作流；模型推测和未确认候选不进入正式事实。
- 不进行真实API调用；可选AI约束卡提取单独授权，首期默认由作者编辑确认。
- 已有32K阶段输出预算和600秒阶段等待不是本计划的扩大范围；工作流步骤预算仍受现有策略约束。
- 当前工作树有大量此前未提交修改；执行前列清本任务基线，按文件/补丁暂存，不执行 git add .、reset 或 checkout 覆盖用户改动。只有能隔离本任务改动时才提交，否则记录待提交范围，不为提交回滚旧工作。

## Review Focus

1. 预览后设定/规划变动：确认操作必须拒绝过期预览，不能静默使用新来源（任务5、6）。
2. 跨剧情点批次及依赖成环：带齐本批点，非法依赖报错而非递归全文（任务3）。
3. 中文人物别名/关键词未命中：显式关联与全局约束仍保留，不把FTS未命中当成无约束（任务2、3）。
4. 同批候选与其他项目同名人物：仅本工作流前序候选可用，正式状态不被覆盖（任务3、5）。
5. 所有必需条目本身超预算：无模型调用、无截断，报告确定的缺口（任务4、7）。

## 文件与公共接口约定

新增 `models/story_memory.py` 存版本化数据，`services/story_memory.py` 管理来源及约束卡，`services/scoped_context.py` 负责选择和预算报告，`services/context_policy.py` 管理策略绑定/转换；不拆改整个现有 context.py。

新增 `agents/memory_contracts.py` 定义以下Pydantic结构，其他任务直接复用：

- `SourceRef(project_id, source_type, source_id, source_version, content_hash, field_path, excerpt_start?, excerpt_end?)`：版本与原文引用。
- `MemoryEntryInput(kind, text, source_refs, entity_ids, point_ids, effective_from, effective_until?, reveal_from?, audience)`：kind 为 rule/character/fact/foreshadowing/summary；audience 为 author_only/narratable。未指定关联不等于全局，需作者确认分类。
- `MemorySelection(required, optional, missing, source_fingerprint)`：条目均带真实文本、范围及引用。
- `ContextPreview(workflow_id, step_id, policy_version, source_fingerprint, legacy_estimated_tokens, estimated_tokens, capacity, deficit, items, blockers)`：items 含来源、类别、必需性、选取/排除原因和估算；deficit=max(0,必需量-容量)。明确标注本地估算。

所有服务构造器为 `(session: Session)`；预览不提交事务，不生成模型产物。业务批准服务事务内调用的函数只能flush，不能独立commit。新增策略名为 `scoped_story_v1`，旧策略名为 `legacy`。

## Task 1：版本化存储与迁移

**Files:** Create `src/ainovel/models/story_memory.py`, `src/ainovel/agents/memory_contracts.py`, `alembic/versions/0009_scoped_story_memory.py`, `tests/test_story_memory_migration.py`; Modify `src/ainovel/models/__init__.py`, `src/ainovel/db.py`。

**Interfaces:** 定义 `MemoryCardVersion`（项目、版本、父版本、DRAFT/APPROVED、条目JSON、来源指纹、批准者/时间）、`StoryMemoryEntry`（来源/关联/有效范围/候选scope、supersedes_id）、`WorkflowContextPolicy`（工作流、策略版本、卡版本、固定来源版本、原策略引用、预览指纹、激活状态）。发布后内容不可变，状态转移单独校验。

- [ ] 写迁移测试：带现有阶段与工作流外键的0008数据库升级后，原行值不变、外键检查为空；新卡默认DRAFT，旧工作流无绑定时解析为legacy。
- [ ] 运行 `pytest tests/test_story_memory_migration.py -v`，确认新表/接口缺失导致失败。
- [ ] 实现模型、契约和0009迁移，更新readiness版本；新表有项目/工作流外键及版本唯一约束，降级遇到已使用新记忆数据时明确拒绝而非静默丢失。
- [ ] 运行迁移测试及 `tests/test_foundation_acceptance.py`，验证升级/拒绝破坏性降级/空库往返。
- [ ] 审查并仅提交本任务变更，建议提交名 `feat: persist versioned story memory`。

## Task 2：来源索引和作者约束卡

**Files:** Create `src/ainovel/services/story_memory.py`, `tests/test_story_memory.py`; Modify `src/ainovel/services/context.py`（来源注册及索引适配）。

**Interfaces:** `StoryMemoryService.create_card(project_id: str, entries: list[MemoryEntryInput], actor: str) -> MemoryCardVersion`；`approve_card(card_id: str, expected_fingerprint: str, actor: str) -> MemoryCardVersion`；`ensure_index(project_id: str) -> str` 返回来源指纹；`validate_refs(refs: list[SourceRef]) -> None`。

- [ ] 写测试：错误项目/哈希/段落偏移被拒绝；自由文本导入不自动批准；空约束卡不能视为所有规则已覆盖；别名无FTS命中也不丢明确关联；结构化无损导入可追溯。
- [ ] 运行 `pytest tests/test_story_memory.py -v`，确认失败。
- [ ] 实现按字段/段落切分，分块保留准确范围和共同来源。超大段落保留为不可装入的来源，不截断。作者确认覆盖哪些原文区段；未覆盖且分类不明的区段成为blocker，不通过“摘要完成”自动消失。
- [ ] 扩展官方索引到已批准阶段规划点；来源变动使卡失效，保留旧索引引用/包快照，不覆盖审计历史。纯本地检查不能宣称识别全部语义矛盾。
- [ ] 验证所有保存、确认、重建调用模型次数为0；运行既有 `tests/test_context.py`；审查并提交 `feat: add approved memory cards and provenance`。

## Task 3：当前剧情点与连续性选择

**Files:** Create `src/ainovel/services/scoped_context.py`, `tests/test_scoped_context_selection.py`; Modify `src/ainovel/services/stages.py`（提供局部上下文路径，不删除legacy接口）。

**Interfaces:** `ScopedContextService.select(workflow_id: str, step_id: str) -> MemorySelection`；消费任务1契约及任务2来源校验。以StageWorkflow/StageWorkflowNode预留slots定位，不按用户当前页面或最新未确认进度猜测。

- [ ] 写测试：批次跨PP1末章和PP2首章带齐两点，未选PP3正文；必需依赖保留短结果；依赖成环拒绝；结局方向标记author_only，尚未揭露细节不得变成角色已知事实。
- [ ] 写隔离测试：同名人物跨项目、其他工作流候选不可见；同批前序候选标未批准，当前/后序候选不得进入前章请求。
- [ ] 运行 `pytest tests/test_scoped_context_selection.py -v`，确认失败。
- [ ] 实现确定性必需集合，再用实体ID/别名和FTS补充可选集合；复用相关正式事实有效章节范围与前章摘要。显式依赖缺少摘要时报告缺口，不自动调用LLM或展开全部历史。
- [ ] 运行选择测试和 `tests/test_plot_point_stages.py`，通过后审查并提交 `feat: select plot-scoped writing memory`。

## Task 4：统一任务包与输入预检

**Files:** Modify `src/ainovel/services/scoped_context.py`, `src/ainovel/workflows/orchestrator.py`, `src/ainovel/services/context.py`; Create `tests/test_scoped_context_budget.py`, `tests/test_scoped_workflow_requests.py`。

**Interfaces:** `ScopedContextService.preview(workflow_id: str, step_id: str) -> ContextPreview`；`build_request(workflow_id: str, step_id: str) -> ModelRequest`。两者共享同一纯组装逻辑，发送前重新验证来源指纹；包快照在现有调度事务边界落地。

- [ ] 写测试：scoped请求无重复的project_constitution/official_outline_tree/完整roadmap，必要约束文本确实在包内而非仅有ID；同来源跨payload和packet去重。
- [ ] 写预算测试：当前配置16000仍按16000检查；输入/输出/安全余量共同生效，必需超限得到deficit和来源列表且provider calls=0；optional剔除理由可见。
- [ ] 运行两个新测试文件，确认失败。
- [ ] 实现策略分流：legacy保持原路径，scoped用统一包；写作、覆盖检查、摘要、批次审查分别带自身必需稿件/目标，不因记忆检索删掉审查正文。保持单次调度最多一次生成请求。
- [ ] 运行新测试与 `tests/test_orchestrator.py tests/test_context.py`；审查并提交 `feat: assemble bounded scoped context packets`。

## Task 5：批准更新、失效与工作流转换

**Files:** Create `src/ainovel/services/context_policy.py`, `tests/test_context_policy.py`; Modify `src/ainovel/services/batches.py`, `src/ainovel/services/story_memory.py`, `src/ainovel/services/workflows.py`。

**Interfaces:** `ContextPolicyService.activate(workflow_id: str, card_id: str, expected_preview_fingerprint: str, actor: str) -> WorkflowContextPolicy`；`revert(workflow_id: str, expected_policy_version: int, actor: str) -> None`；`StoryMemoryService.promote_batch(batch_id: str, actor: str) -> None`（仅flush，由批准事务提交）。

- [ ] 写测试：批准章节、正式事实和进度全部成功或全部回滚；重复批准幂等/冲突安全；取消、驳回不正式化；索引落后时重建或阻塞，不返回旧事实。
- [ ] 写转换测试：仅无模型调用、无产物、无有效租约的输入超限工作流可切换；预览后来源/修订变更拒绝；保存旧策略及审计；未调用前可撤回。
- [ ] 运行 `pytest tests/test_context_policy.py -v`，确认失败。
- [ ] 实现带版本比较的事务切换，不自动resume/调用；已有产物或陈旧来源仅返回不可原地转换理由。提供继任方案预览，创建仍使用现有取消/新建审批入口，禁止双重占用slots。结构化事实缺来源或无法映射的自由state_delta保持候选/待确认，不编造正式事实。
- [ ] 验证 `tests/test_batches.py tests/test_workflows.py tests/test_context_policy.py`，审查并提交 `feat: guard memory promotion and context migration`。

## Task 6：作者页面与现有任务入口

**Files:** Create `src/ainovel/web/memory_routes.py`, `src/ainovel/web/templates/memory.html`, `tests/test_memory_web.py`; Modify `src/ainovel/app.py`, `src/ainovel/web/workflow_routes.py`, `src/ainovel/web/templates/workflow.html`。

**Interfaces:** GET `/projects/{project_id}/memory`；POST同路径 `/cards`、`/cards/{card_id}/approve`；GET `/workflows/{workflow_id}/context-preview`；POST `/workflows/{workflow_id}/context-policy`（card_id、fingerprint、author_confirm、CSRF）；POST `/workflows/{workflow_id}/context-policy/revert`。

- [ ] 写页面测试：显示当前剧情点、章号、卡是否有效、估算前后对比/缺口/来源；GET不改变状态、不调用模型；未确认或缺CSRF返回422/403并保留输入；跨项目card_id拒绝。
- [ ] 运行 `pytest tests/test_memory_web.py -v`，确认失败。
- [ ] 实现“预览精简后的写作输入”“确认采用记忆检索模式”和“继续生成”分离。作者可编辑约束卡和局部事实关联；JSON编辑可作高级入口，但基本文案须能说明规则覆盖与未覆盖来源。
- [ ] 在页面注明旧流程、候选/正式、作者级秘密信息与可叙述内容；可选AI提取仅显示未启用说明，本期无隐藏API入口。
- [ ] 运行页面与既有workflow/stage测试，审查并提交 `feat: expose author-reviewed memory controls`。

## Task 7：当前工作流离线验收与安全部署

**Files:** Create `tests/test_scoped_memory_acceptance.py`, `scripts/preview_scoped_context.py`, `docs/scoped-story-memory.md`。

**Interfaces:** CLI `--database PATH --workflow ID --card ID` 只读诊断；以数据库副本测试需要落地索引的操作，禁止直接修改原库或打印密钥/全量小说。

- [ ] 写验收测试：五章跨点、超长原文、必需溢出、跨项目候选、源版本变更；请求无全文重复，call count与旧策略相同，审批后下一批进度正确。
- [ ] 运行新测试，确认失败；补齐前六任务集成缺口，不旁路预算或审批。
- [ ] 对当前工作流复制库做基线比对。尚无作者确认约束卡时，只报告待确认材料与预算估算，不伪造批准；缺卡阻塞也是正确结果，不承诺立刻生成。
- [ ] 运行完整 `pytest`，独立审查后修复重要问题，记录通过/跳过/失败及未验证的真实模型行为。
- [ ] 作者允许部署后，无活动请求时备份数据库、应用迁移、重启本地测试服务、验证页面200与旧记录数/外键。只展示转换预览；具体卡批准和旧工作流转换仍需作者确认。
- [ ] 审查后仅提交本任务文件，文档列迁移、回退、手动确认步骤和已知语义检索局限；不得自动push/PR或真实生成。

## 执行环境与命令

在 `D:/ainovel/.worktrees/qwen-adapter` 复用现有checkout，先确认现有任务不冲突。PowerShell设置 `$env:PYTHONPATH='D:/ainovel/.worktrees/qwen-adapter/src'`，使用 `D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe -m pytest ...`，防止错误导入其他工作树。执行迁移必须显式指定AINOVEL_DATABASE_URL，并先备份。

## 计划自检与交接

设计各段已映射到任务：存储/约束卡1–2，选取3，预算4，审批及当前工作流兼容5，页面6，验收7。Review Focus均有对应测试。首期可选AI提取不实现调用，避免扩大费用与模型依赖；若作者要求该能力，另行明确授权和测试。

尚未执行任何任务。建议本会话顺序实施、最后独立审查：七项任务共享来源契约与事务边界，不宜并行改同一上下文组装路径。等待作者审阅计划并选择执行方式。

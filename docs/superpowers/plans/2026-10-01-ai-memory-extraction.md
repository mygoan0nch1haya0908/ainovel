# AI 分层记忆提取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AI 从现有原文生成分层记忆候选，保留人工编辑、锁定与批准。
**Architecture:** 独立持久化提取任务复用现有模型档案、AgentRunner、来源校验和记忆卡；本地确定性分块，串行领取调用，人工差异合并，不改写作检索语义。
**Tech Stack:** Python、FastAPI/Jinja、SQLAlchemy/SQLite/Alembic、Pydantic、pytest。
**Spec:** `docs/superpowers/specs/2026-10-01-ai-memory-extraction-design.md`

## Global Constraints

- 每块完整请求输入上限64000估算tokens，输出预算64000tokens。有效输出取64000与模型档案/供应商最大输出能力较小值；有效输入=min(64000, 上下文窗口-有效输出预算-1024)，非正则调用前阻止。单次授权最多8次串行请求，最多预留512000输出tokens；页面展示实际有效预算并明确确认。预算是上限，不要求填满，提取内容仍精简；不得擅自提高模型配置能力。
- 每块每次授权最多一次尝试；无自动重试、fallback、修复调用。未知usage保持None。
- GET不调用模型；不自动批准/合并/恢复写作；真实请求必须由页面明确授权。
- 开发使用FakeProvider和临时数据库，不读取密钥值、不调用真实模型、不升级运行库。
- 当前工作树包含大量既有修改。每项只暂存可隔离文件；无法隔离时记录延后提交，禁止整树提交或还原。
- 当前会话顺序实施，最终一次独立审查；执行前核对AGENTS、隔离工作树、规格与本计划。

## Review Focus

1. 请求已发出但进程崩溃：保留未知尝试，不因重启重发（任务3）。
2. 分块后长中文、空白、元数据或删除来源漏覆盖：引用范围与指纹失效（任务2、4）。
3. 页面授权重放、双击或另一路写作同时启动：一次领取、项目级互斥（任务3、5）。
4. AI伪造ID或作者编辑期间合并覆盖锁定项：服务端ID归属、基础卡CAS（任务4）。
5. 模型配置撤销/能力降低、错误响应仍产生用量：发送前复查、未知用量与预算保留（任务3）。

## Task 1：持久化任务及兼容契约

**Files:** 新增 `src/ainovel/models/memory_extraction.py`、`alembic/versions/0010_memory_extraction.py`、`src/ainovel/agents/memory_extraction_contracts.py`、`tests/test_memory_extraction_schema.py`；修改 models/__init__.py、db.py、agents/memory_contracts.py。

**Interfaces:** MemoryExtractionJob（项目、模型档案版本、基础卡、来源/规则快照、修订号、状态、合并卡ID）；MemoryExtractionChunk（任务、序号、来源片段、请求快照、缓存键、结果、状态）；MemoryExtractionAttempt（块、授权ID、时间、可空usage、错误码、状态）。授权ID唯一消费并存8次/512000输出预算上限，按实际有效输出逐次预留。条目增加 entry_id、origin、author_locked 默认兼容字段；旧记录读取时通过卡ID/条目位置生成稳定身份，不改写旧卡JSON。模型输出不允许指定这些作者控制字段。

- [ ] 写 `test_upgrade_preserves_cards_and_workflows`、`test_unknown_usage_nullable`、`test_old_entries_still_parse`：旧卡原样、无自动策略转换、usage为None。
- [ ] 运行 `pytest tests/test_memory_extraction_schema.py -v`，确认RED。
- [ ] 实现上述模型/迁移/契约，迁移显式建表；有任务数据时拒绝破坏性降级；更新head断言。job状态含DRAFT、READY、RUNNING、PAUSED、NEEDS_REVIEW、STALE、MERGED、CANCELLED；块状态含PENDING、RUNNING、SUCCEEDED、FAILED、UNKNOWN、REUSED、CANCELLED。
- [ ] 运行本测试及 `test_story_memory_migration.py`，全部GREEN；按明确路径提交或记录依赖既有脏文件而延后。

## Task 2：来源分块、覆盖与增量缓存键

**Files:** 新增 `src/ainovel/services/memory_extraction_sources.py`、`tests/test_memory_extraction_sources.py`；修改 services/story_memory.py。

**Interfaces:** `freeze_sources(session, project_id, pending_ids: list[str]) -> dict`；`build_chunks(snapshot: dict, base_entries: list[dict], profile: dict) -> list[dict]`。块包含稳定局部source_id、原引用、相对区间、完整ModelRequest快照、cache_key。规则版本常量 `memory_extract_v1`。

- [ ] 写 `test_long_chinese_roundtrip_ranges`：拆分区间覆盖原段、无重叠/遗漏；`test_full_request_budget_includes_schema`：序列化完整请求≤有效输入预算；`test_metadata_is_not_manual_prose`：outline顶层title/kind/author_locked作为元数据，payload的数值约束仍保留；`test_changed_or_deleted_source_invalidates_cache`：来源、模型/档案能力、规则、关联元数据或锁定基础变化时缓存失效。
- [ ] 运行 `pytest tests/test_memory_extraction_sources.py -v` 确认RED。
- [ ] 实现来源按原段落确定性拆分，不跨来源拼成无法归因的大段；过长区间二分直至完整请求合格。来源ID由服务端生成，source snapshot含正文但页面日志不输出。标题/锁定等元数据变化纳入指纹；固定开销已超预算则调用前阻止。缓存只选同项目且重新校验后仍有效的成功块。
- [ ] 运行本测试及现有story_memory/selection测试，全部GREEN；按范围提交或记录延后。

## Task 3：串行授权、调用互斥与故障恢复

**Files:** 新增 `src/ainovel/services/memory_extraction.py`、`src/ainovel/services/project_llm_guard.py`、`tests/test_memory_extraction_execution.py`；修改 services/stages.py、workflows/orchestrator.py（仅领取互斥接入）。

**Interfaces:** `MemoryExtractionService(session, provider_resolver=None)` 的 `prepare(project_id, profile_version_id, base_card_id, pending_ids, actor) -> MemoryExtractionJob`、`authorize(job_id, expected_revision, authorization_id, actor) -> dict`、`run_next(job_id, authorization_id, provider=None) -> MemoryExtractionJob`、`cancel(job_id, expected_revision, actor)`、`mark_interrupted(job_id, expected_revision, actor)`。`ProjectLLMGuard.claim/release(session, project_id, owner_id)` 通过同一项目行锁及持久化owner记录排他，写作/规划/提取全部在发送前领取、响应后释放；未知中断需人工核对，不根据时间自动重发。

- [ ] 写 `test_nine_chunks_only_eight_calls`（仅8请求）、`test_replayed_authorization_no_extra_calls`、`test_same_project_writing_and_extract_exclusive`（两Session）、`test_timeout_keeps_unknown_usage_and_no_retry`、`test_crash_does_not_resend_running_chunk`、`test_revoked_profile_no_dispatch`、`test_actual_usage_overrun_pauses`、`test_cancel_preserves_inflight_result`。
- [ ] 运行 `pytest tests/test_memory_extraction_execution.py -v` 确认RED。
- [ ] 按StageService已有持久化尝试模式实现：事务领取并提交后调用，结果CAS回写；每次run_next只调用一次，最多8次授权，失败立即暂停。capabilities/配置每次发送前复查；本地估算和供应商用量独立保存；timeout沿用模型档案现有设置不另加无限等待。原文/基础卡变化标STALE不采用。取消只停止待发送块，在途结束仍留审计。复用safe_failure_code与安全诊断，不保存密钥。
- [ ] 运行本测试及stage/orchestrator/provider回归，全部GREEN；提交或记录延后。

## Task 4：候选验证、稳定身份、人工锁定与合并

**Files:** 新增 `src/ainovel/services/memory_extraction_merge.py`、`tests/test_memory_extraction_merge.py`；修改 services/story_memory.py、services/scoped_context.py（兼容新字段映射，不能直接getattr正式行不存在的字段）。

**Interfaces:** `validate_chunk_result(chunk: dict, result: dict) -> dict` 返回entries与unresolved；`MemoryExtractionMergeService(session).preview(job_id) -> dict` 返回新增/修改/删除/锁定冲突及基础指纹；`merge(job_id, expected_revision, expected_fingerprint, edited_entries, actor) -> MemoryCardVersion`。

- [ ] 写 `test_forged_cross_chunk_reference_rejected`、`test_unresolved_blocks_complete_merge`、`test_locked_entries_unchanged`、`test_new_card_during_review_rejects_merge`、`test_manual_edit_creates_new_draft_not_approval`、`test_repeated_merge_returns_same_card`、`test_legacy_entries_keep_identity`。
- [ ] 运行 `pytest tests/test_memory_extraction_merge.py -v` 确认RED。
- [ ] 还原服务端SourceRef并校验区间、类型、关联、覆盖；模型只能提出本块条目及已有允许身份的更新建议，未知身份拒绝。新ID服务端分配；锁定的每一字段保持，冲突独立显示。待确认来源无条目覆盖时不可完成。合并事务复查来源/基础卡/修订号；人工编辑沿用契约，自动提取无权解锁；解锁走人工新版本确认。差异只确认成DRAFT，批准仍用既有入口。
- [ ] 运行本测试及memory/selection/approval回归，全部GREEN；提交或记录延后。

## Task 5：AI与人工双入口页面

**Files:** 新增 `src/ainovel/web/memory_extraction_routes.py`、`src/ainovel/web/templates/memory_extraction.html`、`src/ainovel/web/templates/memory_extraction_diff.html`、`src/ainovel/static/memory_extraction.js`、`tests/test_memory_extraction_web.py`；修改 app.py、web/memory_routes.py、templates/memory.html。

**Interfaces:** POST `/projects/{project_id}/memory/extractions` 仅准备；GET `/memory-extractions/{job_id}` 只读；POST `.../authorize`、`.../step`、`.../cancel`、`.../interrupted`；GET `.../diff`；POST `.../merge`。全部POST检查CSRF及项目归属，authorize展示并确认endpoint/model/范围/调用预算。step由页面串行await最多8次，失败停止，无JS自动重试；页面关闭后停止发后续块。无JS也可手动执行下一块。

- [ ] 写 `test_get_and_prepare_never_call_model`、`test_csrf_and_cross_project_rejected`、`test_authorize_double_submit_no_double_budget`、`test_manual_form_and_json_remain_usable`、`test_preview_hides_keys_and_escapes_model_text`、`test_diff_confirmation_not_auto_approval`。
- [ ] 运行 `pytest tests/test_memory_extraction_web.py -v` 确认RED。
- [ ] 实现模型档案选择、预算预览、串行状态进度、人工编辑/锁定和差异确认。缺失字段/冲突显示中文定位，失败保留已有结果。清楚提示中断后的用量未知，禁止页面刷新自动调用。浏览器返回JSON携带状态/预算，不含credentials。
- [ ] 运行页面测试及现有memory_web，全部GREEN；提交或记录延后。

## Task 6：端到端验收、文档及独立审查

**Files:** 新增 `tests/test_memory_extraction_acceptance.py`；修改 `docs/scoped-story-memory.md`。

- [ ] 写并先运行失败验收：长设定→多块串行→人工修正与锁定→差异合并→作者批准→scoped写作选择；修改一个来源后只重算失效块，未修改成功块不调用；验证旧空卡/旧工作流保留，未批准AI结果不可检索。
- [ ] 完成必要接线并运行 `pytest tests/test_memory_extraction_acceptance.py -v` 至GREEN；不加入绕过验证的演示数据。
- [ ] 更新用户说明：操作、预算、人工接入、未知用量恢复、部署备份/迁移/回退；不存在“保证不矛盾”承诺。
- [ ] 使用现有venv、PYTHONPATH运行全量 `python -m pytest -ra --basetemp=<独立目录>` 与 `git diff --check`；记录真实结果，live skip注明未验证。
- [ ] 一次独立只读审查范围为本计划改动；处理重要发现，以回归测试RED→GREEN证明；复跑全量测试。未解决项明确列出。
- [ ] 交付未部署版本及明确文件范围。部署另行确认：无活动调用→备份→0010迁移→重启→页面/外键检查，不自动开始真实提取。

## 自查结果与执行交接

已检查规格覆盖、跨任务接口、五类风险测试与隐私边界。采用现有顺序执行方式，不再要求用户重选。等待作者确认本计划后开始实现；不将规格批准当作计划批准。当前未修改产品代码。

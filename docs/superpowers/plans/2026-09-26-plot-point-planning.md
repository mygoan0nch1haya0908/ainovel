# Plot-point Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 阶段按“剧情点＋分配章节数”规划，写作前仅细化下一批 1—5 章。

**Architecture:** 增加显式版本契约与纯函数章节映射，复用现有阶段、工作流、审批和进度事务。新格式使用专用批次计划契约；历史冻结请求及逐章格式保持原行为，不迁移数据。

**Tech Stack:** Python、Pydantic、SQLAlchemy、FastAPI/Jinja、pytest。

**Spec:** `docs/superpowers/specs/2026-09-26-plot-point-planning-design.md`

## Global Constraints

- 新格式固定 `format: plot_points_v1`；无标记走旧格式，未知标记拒绝，不猜测。
- 最多 30 个剧情点；每点 1—100 章；合计最多 500 章；point_id 最多 32 个 ASCII 字母、数字、下划线或连字符。
- 每批 1—5 章；正文 4500—6000 可见字符；计划和正文各自必须作者确认。
- 已开始且确认过正文的剧情点整体锁定；仅未开始部分允许调整。
- 旧格式已有确认正文禁止转换；零确认正文可显式新建剧情点提案，旧版本保留。
- 保存意见、建立提案、预留批次均不调用模型；真实模型验证需另行授权。
- 保留配置绑定、预算、异常分类、有限重试及隐私防护。预计不新增迁移或依赖。
- 在现有 `D:/ainovel/.worktrees/qwen-adapter` 工作；先记录 dirty baseline，保留既有未提交改动。每任务仅提交本任务可分离的改动；重叠文件不能安全分离时保留未提交并说明，不批量 git add/reset。

## Review Focus

1. 跨剧情点且最后不足五章：槽位连续唯一，不能越界（任务 1、3）。
2. 规划生成失败后重试或旧请求恢复：仍用原冻结格式，不能默认切换（任务 2、4）。
3. 仅确认某点第一章后修改该点后续安排：整点锁定，审批拒绝（任务 2）。
4. 模型返回同数量但错误槽位/归属：不能保存或批准错误计划（任务 4）。
5. 双击、陈旧审批及正文驳回：不重复推进、不改变已固定映射（任务 3、6）。

## File Map

- `agents/stage_contracts.py`：新增剧情点契约，旧契约不变。
- 新增 `services/stage_planning.py`：无数据库依赖的格式解析、区间、槽位、锁定验证。
- `models/stage.py`、`services/stages.py`：总章数、提案/生成/审批、映射、上下文、进度。
- `agents/contracts.py`、`agents/prompts.py`、`services/workflows.py`、`workflows/orchestrator.py`：专用计划 schema 冻结及所有校验入口。
- `providers/demo.py`：离线示例新旧格式支持。
- `web/stage_routes.py`、`web/chapter_test_routes.py`、`web/templates/stage.html`：显式创建、格式展示、反馈差异和进度。
- 新增 `tests/test_plot_point_planning.py`；扩展 `tests/test_stages.py`、`tests/test_stage_web.py`、`tests/test_workflows.py`、`tests/test_orchestrator.py`。
- 新增 `docs/plot-point-planning.md`：使用步骤、兼容边界和限制。上述代码路径以 `src/ainovel/` 为根。

## Verification Commands

PowerShell，每次测试前设置 `$env:PYTHONPATH='D:/ainovel/.worktrees/qwen-adapter/src'`；使用 `D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe`，下文简称 `$py`（设置为该路径）。关闭真实 OpenAI/Qwen/Ollama 测试，移除当前测试进程的 API key 和 DATABASE_URL 环境覆盖，绝不修改用户保存配置。

定向命令：`& $py -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' <测试文件> -q --tb=short`。不得并发复用 basetemp。RED 必须因预期功能缺失失败；GREEN 必须 exit 0。

### Task 1: 版本契约和纯函数映射

**Interfaces:** 新增 `StagePlotPoint`、`StagePlotRoadmapDraft`。后者字段为 format、goal、start_state、end_state、key_events、foreshadowing、points；文本沿用 CompactText（500），阶段列表最多 100；点包含 point_id、ordinal、title、goal、key_events、character_changes、foreshadowing、chapter_count、dependencies，点内列表最多 30，事件至少一项，其余可空。严格正整数、唯一 ID、连续顺序、仅向前且不重复依赖。

新增 `parse_stage_roadmap(payload: dict) -> StageRoadmapDraft | StagePlotRoadmapDraft`；`roadmap_chapter_count(payload: dict) -> int`；`plot_point_ranges(payload: dict) -> list[dict]`（point_id/start/end）；`chapter_slots(payload: dict, confirmed: int, requested: int) -> list[dict]`（node_id/point_id/point_ordinal/stage_ordinal，旧格式 point 字段省略）；`validate_locked_prefix(previous: dict, candidate: dict, confirmed: int) -> None`。槽位 ID 固定 `point_id + ':' + point_ordinal`，不碰 point_id 允许字符集，长度小于 64；范围从 1 起闭区间。

- [ ] 在 `test_plot_point_planning.py` 写 RED：2+4 章区间是 [(1,2),(3,6)]；confirmed=1/requested=5 返回阶段章号 [2,3,4,5,6]、点内章序 [2,1,2,3,4]；confirmed=5 返回一章，全部完成返回空；拒绝负 confirmed 和 requested 不在 1—5。
- [ ] 添加参数化 RED：0/-1/1.5/字符串/布尔章数，31 点，101 单点，501 总数，重复 ID、无效依赖、未知 format；旧 fixture 原样成功。运行定向测试确认 RED。
- [ ] 实现上述契约、纯函数和 `StageRoadmapVersion.estimated_chapters` 分支。锁定验证旧格式沿用节点前缀，新格式锁定所有 start <= confirmed 的完整点及其顺序；跨格式 confirmed > 0 拒绝。
- [ ] 运行该测试文件 GREEN；审查仅本任务 diff，提交 `feat: add versioned plot-point contracts and chapter slots`。

### Task 2: 提案、冻结格式、反馈和审批锁定

**Interfaces:** `StageService.propose_roadmap` 增加 keyword-only `roadmap_format: str = 'plot_points_v1'`；反馈必须以来源 payload/冻结 schema 为准，不受默认值影响。旧已确认阶段的新普通提案明确拒绝跨格式，旧反馈继续可用。使用任务 1 的解析及锁定函数，保持既有方法返回类型。

- [ ] `test_stages.py` 增加 RED：新提案 schema 为剧情点；旧待生成快照仍解析旧格式；旧反馈仍冻结旧 schema；同一请求失败重试格式不变。
- [ ] 增加 RED：confirmed=1 的首点任何内容/章数修改审批失败，第二未开始点可改；confirmed>0 的旧格式转换失败；零确认正文转换保留旧记录；提案/反馈 provider 调用次数为 0。运行确认 RED。
- [ ] 更新 `services/stages.py` 提示词与生成 schema 分支：约 60 章建议 8—12 点，不强制；只输出点级内容。审批在原事务和 revision 检查内执行锁定验证，不覆写历史快照；全局约束与去重继续生效。
- [ ] 运行 `test_stages.py` 和任务 1 测试 GREEN，提交可分离改动 `feat: support plot-point roadmap proposals and revisions`。

### Task 3: 连续章节预留、上下文与进度

**Interfaces:** `start_next_batch` 保持签名，以 `chapter_slots` 生成现有 StageWorkflowNode。`workflow_context` 保持签名；新格式返回 `format`、`slots`（增加 ordinal/book_ordinal）、`points`（当前涉及点）；仅 include_roadmap=True 添加紧凑全点图。写作 ordinal 指定时仅当前槽位及所属点，不传未来细化内容。旧返回结构保持不变。

- [ ] `test_stages.py` 增加 RED：跨点槽位唯一并正确绑定 stage/book ordinal；末尾不足请求数时实际创建剩余章节数；无剩余章拒绝；新提案不能改变旧工作流 context。
- [ ] 增加 RED：建立/生成计划不推进 confirmed；正文批准后跨点累计正确且重复批准不重复累计；驳回/失败不推进。运行确认 RED。
- [ ] 修改 `services/stages.py` 的预留、context 和进度校验；继续使用原项目锁、事务、固定 roadmap_id 和 committed_batch_id 幂等保护，不新增表。
- [ ] 运行阶段测试 GREEN，提交 `feat: reserve plot-point chapter slots and track approved progress`。

### Task 4: 写作前细化和贯穿校验

**Interfaces:** `PlotPointChapterPlan(ChapterPlanV2)` 增加 `slot_id: str`、`point_id: str`、`point_ordinal: int`（严格正整数）；`PlotPointBatchPlanDraft` 保留 chapters 1—5 和连续 ordinal 验证。`WorkflowService.start` 增加内部 keyword-only `_planning_format: str | None = None`；StageService 为新格式传 plot_points_v1，其他调用不变。新增 `planning_schema_for_workflow(session, workflow_id) -> type[BaseModel]` 到 `services/stage_planning.py` 不合适（纯函数边界）；应在 `services/workflows.py` 定义，以固定 StageWorkflow 的 roadmap 格式选 schema，未关联时沿用 generation_version。生成前冻结新 planner schema 和提示词；校验同时覆盖 AgentRunner、orchestrator 业务检查及 artifact 保存入口。

- [ ] `test_orchestrator.py`/`test_workflows.py` 写 RED：同点两章可有不同标题/目标，场景预算仍在 4500—6000；错误 slot_id、point_id、point_ordinal、数量或顺序均失败；旧计划仍要求原标题/目标。
- [ ] 写 RED：直接 artifact 保存错误归属被拒绝；重启恢复新工作流仍用新 schema；未批准计划不能写正文；writer context 没有全阶段未来场景。运行确认 RED。
- [ ] 在 contracts/prompts/workflows/orchestrator 落实统一分支，StageService 传内部参数。新提示词要求按点内进度分配局部目标/场景/章末钩子，不要求每章完成整点；不得将点目标复制成所有章节目标。保持 writer/reviewer schema 不变。
- [ ] 运行阶段、工作流、orchestrator 测试 GREEN，提交 `feat: refine reserved plot-point slots into chapter plans`。

### Task 5: 作者页面、版本差异和显式启用

**Interfaces:** 保留现有路由；新建表单默认新格式，历史反馈走来源格式。路由层生成只读 `planning_view`（format、total_chapters、points，每点含区间与 confirmed_count）；从累计范围计算进度 clamp(confirmed-start+1,0,chapter_count)。差异展示前后点内容、章数和派生区间，不仅顶层字段名。

- [ ] `test_stage_web.py` 写 RED：60 章页面显示点/范围/总章数及展开详情；已确认 3 章、首点 6 章显示 3/6；有旧确认正文时说明不能转换；零确认旧阶段展示独立“新建剧情点规划”入口。
- [ ] 写 RED：保存意见不调用模型；新提案生成须显式确认；反馈不切格式；章节数改变后的后续范围差异可见；恶意标题/意见转义、CSRF 与失败表单保留回归。运行确认 RED。
- [ ] 更新 routes/template 与 chapter_test 新建入口；只做必要样式补充，保留收起展开、原作者审批按钮及模型确认。
- [ ] 运行 web/阶段测试 GREEN，提交 `feat: show plot-point allocation and revision differences`。

### Task 6: 离线完整流程与交付

**Interfaces:** DemoProvider 按请求冻结 output schema 选择新旧 stage/planner 输出；不按模型名猜格式。新增 `tests/test_plot_point_acceptance.py` 使用临时数据库和 Demo/Fake，不接真实账户。

- [ ] 写离线 RED：2+4 章阶段→作者批准→预留 5 章跨点→细化→计划批准→有效正文→正文批准→confirmed=5→下一批仅 1 章；每处审批门槛单独断言；陈旧/重复提交及驳回不推进。运行确认 RED。
- [ ] 扩展 DemoProvider 新格式与上述流程，保留旧 fixture；更新用户文档说明不保证消除供应商超时、只减少规划输出量。
- [ ] 定向 acceptance GREEN，然后执行全量 `& $py -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short`；记录实际数量与 exit code，不借用历史测试结果。
- [ ] `git diff --check`；审查所有 schema 入口、上下文泄漏、审批竞态和旧格式兼容。仅提交可分离改动 `test: cover plot-point planning acceptance and compatibility`。
- [ ] 离线验证通过后，确认本地服务无活动任务再按既有隐藏启动方式重载；只 GET 检查测试页可达与新入口，不自动付费生成或批准。交付测试入口、测试结果及兼容限制，不推送/建 PR（本次未要求）。

## Self-review and Handoff

已逐项检查规格覆盖、接口衔接、五项 Review Focus 的测试归属及文档比例。任务 4 特别覆盖冻结请求与 artifact 二次校验，避免仅在模型输出解析处新增格式。计划批准后再实施；推荐同一会话顺序实施，因阶段事务、上下文和冻结 schema 依赖紧密且工作树含既有改动，末尾独立审查。

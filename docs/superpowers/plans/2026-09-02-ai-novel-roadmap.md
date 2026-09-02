# AI 超长篇小说生成器实施路线图

**设计规格：** `docs/superpowers/specs/2026-09-02-ai-novel-agent-design.md`

## 技术基线

- 本地优先的 Python Web 应用；本机现有 Python 3.14.2。
- FastAPI 提供 HTTP 和服务端页面。
- Jinja2 + 本地静态 JavaScript 提供界面，不引入 Node.js 构建链。
- SQLite + SQLAlchemy 2 保存小说、版本、大纲、章节、事实和审计记录。
- Alembic 管理数据库迁移。
- Pydantic 2 定义 Agent 任务包和结构化输出。
- pytest + FastAPI TestClient 完成单元、集成和工作流测试。
- 模型访问通过 `ModelProvider` 接口隔离；OpenAI 实现使用 Responses API 和 JSON Schema 结构化输出，不把 API 响应当作长期记忆。

## 分阶段计划

### 阶段一：项目基础、版本状态和审批事务

计划：`docs/superpowers/plans/2026-09-02-ai-novel-foundation.md`

交付一个可运行的本地 Web 骨架，支持创建小说项目、版本化大纲、创建候选批次、保存候选章节、批准或退回批次，以及查询审计记录。该阶段不调用真实模型。

### 阶段二：模型适配、多 Agent 编排和上下文工程

交付 `ModelProvider`、OpenAI Responses API 适配器、提示词注册表、结构化 Agent 任务包、上下文预算器、检索式事实包、顺序章节生成器和可恢复工作流。使用假模型完成确定性集成测试，真实 API 测试默认跳过。

### 阶段三：连续性、文风和作者内容融合

交付人物知识边界、时间线、资源账本、因果链、伏笔生命周期、矛盾分级、文风指纹、千字样章审批、作者原稿/建议稿/确认稿三版本差异流程，以及已发布正文保护策略。

### 阶段四：完整作者工作台和验收

交付立项向导、三个创意方案比较、分层大纲浏览、批次计划审批、章节编辑与差异、状态查询、风险中心、版本恢复、导出和端到端验收。平台规则检索作为有来源、有时间戳的可选工具，不写死规则文本。

## 阶段依赖

```text
阶段一：正式/候选状态与版本事务
  ↓
阶段二：Agent 编排与上下文
  ↓
阶段三：连续性、文风、插章融合
  ↓
阶段四：完整作者工作台与验收
```

每个阶段都必须交付可运行、可测试的软件。不得在前一阶段的数据边界尚不可靠时接入真实模型批量生成。

## OpenAI 接口约束

OpenAI 适配器使用 Responses API。结构化 Agent 输出使用 JSON Schema，而不是只保证语法有效的旧式 JSON 模式。自定义工具参数和返回值必须有明确类型，响应正文通过 SDK 的聚合文本属性读取，不能假定输出数组第一项一定是文本消息。应用自己的 SQLite 状态是唯一业务事实来源，API 会话标识只能作为调用元数据保存。

参考：

- https://developers.openai.com/api/reference/cli/resources/responses/methods/create
- https://developers.openai.com/api/docs/guides/latest-model

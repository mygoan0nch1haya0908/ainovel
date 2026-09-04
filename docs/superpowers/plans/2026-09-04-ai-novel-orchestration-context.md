# AI Novel Orchestration and Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-verifiable, resumable 1–5 chapter generation workflow with Fake, Ollama, and OpenAI providers, bounded retrieval context, two author approval gates, and a minimal local Web workflow UI.

**Architecture:** A synchronous `WorkflowOrchestrator` advances one persisted step at a time and is the only component allowed to coordinate model artifacts into the existing candidate-batch service. Provider calls are isolated behind a typed contract; prompts, context packets, attempts, and artifacts are immutable snapshots in SQLite. Default tests use a deterministic Fake provider and local fake HTTP transports, so no network or paid API key is required.

**Tech Stack:** Python 3.14, FastAPI, Pydantic 2, SQLAlchemy 2, SQLite FTS5, Alembic, httpx 0.28, OpenAI Python SDK 3.x, Jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-ai-novel-orchestration-context-design.md`

## Global Constraints

- Keep the app loopback-only and validate session-backed CSRF on every POST.
- Default test runs must make zero external network calls and require no API key.
- `FakeProvider` is the mandatory deterministic acceptance path; Ollama and OpenAI are optional runtime integrations.
- Use `openai>=3.8,<4` and `httpx>=0.28,<1`; read OpenAI prose through `response.output_text`, never by assuming `output[0]` is a message.
- Do not store API keys, authorization headers, hidden reasoning, or model chain-of-thought in SQLite or logs.
- One project may own at most one active generation workflow and one active candidate batch.
- A plan must be author-approved before any chapter call; candidate chapters must be author-approved through the existing `BatchService` before becoming official.
- Generate chapter ordinals sequentially; never dispatch multiple prose writers for one timeline.
- Enforce 1–5 chapters per batch and 4,500–6,000 visible characters per chapter.
- Never use the entire novel or entire five-chapter batch as one model input.
- Initial hard budgets are planner 16K/4K, writer 32K/12K, summarizer 16K/4K, reviewer 32K/6K input/output tokens.
- Required context is never silently trimmed; required-content overflow pauses before the provider call.
- Retrieval is structured filtering plus SQLite FTS5. Do not add embeddings or a vector database.
- Provider calls occur outside long-lived database write transactions.
- Preserve the existing `BatchService` approval transaction, published-text freeze, `/health` body, `/ready` semantics, and Phase 1 tests.
- Add one new Alembic revision after `0001_foundation`; production startup must never call `create_all`.

## File Map

- `src/ainovel/providers/contracts.py`: provider request, response, capabilities, diagnostic, and error types.
- `src/ainovel/providers/fake.py`: deterministic scripted provider.
- `src/ainovel/providers/demo.py`: request-driven deterministic Fake demo provider.
- `src/ainovel/providers/ollama.py`: Ollama HTTP adapter and diagnostic.
- `src/ainovel/providers/openai.py`: Responses API adapter and diagnostic.
- `src/ainovel/providers/registry.py`: configured provider lookup without global mutable singletons.
- `src/ainovel/agents/contracts.py`: batch plan, chapter draft, summary/delta, and review Pydantic schemas.
- `src/ainovel/agents/runner.py`: budgeted provider invocation and typed output validation.
- `src/ainovel/agents/prompts.py`: built-in immutable prompt bodies and role identifiers.
- `src/ainovel/models/prompt.py`: prompt versions and workflow prompt snapshots.
- `src/ainovel/models/context.py`: context sources, packets, and packet items.
- `src/ainovel/models/workflow.py`: workflows, steps, attempts, artifacts, and plan decisions.
- `src/ainovel/models/batch.py`: nullable unique workflow provenance on generated batches.
- `src/ainovel/services/prompts.py`: prompt version allocation, activation, and snapshotting.
- `src/ainovel/services/context.py`: source indexing, FTS lookup, provenance, and packet persistence.
- `src/ainovel/context/budget.py`: deterministic token estimation and required-first packing.
- `src/ainovel/services/workflows.py`: project ownership CAS, state transitions, claims, and reconciliation.
- `src/ainovel/workflows/orchestrator.py`: plan, author gate, sequential chapter, review, and candidate-batch flow.
- `src/ainovel/web/workflow_routes.py`: workflow and provider diagnostic pages/actions.
- `src/ainovel/web/templates/workflow.html`: minimal run detail and approval controls.
- `alembic/versions/0002_orchestration_context.py`: Stage 2 schema and FTS5 table.

## Shared Test Fixture Contracts

- Extend `tests/conftest.py` only in the first task that needs each fixture.
- `ready_project` creates a project with an author-approved constitution and a current official outline; it returns the refreshed `NovelProject`.
- `session_factory` returns `client.app.state.session_factory` so concurrency tests open independent sessions.
- `workflow` creates a persisted `GenerationWorkflow` for `ready_project` without prompt snapshots or completed steps.
- Task-specific HTTP/provider helpers live in their named test file and never perform external network calls.

---

### Task 1: Typed Provider Contract, Agent Schemas, and Fake Provider

**Files:**
- Create: `src/ainovel/providers/__init__.py`
- Create: `src/ainovel/providers/contracts.py`
- Create: `src/ainovel/providers/fake.py`
- Create: `src/ainovel/providers/demo.py`
- Create: `src/ainovel/agents/__init__.py`
- Create: `src/ainovel/agents/contracts.py`
- Create: `src/ainovel/agents/runner.py`
- Test: `tests/test_provider_contracts.py`

**Interfaces:**
- `ProviderCapabilities(context_window: int, max_output_tokens: int, strict_structured_output: bool, token_counting: bool, local: bool, real_calls_allowed: bool)`.
- `ModelRequest(model: str, system_prompt: str, input_payload: dict[str, object], output_schema: dict[str, object], max_input_tokens: int, max_output_tokens: int, timeout_seconds: float, metadata: dict[str, str])`.
- `ModelResponse(structured: dict[str, object] | None, text: str | None, provider_response_id: str | None, input_tokens: int | None, output_tokens: int | None, latency_ms: int)`.
- `ProviderDiagnostic(available: bool, detail: str, models: tuple[str, ...])`.
- `ModelProvider` protocol: `capabilities(model)`, `generate(request)`, and `diagnose(model)`.
- Base `ProviderError` with typed subclasses `ProviderUnavailable`, `ProviderAuthenticationError`, `ProviderTimeout`, and `ProviderProtocolError`.
- Agent schemas: `ChapterPlan`, `BatchPlanDraft`, `ChapterDraft`, `ChapterSummaryDelta`, `BatchReview`.
- `AgentRunner.run(provider, request, result_type) -> BaseModel` validates `ModelResponse.structured` with `result_type.model_validate`.
- `DemoFakeProvider` synthesizes deterministic valid responses from `request.metadata["agent_role"]`; it is stateless across workflows and is the default browser demo provider.
- Test helper `chapter_request(ordinal: int) -> ModelRequest` builds a writer request with that ordinal, the `ChapterDraft` schema, and the writer 32K/12K limits.

- [ ] **Step 1: Write failing contract and runner tests**

```python
def test_runner_returns_a_validated_batch_plan() -> None:
    request = ModelRequest(model="scripted", system_prompt="只返回结构化计划", input_payload={"requested_chapters": 1}, output_schema=BatchPlanDraft.model_json_schema(), max_input_tokens=16000, max_output_tokens=4000, timeout_seconds=5.0, metadata={"schema_name": "batch_plan", "agent_role": "batch_planner"})
    scripted = ModelResponse(
        structured={"chapters": [{"ordinal": 1, "title": "入局", "goal": "主角接下委托", "ending_hook": "发现追踪者"}]},
        text=None,
        provider_response_id="fake-1",
        input_tokens=120,
        output_tokens=80,
        latency_ms=1,
    )
    provider = FakeProvider([scripted])
    result = AgentRunner().run(provider, request, BatchPlanDraft)
    assert result.chapters[0].ordinal == 1


def test_runner_rejects_invalid_structured_output() -> None:
    request = ModelRequest(model="scripted", system_prompt="只返回结构化计划", input_payload={"requested_chapters": 1}, output_schema=BatchPlanDraft.model_json_schema(), max_input_tokens=16000, max_output_tokens=4000, timeout_seconds=5.0, metadata={"schema_name": "batch_plan", "agent_role": "batch_planner"})
    provider = FakeProvider([ModelResponse(structured={"chapters": []}, text=None, provider_response_id=None, input_tokens=None, output_tokens=None, latency_ms=1)])
    with pytest.raises(ProviderProtocolError):
        AgentRunner().run(provider, request, BatchPlanDraft)


def test_demo_fake_can_generate_an_exact_length_chapter() -> None:
    response = DemoFakeProvider().generate(chapter_request(ordinal=3))
    draft = ChapterDraft.model_validate(response.structured)
    assert count_visible_characters(draft.body) == 4500
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_provider_contracts.py -v`

Expected: collection fails because `ainovel.providers` and `ainovel.agents` do not exist.

- [ ] **Step 3: Implement the immutable contracts and Pydantic schemas**

```python
@runtime_checkable
class ModelProvider(Protocol):
    def capabilities(self, model: str) -> ProviderCapabilities:
        raise NotImplementedError

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        raise NotImplementedError


class BatchPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chapters: list[ChapterPlan] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def ordinals_are_contiguous(self) -> "BatchPlanDraft":
        if [chapter.ordinal for chapter in self.chapters] != list(range(1, len(self.chapters) + 1)):
            raise ValueError("chapter ordinals must be contiguous from 1")
        return self
```

Implement `ChapterDraft` with nonblank `title` and `body`; `ChapterSummaryDelta` with nonblank `summary` and `state_delta: dict[str, object]`; and `BatchReview` with `passed: bool`, `issues: list[str]`, and `evidence_queries: list[str]`. `AgentRunner` must reject missing structured data and wrap Pydantic validation failures as `ProviderProtocolError` without copying hidden provider data into the message.

- [ ] **Step 4: Implement deterministic FakeProvider behavior**

```python
class FakeProvider:
    def __init__(self, script: Sequence[ModelResponse | Exception]) -> None:
        self._script = deque(script)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._script:
            raise ProviderProtocolError("fake provider script exhausted")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item
```

Return fixed local capabilities and a deterministic diagnostic; copy requests on receipt so later caller mutation cannot rewrite test history.

Implement `DemoFakeProvider` separately from the scripted test double. For `batch_planner` it derives 1–5 contiguous plans from `requested_chapters`; for `chapter_writer` it derives the requested ordinal and emits exactly 4,500 visible Chinese characters; for `chapter_summarizer` it returns a stable summary/state delta; for `batch_reviewer` it returns a passing review. Reject missing or unknown role metadata. It must never call another provider, share a consumable global script, or claim literary quality.

- [ ] **Step 5: Run focused and baseline tests**

Run: `.\.venv\Scripts\python -m pytest tests\test_provider_contracts.py tests\test_batches.py -v`

Expected: all selected tests pass with no new warnings.

- [ ] **Step 6: Commit**

```powershell
git add src/ainovel/providers src/ainovel/agents tests/test_provider_contracts.py
git commit -m "feat: add typed model provider contract"
```

---

### Task 2: Ollama and OpenAI Provider Adapters

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/ainovel/config.py`
- Create: `src/ainovel/providers/ollama.py`
- Create: `src/ainovel/providers/openai.py`
- Create: `src/ainovel/providers/registry.py`
- Test: `tests/test_provider_adapters.py`

**Interfaces:**
- Runtime dependencies: `httpx>=0.28,<1` and `openai>=3.8,<4`; remove duplicate httpx declaration from the test extra.
- Settings: `ollama_base_url="http://127.0.0.1:11434"`, `openai_api_key: SecretStr | None`, `openai_base_url: str | None`, `allow_real_openai=False`, `provider_timeout_seconds=120.0`.
- `OllamaProvider(client: httpx.Client, base_url: str)` maps `/api/chat` JSON Schema responses to `ModelResponse` and `/api/tags` to `ProviderDiagnostic`.
- `OpenAIProvider(client: OpenAI, allow_real_calls: bool)` calls `client.responses.create` only when enabled and reads `response.output_text` plus `response.usage`.
- `ProviderRegistry.get(name: Literal["fake", "ollama", "openai"]) -> ModelProvider`.
- `make_summary_request() -> ModelRequest` in `tests/test_provider_adapters.py` returns model `test-model`, schema name `chapter_summary`, a `ChapterSummaryDelta.model_json_schema()` schema, 16K/4K limits, and a five-second timeout.
- `ollama_provider_with_transport(payload: dict[str, object]) -> tuple[OllamaProvider, dict[str, object]]` uses `httpx.MockTransport`, records the decoded request body, and returns the supplied JSON response without opening a socket.
- `fake_openai_client` exposes `responses.create(**kwargs)`, records copied kwargs in `responses.calls`, and returns an object with `output_text`, `id`, and `usage.input_tokens/output_tokens` fields.

- [ ] **Step 1: Write failing adapter protocol tests using local transports**

```python
def test_ollama_posts_schema_and_parses_usage() -> None:
    request = make_summary_request()
    provider, captured = ollama_provider_with_transport(
        {"message": {"content": '{"summary":"有效"}'}, "prompt_eval_count": 21, "eval_count": 9}
    )
    response = provider.generate(request)
    assert captured["format"] == request.output_schema
    assert response.structured == {"summary": "有效"}
    assert response.input_tokens == 21


def test_openai_is_disabled_without_explicit_opt_in(fake_openai_client) -> None:
    provider = OpenAIProvider(fake_openai_client, allow_real_calls=False)
    with pytest.raises(ProviderAuthenticationError, match="disabled"):
        provider.generate(make_summary_request())
    assert fake_openai_client.responses.calls == []
```

Add cases for timeout classification, malformed JSON, Ollama model listing, missing OpenAI key, `output_text` parsing, usage mapping, and registry unknown-name rejection. Assert that error strings and captured logs do not contain test secrets.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_provider_adapters.py -v`

Expected: imports for Ollama, OpenAI, and registry implementations fail.

- [ ] **Step 3: Move httpx to runtime and add the OpenAI SDK**

```toml
dependencies = [
  "httpx>=0.28,<1",
  "openai>=3.8,<4",
]
```

Preserve all existing dependencies and test tools. Add the settings with `AINOVEL_` environment names inherited from `Settings.model_config`; keep secrets as `SecretStr` and never interpolate them into diagnostics.

- [ ] **Step 4: Implement OllamaProvider**

Send `model`, nonstreaming `messages`, `format` equal to the JSON Schema, and bounded options. Parse `message.content` as JSON; map connect errors to `ProviderUnavailable`, timeouts to `ProviderTimeout`, non-2xx or malformed responses to `ProviderProtocolError`. `diagnose` calls `/api/tags` and returns sorted model names.

- [ ] **Step 5: Implement OpenAIProvider and ProviderRegistry**

```python
response = self._client.responses.create(
    model=request.model,
    instructions=request.system_prompt,
    input=json.dumps(request.input_payload, ensure_ascii=False),
    text={"format": {"type": "json_schema", "name": request.metadata["schema_name"], "strict": True, "schema": request.output_schema}},
    max_output_tokens=request.max_output_tokens,
)
```

Parse `response.output_text` as JSON for structured tasks. Map SDK authentication, timeout, connection, and response errors to the typed provider errors. The registry receives factories through its constructor so tests and `create_app` can inject providers without module globals.

- [ ] **Step 6: Run focused and full tests**

Run: `.\.venv\Scripts\python -m pytest tests\test_provider_adapters.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: both commands pass without external network access.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src/ainovel/config.py src/ainovel/providers tests/test_provider_adapters.py
git commit -m "feat: add ollama and openai providers"
```

---

### Task 3: Stage 2 Persistence Schema and Migration

**Files:**
- Modify: `src/ainovel/models/batch.py`
- Modify: `src/ainovel/models/project.py`
- Modify: `src/ainovel/models/__init__.py`
- Modify: `src/ainovel/db.py`
- Create: `src/ainovel/models/prompt.py`
- Create: `src/ainovel/models/context.py`
- Create: `src/ainovel/models/workflow.py`
- Create: `alembic/versions/0002_orchestration_context.py`
- Modify: `tests/test_foundation_acceptance.py`
- Test: `tests/test_orchestration_schema.py`

**Interfaces:**
- Add nullable `NovelProject.active_workflow_id: str | None` without a cyclic database FK.
- Add nullable unique `WritingBatch.source_workflow_id: str | None` without a cyclic database FK; ordinary author-created Phase 1 batches keep `NULL`.
- `PromptVersion`: UUID id, role, version number, body, SHA-256 content hash, active boolean, source, timestamps; unique `(role, version_number)` and partial service invariant of one active version per role.
- Add unique partial index `uq_prompt_versions_one_active_role` on `PromptVersion.role WHERE active = 1`; service CAS and the database both enforce one active version per role.
- `WorkflowPromptSnapshot`: UUID id, workflow id FK, role, prompt version id FK, prompt body, output schema JSON, parameters JSON; unique `(workflow_id, role)`.
- `GenerationWorkflow`: UUID id, project id FK, base outline id FK, provider/model, requested chapter count, status, current position, candidate batch id, eight input/output budget values, actual input/output usage totals, revision, last error code/detail, timestamps.
- `WorkflowStep`: UUID id, workflow id FK, kind, ordinal nullable, position, status, attempt count, active artifact id nullable, `lease_owner` nullable, `lease_expires_at` nullable UTC timestamp, revision; unique `(workflow_id, position)` and `(workflow_id, kind, ordinal)`.
- `ModelAttempt`: UUID id, step id FK, attempt number, status, request digest, provider response id nullable, input/output tokens nullable, latency nullable, error code/detail nullable, timestamps; unique `(step_id, attempt_number)`.
- `WorkflowArtifact`: UUID id, workflow/step FK, kind, ordinal nullable, text content nullable, JSON payload, visible character count nullable, content hash, timestamps; no update service is permitted.
- `PlanDecision`: UUID id, workflow id FK, decision, reason, actor, timestamps.
- `ContextSource`: UUID id, project id, source type/id, source version, state scope, layer, text, content hash, timestamps; unique `(project_id, source_type, source_id, source_version, state_scope)`.
- `ContextPacket`: UUID id, workflow/step FK, max/used input tokens, fixed overhead tokens, reserved output, status, timestamps.
- `ContextPacketItem`: UUID id, packet/source FK nullable, stable source key, layer, text snapshot, selected boolean, required boolean, relevance integer, temporal distance integer, estimated tokens, excerpt start/end nullable, trim reason nullable, position, timestamps.
- FTS5 virtual table `context_source_fts(source_id UNINDEXED, project_id UNINDEXED, text)`.
- `migrated_engine` fixture builds a `tmp_path` database, sets Alembic `sqlalchemy.url`, upgrades to head, yields a SQLAlchemy Engine, disposes it, and removes the database plus `-wal`/`-shm` siblings.

- [ ] **Step 1: Write failing metadata and migration tests**

```python
def test_stage_two_tables_and_fts_exist(migrated_engine) -> None:
    names = set(inspect(migrated_engine).get_table_names())
    assert {"generation_workflows", "workflow_steps", "model_attempts", "workflow_artifacts", "prompt_versions", "workflow_prompt_snapshots", "plan_decisions", "context_sources", "context_packets", "context_packet_items"} <= names
    with migrated_engine.connect() as connection:
        fts = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='context_source_fts'")).scalar_one()
    assert fts == "context_source_fts"
```

Add assertions for every uniqueness constraint, server default, foreign key, the nullable active-workflow pointer, unique batch workflow provenance, lease columns, and downgrade removal. Extend the head-binding test from `0001_foundation` to `0002_orchestration_context`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_orchestration_schema.py tests\test_foundation_acceptance.py::test_readiness_revision_matches_the_migration_head -v`

Expected: Stage 2 tables/revision are absent and readiness still names `0001_foundation`.

- [ ] **Step 3: Implement focused ORM models and registration**

Use SQLAlchemy typed mappings, UUID strings, explicit nullable flags, indexes on project/status and workflow/status, and matching Python/server defaults. Import every model from `ainovel.models.__init__` so Alembic metadata is complete.

- [ ] **Step 4: Implement revision `0002_orchestration_context`**

Set `down_revision = "0001_foundation"`. Create relational tables before FTS, create the FTS5 virtual table with `op.execute`, and drop FTS before relational tables in downgrade. Change `ALEMBIC_HEAD_REVISION` to `0002_orchestration_context`.

- [ ] **Step 5: Run migration roundtrip and metadata check**

Run: `.\.venv\Scripts\python -m pytest tests\test_orchestration_schema.py tests\test_foundation_acceptance.py -v`

Expected: upgrade, `alembic.command.check`, downgrade to base, and re-upgrade all pass on `tmp_path` databases.

- [ ] **Step 6: Run the full suite and commit**

Run: `.\.venv\Scripts\python -m pytest -q`

```powershell
git add src/ainovel/models src/ainovel/db.py alembic/versions/0002_orchestration_context.py tests/test_orchestration_schema.py tests/test_foundation_acceptance.py
git commit -m "feat: add orchestration persistence schema"
```

---

### Task 4: Versioned Prompt Registry and Immutable Run Snapshots

**Files:**
- Create: `src/ainovel/agents/prompts.py`
- Create: `src/ainovel/services/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Roles are exact strings: `batch_planner`, `chapter_writer`, `chapter_summarizer`, `batch_reviewer`.
- `BUILTIN_PROMPTS: Mapping[str, str]` contains complete role instructions and forbids unrequested prose in structured tasks.
- `PromptService.ensure_builtins() -> list[PromptVersion]` is idempotent by role and content hash.
- `PromptService.create_version(role: str, body: str, source: str) -> PromptVersion` rejects blank bodies and allocates a monotonic role version under a uniqueness-retry bound.
- `PromptService.activate(prompt_version_id: str) -> PromptVersion` atomically deactivates the previously active version for that role.
- `PromptService.snapshot(workflow_id: str, schemas: Mapping[str, type[BaseModel]], parameters: Mapping[str, dict[str, object]]) -> list[WorkflowPromptSnapshot]` stores the active body, `model_json_schema()`, and copied parameters exactly once.
- `AGENT_SCHEMAS` maps the four roles to `BatchPlanDraft`, `ChapterDraft`, `ChapterSummaryDelta`, and `BatchReview` respectively.
- `AGENT_PARAMETERS` maps planner to `{max_input_tokens: 16000, max_output_tokens: 4000}`, writer to `{32000, 12000}`, summarizer to `{16000, 4000}`, and reviewer to `{32000, 6000}` using those exact key names.

- [ ] **Step 1: Write failing prompt lifecycle tests**

```python
def test_workflow_snapshot_does_not_change_when_prompt_is_replaced(session, workflow) -> None:
    service = PromptService(session)
    service.ensure_builtins()
    before = service.snapshot(workflow.id, AGENT_SCHEMAS, AGENT_PARAMETERS)
    replacement = service.create_version("chapter_writer", "新的完整主笔提示词", "author")
    service.activate(replacement.id)
    after = service.list_snapshots(workflow.id)
    assert [(row.prompt_body, row.output_schema) for row in after] == [(row.prompt_body, row.output_schema) for row in before]
```

Add tests for idempotent seeding, blank rejection, concurrent version allocation conflict, one active version per role, unknown roles, copied parameter dictionaries, and exactly four workflow snapshots.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_prompts.py -v`

Expected: prompt constants and `PromptService` imports fail.

- [ ] **Step 3: Add complete built-in prompts**

Each prompt must state its role, accepted structured task, forbidden actions, output Schema obligation, official/candidate separation, and no hidden reasoning requirement. The writer prompt must state 4,500–6,000 visible Chinese characters, approved-plan fidelity, sequential context only, and no future-chapter assumptions. The reviewer prompt must use chapter reports and summaries by default and request exact excerpts only through `evidence_queries`.

- [ ] **Step 4: Implement version allocation, activation, and snapshots**

Use `sha256(body.encode("utf-8")).hexdigest()`. Copy JSON through `deepcopy` before persistence. On activation, condition updates include the target role and current active id; stale sessions raise `ValueError("prompt activation conflict")` rather than create two active versions.

- [ ] **Step 5: Run focused and full tests**

Run: `.\.venv\Scripts\python -m pytest tests\test_prompts.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all tests pass and prompt history remains queryable.

- [ ] **Step 6: Commit**

```powershell
git add src/ainovel/agents/prompts.py src/ainovel/services/prompts.py tests/test_prompts.py
git commit -m "feat: add versioned prompt registry"
```

---

### Task 5: Structured Retrieval, FTS5, and Hard Context Budgets

**Files:**
- Create: `src/ainovel/context/__init__.py`
- Create: `src/ainovel/context/budget.py`
- Create: `src/ainovel/services/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- `ContextCandidate(stable_key: str, layer: int, text: str, required: bool, relevance: int, temporal_distance: int, source_id: str | None, source_type: str, source_version: str, state_scope: str, excerpt_start: int | None, excerpt_end: int | None)`.
- `TrimmedContext(item: ContextCandidate, reason: str)` and `RequiredContextOverflow(stable_key: str, required_tokens: int, capacity: int)`.
- `PackedContext(selected: tuple[ContextCandidate, ...], trimmed: tuple[TrimmedContext, ...], used_tokens: int, max_input_tokens: int, reserved_output_tokens: int)`.
- `TokenEstimator.estimate(text: str) -> int`; default `ConservativeEstimator` returns at least one token and uses `ceil(len(text.encode("utf-8")) / 3)` plus per-item framing overhead.
- `effective_input_capacity(configured_input_tokens, provider_context_window, reserved_output_tokens, safety_tokens=1024) -> int` returns `min(configured_input_tokens, provider_context_window - reserved_output_tokens - safety_tokens)` and rejects nonpositive results.
- `ContextBudgeter.pack(candidates, input_capacity_tokens, reserved_output_tokens, fixed_overhead_tokens=0) -> PackedContext` treats `input_capacity_tokens` as the already-computed input limit, starts usage at the system-prompt/JSON framing overhead, and does not subtract output a second time.
- `ContextBuilder.candidates_for_step(workflow_id: str, step_id: str) -> list[ContextCandidate]` maps the exact task to L0–L7 sources with project/version/state isolation.
- `SemanticRetriever` is a Protocol reserved for a future implementation; Stage 2 wiring does not instantiate or call one.
- `ContextIndexService.rebuild_official(project_id: str) -> int` idempotently indexes the approved constitution, official outline nodes, and official/published chapters.
- `ContextIndexService.index_workflow_artifact(artifact_id: str) -> ContextSource` indexes only validated candidate summaries/deltas or explicitly requested excerpts under their workflow state scope.
- `ContextIndexService.search(project_id: str, query: str, source_types: set[str], limit: int) -> list[ContextSource]` uses bound parameters and stable ordering.
- `ContextService.build_packet(workflow_id: str, step_id: str, required, optional, limits) -> ContextPacket` persists selected and trimmed item snapshots.
- Test helper `candidate(text, layer, required, relevance, temporal_distance=0) -> ContextCandidate` uses `text` as `stable_key`, source type `test`, source version `1`, state scope `official`, and null excerpt offsets.
- Test helper `FixedEstimator(costs: Mapping[str, int])` implements `estimate` by returning `costs[text]`; it never falls back to production estimation.

- [ ] **Step 1: Write failing budget and retrieval tests**

```python
def test_budget_never_trims_required_context() -> None:
    budgeter = ContextBudgeter(FixedEstimator({"constitution": 40, "old excerpt": 80}))
    packed = budgeter.pack(
        [candidate("constitution", layer=0, required=True, relevance=100), candidate("old excerpt", layer=7, required=False, relevance=1)],
        input_capacity_tokens=48,
        reserved_output_tokens=8,
    )
    assert [item.text for item in packed.selected] == ["constitution"]
    assert packed.trimmed[0].reason == "budget"


def test_required_overflow_fails_before_provider_call() -> None:
    with pytest.raises(RequiredContextOverflow):
        ContextBudgeter(FixedEstimator({"constitution": 65})).pack(
            [candidate("constitution", layer=0, required=True, relevance=100)], 64, 0
        )
```

Add tests for project/state/version isolation, FTS query escaping, deterministic order `(required desc, layer asc, relevance desc, temporal distance asc, stable_key asc)`, overlapping-source deduplication, per-source limits, selected/trimmed persistence, and idempotent official reindexing.

Add tests for the exact L0–L7 mapping, 20–30 event-chain and 3–5 recent-summary limits, candidate workflow isolation, excerpt offsets, temporal-distance tie-breaking, and a case where required context would fit alone but fixed system-prompt/JSON overhead causes pre-call overflow.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_context.py -v`

Expected: budget and context services do not exist.

- [ ] **Step 3: Implement deterministic required-first packing**

```python
ordered = sorted(candidates, key=lambda item: (not item.required, item.layer, -item.relevance, item.temporal_distance, item.stable_key))
capacity = input_capacity_tokens
used = fixed_overhead_tokens
for item in ordered:
    cost = estimator.estimate(item.text) + ITEM_FRAMING_TOKENS
    if item.required and used + cost > capacity:
        raise RequiredContextOverflow(item.stable_key, used + cost, capacity)
    if used + cost <= capacity:
        selected.append(item)
        used += cost
    else:
        trimmed.append(TrimmedContext(item=item, reason="budget"))
```

Set `ITEM_FRAMING_TOKENS = 4`. Reject nonpositive limits, negative overhead, impossible overhead, invalid excerpt ranges, and duplicate stable keys. Do not mutate caller candidates. Test `effective_input_capacity(32000, 40000, 12000, 1024) == 26976` and `effective_input_capacity(32000, 128000, 12000, 1024) == 32000`.

Define the task mapping explicitly: L0 constitution/core prohibitions/provisional ending; L1 relevant world rules, power system, and core characters; L2 current volume and plot stage; L3 the latest 20–30 structured event chains; L4 the latest 3–5 chapter/scene summaries; L5 relevant characters, locations, items, relationships, and foreshadowing; L6 approved current batch plan plus earlier candidate summaries/deltas; L7 small exact historical excerpts requested for fact verification. Constitution, provisional ending, author-locked facts, current stage goal, approved plan, and current critical character state are required. Optional trimming priority is low-relevance L7, older excerpts, older events, optional style samples, then other optional supplements.

- [ ] **Step 4: Implement source indexing and safe FTS lookup**

Rebuild only the project's official sources in one short transaction. Upsert relational `ContextSource` rows and mirror exact text into `context_source_fts`; delete obsolete official index rows for that project without touching candidate workflow sources. FTS queries use SQLAlchemy `text()` with bound values and return relational rows after verifying project ownership.

Index validated workflow artifacts after persistence with a stable source version equal to their content hash. Candidate searches must include the exact workflow state scope and can never fall back to another workflow's candidate rows.

- [ ] **Step 5: Persist complete context packet provenance**

Store one packet row and one item row for every selected or trimmed candidate. Item snapshots retain text, layer, stable source key, source version, temporal distance, excerpt offsets, estimated tokens, required flag, position, and trim reason; the packet retains fixed prompt/payload overhead so a later prompt change cannot rewrite historical context or its accounting.

- [ ] **Step 6: Run focused and full tests**

Run: `.\.venv\Scripts\python -m pytest tests\test_context.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all tests pass without model or network calls.

- [ ] **Step 7: Commit**

```powershell
git add src/ainovel/context src/ainovel/services/context.py tests/test_context.py
git commit -m "feat: add bounded context retrieval"
```

---

### Task 6: Persistent Workflow Repository and CAS State Machine

**Files:**
- Create: `src/ainovel/services/workflows.py`
- Test: `tests/test_workflows.py`

**Interfaces:**
- `WorkflowBudgets(planner_input=16000, planner_output=4000, writer_input=32000, writer_output=12000, summarizer_input=16000, summarizer_output=4000, reviewer_input=32000, reviewer_output=6000)` is frozen; `DEFAULT_BUDGETS` is one instance with those exact values.
- `PROVIDER_NAMES = frozenset({"fake", "ollama", "openai"})`; `WorkflowService.start` rejects every other provider name before ownership mutation.
- Status values: `PREPARING`, `PLANNING`, `AWAITING_PLAN_APPROVAL`, `GENERATING_CHAPTERS`, `REVIEWING_BATCH`, `CREATING_CANDIDATE_BATCH`, `AWAITING_CONTENT_APPROVAL`, `PAUSED_PROVIDER`, `PAUSED_CONTEXT_OVERFLOW`, `PAUSED_ATTEMPTS`, `PAUSED_REVIEW`, `PAUSED_STALE_VERSION`, `COMPLETED`, `REJECTED`, `CANCELLED`, `FAILED`.
- Step statuses: `PENDING`, `RUNNING`, `COMPLETED`, `PAUSED`, `FAILED`.
- `WorkflowService.start(project_id, provider_name, model_name, requested_chapters, budgets) -> GenerationWorkflow`.
- `WorkflowService.claim_step(workflow_id, expected_statuses, worker_id, lease_seconds=300) -> WorkflowStep | None` uses revision/status CAS and records a finite UTC lease.
- `WorkflowService.recover_expired_claims(workflow_id, now) -> int` returns expired `RUNNING` steps without an active completed artifact to `PENDING`; it never resets a completed step.
- `WorkflowService.record_attempt_start(step_id, request_digest) -> ModelAttempt` allocates a monotonic attempt number, maximum two.
- `WorkflowService.complete_attempt(attempt_id, response, artifact, finalize_step=True) -> WorkflowArtifact` commits the completed attempt, immutable artifact, and usage totals atomically. A finalized artifact becomes the step's active result; reviewer evidence-request artifacts use `finalize_step=False`, clear the lease, and return the step to `PENDING` without becoming its active result.
- `WorkflowService.fail_attempt(attempt_id, error) -> GenerationWorkflow` maps typed provider failures to retryable or paused states.
- `WorkflowService.approve_plan(workflow_id, actor)`, `reject_plan(workflow_id, reason, actor)`, `resume(workflow_id)`, and `reconcile_batch_decision(workflow_id)`.

- [ ] **Step 1: Write failing ownership and transition tests**

```python
def test_two_sessions_start_only_one_project_workflow(session_factory, ready_project) -> None:
    left = session_factory()
    right = session_factory()
    first = WorkflowService(left).start(ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS)
    with pytest.raises(ValueError, match="active workflow"):
        WorkflowService(right).start(ready_project.id, "fake", "scripted", 5, DEFAULT_BUDGETS)
    assert first.id == left.get(NovelProject, ready_project.id).active_workflow_id


def test_only_expired_unfinished_claim_is_recovered(session, workflow, clock) -> None:
    claimed = WorkflowService(session).claim_step(workflow.id, {"PLANNING"}, "worker-a", lease_seconds=30)
    assert claimed is not None
    assert WorkflowService(session).recover_expired_claims(workflow.id, clock.now()) == 0
    clock.advance(seconds=31)
    assert WorkflowService(session).recover_expired_claims(workflow.id, clock.now()) == 1
```

Add a barrier-based two-session winner test, stale outline start rejection, active batch rejection, invalid chapter count, stale step claim, live-lease exclusion, expired-lease recovery, completed-step non-recovery, maximum-attempt pause, no long transaction across provider calls, plan approval/rejection audit, owner release only for matching workflow, and resume-state whitelist.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_workflows.py -v`

Expected: `WorkflowService` does not exist.

- [ ] **Step 3: Implement start CAS and deterministic initial steps**

Validate an approved constitution, current official outline, no active batch, no active workflow, 1–5 requested chapters, known Provider, and positive budgets. Ensure built-in prompts exist before the transaction. Preallocate the workflow id, update `NovelProject.active_workflow_id` where it is null and official pointers still match, insert the workflow, exactly four prompt snapshots, the `PLANNING` step, and audit in one commit. `PromptService.snapshot` flushes into the caller's transaction and never commits independently.

- [ ] **Step 4: Implement step claims, attempts, artifacts, and usage accounting**

Expire stale ORM state before decisions. Claims update exact workflow/step revision and expected state, set worker/lease fields, and commit before returning. Recovery compares an injected UTC clock against `lease_expires_at`. Provider execution is performed by callers after the claim transaction commits. Completion verifies the same running step, lease owner, and attempt before inserting an immutable artifact, clearing the lease, and incrementing usage totals.

- [ ] **Step 5: Implement author decisions, pause/resume, and batch reconciliation**

Plan approval requires `AWAITING_PLAN_APPROVAL` and one active plan artifact, then creates writer/summarizer steps in strict ordinal order followed by reviewer and candidate-batch steps. Rejection requires a nonblank reason and releases only matching ownership. Reconciliation treats approved/rejected `WritingBatch.status` as authoritative and is idempotent after a crash between the batch transaction and workflow update.

- [ ] **Step 6: Run focused concurrency tests and full suite**

Run: `.\.venv\Scripts\python -m pytest tests\test_workflows.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all tests pass; exactly one workflow/audit exists after start races.

- [ ] **Step 7: Commit**

```powershell
git add src/ainovel/services/workflows.py tests/test_workflows.py
git commit -m "feat: add recoverable workflow state machine"
```

---

### Task 7: End-to-End Orchestrator and Sequential Chapter Writer

**Files:**
- Modify: `src/ainovel/models/batch.py`
- Modify: `src/ainovel/services/batches.py`
- Create: `src/ainovel/workflows/__init__.py`
- Create: `src/ainovel/workflows/orchestrator.py`
- Modify: `tests/test_batches.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- `AdvanceResult(workflow_id: str, status: str, completed_step: str | None, waiting_for: str | None, candidate_batch_id: str | None)`.
- `WorkflowOrchestrator.advance(workflow_id: str) -> AdvanceResult` executes at most one persisted workflow step.
- `WorkflowOrchestrator.run_until_blocked(workflow_id: str, max_steps: int = 32) -> AdvanceResult` repeatedly calls `advance` until an author gate, pause, failure, or terminal state.
- The orchestrator receives `session_factory`, `ProviderRegistry`, `AgentRunner`, `ContextService`, and `PromptService` through its constructor.
- The orchestrator owns a stable per-process `worker_id`; each `advance` first recovers expired claims, then claims with that worker id, and completion verifies the same lease owner.
- Planner input contains project constitution, official outline id/tree, requested chapter count, official chapter statistics, and budgets.
- Writer input contains exactly one approved `ChapterPlan`, selected context packet, previous candidate summary/delta, and current ordinal.
- Summarizer input contains only the generated chapter and approved plan for that ordinal.
- Reviewer input contains the approved batch plan plus per-chapter summaries, deltas, visible counts, and validation results; it does not contain all chapter bodies.
- Each request computes fixed overhead from the snapshotted system prompt plus canonical JSON task framing before packing candidates; the final canonical serialized request is estimated again and must remain within the effective input capacity.
- `EXECUTABLE_WORKFLOW_STATUSES = frozenset({"PLANNING", "GENERATING_CHAPTERS", "REVIEWING_BATCH", "CREATING_CANDIDATE_BATCH"})`.
- `digest_request(request: ModelRequest) -> str` canonicalizes request fields with sorted-key UTF-8 JSON and returns SHA-256.
- Private orchestrator helpers used below have exact signatures: `_workflow_service() -> WorkflowService`, `_build_request(step: WorkflowStep) -> ModelRequest`, `_provider(step: WorkflowStep) -> ModelProvider`, `_result_type(step: WorkflowStep) -> type[BaseModel]`, `_record_failure(attempt_id: str, error: ProviderError) -> AdvanceResult`, `_save_result_and_advance(attempt_id: str, step: WorkflowStep, result: BaseModel) -> AdvanceResult`, and `_current_result(workflow_id: str) -> AdvanceResult`.
- `scripted_five_chapter_provider` is a Fake Provider scripted with one valid five-item plan, writer/summarizer pairs for ordinals 1–5, and one passing review; it records all requests.
- `orchestrator` uses the test `session_factory`, a registry containing `scripted_five_chapter_provider`, real prompt/context/workflow services, and `AgentRunner`.
- `approved_plan_workflow` starts a five-chapter workflow, advances through planning, approves the saved plan through `WorkflowService`, and returns the refreshed workflow row. `provider` is an alias of `scripted_five_chapter_provider` in this test module.

- [ ] **Step 1: Write failing author-gate and five-chapter tests**

```python
def test_plan_gate_blocks_every_chapter_call(orchestrator, scripted_five_chapter_provider, workflow) -> None:
    result = orchestrator.run_until_blocked(workflow.id)
    assert result.status == "AWAITING_PLAN_APPROVAL"
    assert [request.metadata["agent_role"] for request in scripted_five_chapter_provider.requests] == ["batch_planner"]


def test_five_chapters_are_generated_and_copied_to_one_candidate_batch(orchestrator, approved_plan_workflow, provider, session) -> None:
    result = orchestrator.run_until_blocked(approved_plan_workflow.id)
    assert result.status == "AWAITING_CONTENT_APPROVAL"
    assert result.candidate_batch_id is not None
    chapters = BatchService(session).list_chapters(result.candidate_batch_id)
    assert [chapter.ordinal for chapter in chapters] == [1, 2, 3, 4, 5]
    assert all(4500 <= chapter.visible_char_count <= 6000 for chapter in chapters)
    assert BatchService(session).official_chapter_statistics(approved_plan_workflow.project_id).chapter_count == 0
```

Add tests proving each writer request sees only earlier candidate summaries, reviewer input excludes raw five-chapter bodies, invalid length pauses before candidate creation, failed review preserves evidence queries, provider attempt exhaustion pauses, stale outline pauses, and candidate batch creation occurs exactly once after crash/retry.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_orchestrator.py -v`

Expected: orchestrator imports fail.

- [ ] **Step 3: Implement one-step dispatch without holding a write transaction**

```python
def advance(self, workflow_id: str) -> AdvanceResult:
    self._workflow_service().recover_expired_claims(workflow_id, self._clock.now())
    claim = self._workflow_service().claim_step(workflow_id, EXECUTABLE_WORKFLOW_STATUSES, self._worker_id)
    if claim is None:
        return self._current_result(workflow_id)
    request = self._build_request(claim)
    attempt = self._workflow_service().record_attempt_start(claim.id, digest_request(request))
    try:
        response_model = self._runner.run(self._provider(claim), request, self._result_type(claim))
    except ProviderError as error:
        return self._record_failure(attempt.id, error)
    return self._save_result_and_advance(attempt.id, claim, response_model)
```

Every helper opens a short session through `session_factory`; no Session remains in a transaction while `AgentRunner.run` calls a Provider.

- [ ] **Step 4: Implement planner and author gate**

The planner step builds and persists its context packet, calls `batch_planner`, saves one `BatchPlanDraft`, and moves to `AWAITING_PLAN_APPROVAL`. `run_until_blocked` must return immediately at that status. Plan approval uses `WorkflowService.approve_plan`; no writer step exists before that transaction succeeds.

- [ ] **Step 5: Implement strict writer/summarizer ordering and base review**

Writer N requires completed summary N-1 when N > 1. Validate visible characters with `count_visible_characters`; also persist deterministic checks for nonblank title/body, approved chapter goal/key-event coverage, obvious repeated blocks, and ordinal continuity. Invalid output consumes an attempt and retries once, then pauses. Summarizer produces the only candidate summary/delta used by later chapters and its validated artifact is indexed under the workflow scope.

Reviewer receives no raw body list. A first response containing `evidence_queries` remains an immutable non-final attempt artifact; resolve only those queries to bounded, offset-bearing snippets and use the second and final allowed attempt. A second response that still requests evidence, or a failed final review, moves to `PAUSED_REVIEW`. Only a passing final review completes the reviewer step.

- [ ] **Step 6: Implement idempotent candidate-batch creation and reconciliation**

Extend `BatchService.create` with optional keyword-only `source_workflow_id: str | None = None`, preserving all existing callers. Before creating, query `WritingBatch.source_workflow_id`; after any ownership/conflict error, query it again. Create one batch with the workflow id, copy each validated writer artifact with `save_candidate_chapter`, and call `mark_ready`. Persist `candidate_batch_id` with CAS. A unique constraint makes the crash window and concurrent retry idempotent. `reconcile_batch_decision` completes or rejects the workflow after reading the authoritative batch status and releases only matching `active_workflow_id`.

- [ ] **Step 7: Run focused recovery tests and full suite**

Run: `.\.venv\Scripts\python -m pytest tests\test_orchestrator.py tests\test_workflows.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all tests pass; no test accesses an external host.

- [ ] **Step 8: Commit**

```powershell
git add src/ainovel/workflows src/ainovel/models/batch.py src/ainovel/services/batches.py tests/test_orchestrator.py tests/test_batches.py
git commit -m "feat: orchestrate sequential chapter generation"
```

---

### Task 8: Minimal Web Workflow and Provider Diagnostics

**Files:**
- Modify: `src/ainovel/app.py`
- Modify: `src/ainovel/web/routes.py`
- Create: `src/ainovel/web/workflow_routes.py`
- Modify: `src/ainovel/web/templates/project.html`
- Create: `src/ainovel/web/templates/workflow.html`
- Modify: `tests/conftest.py`
- Test: `tests/test_workflow_web.py`
- Modify: `tests/test_web.py`

**Interfaces:**
- `create_app(database_url: str | None = None, provider_registry: ProviderRegistry | None = None) -> FastAPI` stores the registry and orchestrator factory on `app.state`.
- `GET /workflows/{workflow_id}` shows status, budgets, usage, prompt snapshots, steps, attempts, plan artifact, review issues, and linked candidate batch.
- `POST /projects/{project_id}/workflows` accepts `provider_name`, `model_name`, and `requested_chapters`.
- `POST /workflows/{workflow_id}/run` calls `run_until_blocked`.
- `POST /workflows/{workflow_id}/plan/approve` and `/plan/reject` enforce the first author gate.
- `POST /workflows/{workflow_id}/resume` resumes only a resumable pause.
- `POST /projects/{project_id}/providers/diagnose` checks one configured Provider and does not download models.
- All POST routes depend on `require_csrf`; ownership is resolved from database rows, never trusted form project ids.
- `csrf(client, path) -> str` in `tests/test_workflow_web.py` performs a GET, extracts the hidden `csrf_token`, and returns its HTML-unescaped value.
- `fake_registry` fixture contains one scripted Fake Provider with a valid one-chapter planning response and is passed to `create_app` by the test client fixture.
- Add a default `provider_registry` fixture returning `None`; change the shared `client(database_url, provider_registry)` fixture to pass it to `create_app`. A test module can override `provider_registry` without changing existing Phase 1 tests.

- [ ] **Step 1: Write failing route, CSRF, and rendering tests**

```python
def test_start_workflow_redirects_to_plan_gate(client, ready_project, fake_registry) -> None:
    response = client.post(
        f"/projects/{ready_project.id}/workflows",
        data={"provider_name": "fake", "model_name": "scripted", "requested_chapters": "5", "csrf_token": csrf(client, ready_project.id)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/workflows/")


def test_every_workflow_mutation_rejects_missing_csrf(client, workflow) -> None:
    paths = [
        f"/workflows/{workflow.id}/run",
        f"/workflows/{workflow.id}/plan/approve",
        f"/workflows/{workflow.id}/plan/reject",
        f"/workflows/{workflow.id}/resume",
        f"/projects/{workflow.project_id}/providers/diagnose",
    ]
    for path in paths:
        assert client.post(path, data={}).status_code == 403
```

Add tests for invalid provider/model/chapter count, plan details and budget display, approve/reject confirmation only on author decisions, run-to-next-gate, pause reason display, resume whitelist, diagnostic success/failure without secrets, all forms containing the session token, untrusted Host rejection, and candidate batch link.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_workflow_web.py -v`

Expected: workflow routes and template do not exist.

- [ ] **Step 3: Wire ProviderRegistry and orchestrator factory into create_app**

Default registry includes a fresh `DemoFakeProvider` per workflow, an Ollama provider using configured loopback URL, and an OpenAI provider that remains disabled unless settings explicitly allow it and an API key exists. Tests inject a registry; app creation must not contact any Provider. The browser Fake path must complete any requested 1–5 chapter demo without consuming shared global script state.

- [ ] **Step 4: Implement workflow routes with service-owned validation**

Routes parse forms, invoke services, translate domain validation to 422 pages, and redirect after successful POST. Never place business rules only in routes. Provider diagnostics return a rendered result and must redact settings secrets and exception internals.

- [ ] **Step 5: Implement minimal templates**

Project page shows the active workflow or a start form only when no batch/workflow is active. Workflow page shows one primary action matching current state, collapsed technical details, all serious pause reasons expanded, usage totals, and links to the project/candidate batch. Approve and reject require browser confirmation; ordinary run/resume actions do not.

- [ ] **Step 6: Run focused and full Web tests**

Run: `.\.venv\Scripts\python -m pytest tests\test_workflow_web.py tests\test_web.py -v`

Run: `.\.venv\Scripts\python -m pytest -q`

Expected: all tests pass and existing Phase 1 dashboard behavior remains unchanged.

- [ ] **Step 7: Commit**

```powershell
git add src/ainovel/app.py src/ainovel/web src/ainovel/web/templates tests/conftest.py tests/test_workflow_web.py tests/test_web.py
git commit -m "feat: add local generation workflow UI"
```

---

### Task 9: Offline Acceptance, Optional Ollama Smoke Test, Packaging, and Documentation

**Files:**
- Create: `tests/test_stage_two_acceptance.py`
- Create: `tests/test_ollama_live.py`
- Modify: `src/ainovel/web/workflow_routes.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_packaging.py`

**Interfaces:**
- Marker `local_model` is registered in pytest configuration and excluded only by explicit skip logic inside the live test.
- Live Ollama test requires both `AINOVEL_RUN_OLLAMA_TESTS=1` and `AINOVEL_OLLAMA_MODEL`; otherwise it calls `pytest.skip` before constructing a network client.
- The default suite monkeypatches provider construction so no external DNS or socket is needed.
- README commands cover migration, Fake demo, Ollama diagnosis, optional live test, and OpenAI opt-in safety without displaying credentials.
- `POST /workflows/{workflow_id}/reconcile` invokes `WorkflowService.reconcile_batch_decision`, is CSRF-protected, and redirects to the workflow page; this is the Task 9 behavior that makes the acceptance test RED before implementation.
- Acceptance helpers in `tests/test_stage_two_acceptance.py`: `start_fake_workflow(client, project_id, count) -> str` POSTs the start form and parses the workflow id from `Location`; `run_workflow(client, workflow_id)`, `approve_plan(client, workflow_id)`, and `reconcile_workflow(client, workflow_id)` POST their named routes with fresh CSRF tokens; `approve_candidate_batch(client, batch_id)` POSTs the existing batch approval route; `workflow_status(session, workflow_id)` expires state and returns `GenerationWorkflow.status`; `load_workflow(session, workflow_id)` expires state and returns the workflow row.
- `five_chapter_registry` scripts one five-item `BatchPlanDraft`, then writer/summarizer pairs for ordinals 1–5, then one passing `BatchReview`. Writer bodies are `"甲" * (4499 + ordinal)`, producing exact visible counts 4500 through 4504.
- The acceptance module defines `provider_registry(five_chapter_registry)` and returns that registry, overriding the shared fixture before `client` creates the app.
- Live-test helpers in `tests/test_ollama_live.py`: `ollama_provider_from_settings()` constructs an `httpx.Client` limited to `Settings.ollama_base_url`; `live_summary_request(model)` returns a 16K/4K `ModelRequest` using `ChapterSummaryDelta.model_json_schema()` and schema name `chapter_summary`.

- [ ] **Step 1: Write the failing Stage 2 acceptance test**

```python
def test_author_can_plan_generate_and_approve_five_chapters_offline(client, session, ready_project, five_chapter_registry) -> None:
    workflow_id = start_fake_workflow(client, ready_project.id, 5)
    run_workflow(client, workflow_id)
    assert workflow_status(session, workflow_id) == "AWAITING_PLAN_APPROVAL"
    approve_plan(client, workflow_id)
    run_workflow(client, workflow_id)
    workflow = load_workflow(session, workflow_id)
    assert workflow.status == "AWAITING_CONTENT_APPROVAL"
    chapters = BatchService(session).list_chapters(workflow.candidate_batch_id)
    assert [chapter.visible_char_count for chapter in chapters] == [4500, 4501, 4502, 4503, 4504]
    assert BatchService(session).official_chapter_statistics(ready_project.id).chapter_count == 0
    approve_candidate_batch(client, workflow.candidate_batch_id)
    reconcile_workflow(client, workflow_id)
    assert workflow_status(session, workflow_id) == "COMPLETED"
    assert BatchService(session).official_chapter_statistics(ready_project.id).chapter_count == 5
```

Add acceptance cases for crash after every model-step class, crash after batch approval before reconciliation, project ownership races, required-context overflow before any Provider request, prompt replacement during a paused workflow, reviewer evidence retrieval, and default app construction with neither Ollama nor OpenAI available.

Add `test_reconcile_rejects_missing_csrf`, asserting a POST to `/workflows/{workflow_id}/reconcile` without the session token returns 403 after the route exists.

- [ ] **Step 2: Run acceptance tests and verify RED**

Run: `.\.venv\Scripts\python -m pytest tests\test_stage_two_acceptance.py -v`

Expected: the final reconciliation POST returns 404 because `/workflows/{workflow_id}/reconcile` is not implemented yet.

- [ ] **Step 3: Close integration seams with tests first**

Add the CSRF-protected reconciliation route and its workflow-page action, then keep each failing acceptance assertion and make the smallest change in the owning service. Rerun the single failing test before the full acceptance file. Do not add compatibility branches that bypass prompt snapshots, context packets, workflow CAS, or `BatchService`.

- [ ] **Step 4: Add the opt-in Ollama live test**

```python
@pytest.mark.local_model
def test_configured_ollama_model_returns_structured_output() -> None:
    if os.getenv("AINOVEL_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set AINOVEL_RUN_OLLAMA_TESTS=1 to run local model tests")
    model = os.getenv("AINOVEL_OLLAMA_MODEL")
    if not model:
        pytest.skip("set AINOVEL_OLLAMA_MODEL to an installed model")
    diagnostic = ollama_provider_from_settings().diagnose(model)
    assert diagnostic.available
    assert model in diagnostic.models
    result = AgentRunner().run(ollama_provider_from_settings(), live_summary_request(model), ChapterSummaryDelta)
    assert result.summary.strip()
```

The test targets only `127.0.0.1`; it must not pull a model or call a cloud endpoint.

- [ ] **Step 5: Update packaging and operator documentation**

Add `web/templates/workflow.html` to the existing wheel asset assertion. Document exact PowerShell commands for `alembic upgrade head`, default tests, optional Ollama test, loopback server start, and OpenAI environment opt-in. State that Fake output tests workflow correctness, not literary quality.

- [ ] **Step 6: Run fresh complete verification**

Run: `.\.venv\Scripts\python -m pytest -q --cov=ainovel --cov-report=term-missing`

Run: `.\.venv\Scripts\python -m pytest tests\test_orchestration_schema.py::test_stage_two_migration_round_trip -v`

Run: `.\.venv\Scripts\python -m pytest tests\test_packaging.py -v`

Run: `.\.venv\Scripts\python -m compileall -q src tests`

Run: `git diff --check`

Expected: zero failures, no external network calls in the default suite, migration roundtrip/check passes, wheel contains all templates/static assets, compileall exits zero, and only explicitly accepted upstream warnings remain.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_stage_two_acceptance.py tests/test_ollama_live.py pyproject.toml README.md tests/test_packaging.py src/ainovel/web/workflow_routes.py src/ainovel/web/templates/workflow.html
git commit -m "test: verify offline orchestration workflow"
```

## Plan Completion Gate

### Spec Coverage Matrix

| Approved requirement | Implemented and verified in |
| --- | --- |
| Typed Fake/Ollama/OpenAI providers, offline defaults, secret redaction | Tasks 1, 2, 8, 9 |
| Immutable prompt versions and per-workflow snapshots | Tasks 3, 4, 6, 9 |
| L0–L7 structured/FTS retrieval, provenance, hard budgets, no whole-book prompt | Tasks 3, 5, 7, 9 |
| Project CAS, finite leases, attempts, immutable artifacts, crash recovery | Tasks 3, 6, 7, 9 |
| Plan approval before prose and sequential 1–5 chapter generation | Tasks 6, 7, 8, 9 |
| 4,500–6,000 visible characters and summary-only forward context | Tasks 1, 7, 9 |
| Review evidence retrieval without five raw chapters in one prompt | Tasks 5, 7, 9 |
| Idempotent candidate creation, Phase 1 approval authority, reconciliation | Tasks 3, 6, 7, 9 |
| Loopback Web UI, CSRF on every mutation, diagnostics, usage visibility | Tasks 8, 9 |
| Migration roundtrip, packaging, optional local Ollama smoke test | Tasks 2, 3, 9 |

After Task 9, generate one review package for the full branch from its merge base. Dispatch the most capable available reviewer to assess the complete Stage 2 spec, every deferred ledger item, concurrency behavior, absence of default network calls, secret handling, migration integrity, context hard caps, sequential chapter generation, author gates, and Phase 1 regression safety.

If the final review has findings, use exactly one combined final-fix wave and one scoped re-review. Do not start a second final-fix wave. Before presenting integration options, run the complete verification commands from Task 9 again and report every ledger `Ruling:` with its cost if wrong.

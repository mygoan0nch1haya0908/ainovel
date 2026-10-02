# Stage planning and bounded draft repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover short chapter drafts with bounded expansion and let authors drive rolling chapter generation from a stage architecture.

**Architecture:** Add versioned generation behavior without changing frozen legacy workflows. Separate work drafts from accepted chapter artifacts; stage roadmap services own progress while the existing workflow owns model attempts, leases and batch approval.

**Tech Stack:** Python, Pydantic, SQLAlchemy/SQLite, Alembic, FastAPI/Jinja, pytest, existing provider abstractions.

**Spec:** docs/superpowers/specs/2026-09-14-stage-planning-and-draft-repair-design.md (author approved for execution).

## Global Constraints

- 单章仍须达到 4500–6000 个可见字符，规划目标约 5200；不降低验收门槛，不以重复或无关填充凑字。
- 每批最多五章，批次计划和批次正文分别确认。正文确认前不能推进下一批；确认不等于发布。
- 每章首次生成后最多两轮修补；正常协议重试维持现有上限，两者单独计数，且都受工作流总调用和 Token 预算约束。
- 重启或作者点击继续不重置修补次数。
- 现有暂停任务、提示词快照和历史大纲不静默升级，不自动重试，也不自动把旧第一章提纲转换为阶段架构。
- All verification is offline. No paid calls, runtime DB edits, service restart, push, merge, or credential changes.
- Use the existing D:/ainovel/.worktrees/qwen-adapter worktree. Use apply_patch for edits; do not delete user artifacts.

## Verification environment

PowerShell, workdir D:/ainovel/.worktrees/qwen-adapter:

```powershell
$env:PYTHONPATH='D:/ainovel/.worktrees/qwen-adapter/src'
Remove-Item Env:AINOVEL_RUN_OLLAMA_TESTS,Env:AINOVEL_ALLOW_REAL_QWEN,Env:AINOVEL_QWEN_API_KEY,Env:DASHSCOPE_API_KEY,Env:AINOVEL_ALLOW_REAL_OPENAI,Env:AINOVEL_OPENAI_API_KEY,Env:AINOVEL_DATABASE_URL -ErrorAction SilentlyContinue
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short
```

Use focused test files during RED/GREEN, full suite before task commit. Do not run pytest concurrently. Use short explicit parameter IDs, never long generated bodies in test IDs. Existing baseline warning: Starlette BlockingPortal deprecation; optional live Ollama test skips.

### Task 1: Durable bounded draft repair and evidenced chapter validation

**Files:**
- Modify: src/ainovel/agents/contracts.py, prompts.py, runner.py
- Modify: src/ainovel/services/workflows.py, src/ainovel/workflows/orchestrator.py
- Modify: src/ainovel/models/workflow.py, src/ainovel/models/__init__.py
- Create: src/ainovel/services/draft_repair.py
- Create: alembic/versions/0003_draft_repair.py
- Test: tests/test_draft_repair.py; extend tests/test_orchestrator.py and migration tests as required

**Interfaces:**
- Consumes existing WorkflowService.start/claim_step/record_attempt_start/complete_attempt/fail_attempt, ModelResponse and frozen prompt snapshots.
- Produces opt-in WorkflowService.start(..., generation_version=2), default legacy version 1. Migration existing rows to 1. New UI explicitly opts into 2 in Task 3.
- Produces WorkChapterDraft(title: str, body: str), strict fields/nonblank but not final length; ChapterDraft stays strict. Version 2 ChapterPlan includes ordered scenes with positive target_characters totaling 4500–6000.
- Produces persisted isolated repair state keyed by writing step; latest draft payload plus revision/attempt association and durable repair count. Expose read-only draft state to Task 3; no raw body in diagnostics.
- Produces version 2 coverage validation with goal/hook verdicts and exact source excerpts, without literal goal/hook inclusion requirement. Task 2 inherits version 2 behavior.

- [ ] **RED: Add real service/orchestrator tests.** Reuse fixtures and FakeProvider helpers from test_orchestrator; script stage-free v2 planning, short body, second short body, full body and evidence response. Assert persisted state, bounded calls, requests containing latest draft and visible gap, no premature summary/body. Example assertions for independently constructed bodies:

```python
assert work_draft.visible_count == 2799
assert next_payload['repair']['visible_count'] == 2799
assert next_payload['repair']['minimum_gap'] == 1701
assert next_payload['repair']['draft']['body'] == first_body
assert not session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.kind == 'chapter_draft')).all()
```

Test 2799 → 3386 → 5200 succeeds; three shorts pause after two repairs; recreated orchestrator cannot reset count; budget exhaustion pauses before provider call; stale lease/version cannot persist a draft; malformed/extra-field payload cannot be repaired. Verify token usage is saved for short drafts. Verify excerpt-free/forged/negative coverage fails while valid evidence permits a body not quoting plan labels.

- [ ] **Run focused tests, record expected missing-feature failures.**
- [ ] **GREEN: Implement v2 path.** Store generation version with legacy default, add versioned prompts/schemas to snapshots without altering existing workflows. The shape of a repair request is:

```python
repair = {
    'draft': {'title': title, 'body': body},
    'visible_count': visible_count,
    'minimum_gap': 4500 - visible_count,
    'target_characters': 5200,
    'instruction': '在已批准场景内扩写，返回完整修订章；不得提前消耗后续事件或重复填充。',
}
```

Keep the ordinary writer schema strict for legacy paths; parse v2 work draft then validate final body separately. Save short draft and response usage under the existing lease/revision fence in one transaction. Do not advance WRITING until accepted. Count repairs durably at dispatch, so a crash cannot grant another free repair; protocol retries and semantic repair count remain independently bounded. All new calls must participate in workflow accounting and existing input/output capacity checks; add explicit finite v2 total call/token ceilings if existing budgets only describe per-request limits. Reject oversized repair context rather than trimming required text. Put repair-specific persistence in draft_repair.py rather than duplicating large WorkflowService transaction blocks.

Add a dedicated evidenced coverage step or equivalently persisted, budgeted chapter validation step before summarization; exact excerpt validation is deterministic. Missing/negative verdict pauses for review. Preserve existing legacy validation and all final body length/repetition checks.

- [ ] **Run focused tests then full suite; self-review migration upgrade/downgrade and legacy snapshots.**
- [ ] **Commit task changes:** `git commit -m "feat: add versioned bounded chapter draft repair"`.

### Task 2: Versioned stage roadmap and transactional rolling progress

**Files:**
- Create: src/ainovel/models/stage.py, src/ainovel/services/stages.py, src/ainovel/agents/stage_contracts.py
- Create: alembic/versions/0004_stage_roadmaps.py
- Modify: src/ainovel/models/__init__.py, src/ainovel/services/workflows.py, src/ainovel/workflows/orchestrator.py, src/ainovel/services/batches.py
- Modify: src/ainovel/agents/prompts.py only for stage planning snapshots
- Test: tests/test_stages.py, tests/test_orchestrator.py, migration tests

**Interfaces:**
- Consumes Task 1's generation_version=2 and evidenced chapter validation. Do not change their behavior.
- Produces StageService(session) methods to create a proposed stage from architecture, request bounded model roadmap generation, approve a particular roadmap version, query roadmap/progress/diff, and start_next_batch(stage_id, actor, provider_name, model_name, requested_chapters=5).
- Produces a persisted StageRoadmapDraft with goal, start/end state, key events, foreshadowing and compact ordered chapter nodes with stable IDs, titles, goals and dependencies. Estimated chapter count derives from nodes; no separate inconsistent count.
- Produces workflow-to-stage-version and chapter-node mappings with stage and book ordinals; pending mapping is not official progress.

- [ ] **RED: Tests against real DB/service plus FakeProvider.** Author architecture generates e.g. seven compact nodes. Approval is required before next batch. After approving five chapter contents, next batch contains two nodes, with global ordinals six and seven; before content approval a second batch is rejected. Core independent expectations:

```python
assert [n.stage_ordinal for n in first.nodes] == [1, 2, 3, 4, 5]
assert stage.confirmed_chapters == 0
assert [n.stage_ordinal for n in second.nodes] == [6, 7]
assert stage.confirmed_chapters == 5
```

Test duplicate approvals don't double-advance, rejected/cancelled batches don't advance, concurrent starts cannot reserve the same range, stale roadmap cannot complete, roadmap revision diff preserves approved records, and budget/oversized roadmap rejects rather than silently truncates. Test model-supplied invalid ordering/dependencies and stage completion requiring no extra chapters.

- [ ] **Run focused tests, record RED.**
- [ ] **GREEN: Persist stage/version/progress models with constraints.** Stage model calls use a finite persisted attempt budget and token accounting rather than untracked provider calls. Keep roadmap creation separate from batch writing and require explicit approval. Use compact roadmap payload in batch planning; include only relevant chapter plan in writing. Enforce 1–5 chapters server-side regardless of client input. Use existing transaction/optimistic revision patterns and serialize stage start/approval transitions.

Integrate progress with the authoritative BatchService approval transaction, not a later best-effort refresh. Retain approved roadmap versions and compare a proposed revision before approval. Guard edits that conflict with active batches. Do not overwrite committed earlier chapter nodes when revision changes future pacing. Keep stage and book chapter positions distinct from local batch ordinals.

- [ ] **Run focused tests then full suite; self-review transaction races and migration compatibility.**
- [ ] **Commit:** `git commit -m "feat: add approved stage roadmaps and rolling chapter progress"`.

### Task 3: Author-facing stage and repair controls, compatibility and acceptance

**Files:**
- Modify: src/ainovel/web/chapter_test_routes.py, workflow_routes.py, routes.py
- Modify: src/ainovel/chapter_test.py, app.py
- Create: src/ainovel/web/stage_routes.py, templates/stage.html
- Modify: src/ainovel/web/templates/chapter_test.html, workflow.html, project.html, presentation.py
- Modify: src/ainovel/static/app.css only for readable existing-style components
- Test: tests/test_stage_web.py, test_chapter_test.py, test_workflow_web.py
- Modify: README.md

**Interfaces:**
- Consumes Task 1 isolated repair read API and Task 2 StageService.
- Produces stage architecture setup and route previews with POST+CSRF approval/generation actions, and read-only GET pages. Existing single chapter test remains selectable and explicitly opts into v2 for newly created runs only.

- [ ] **RED: TestClient acceptance tests.** Submit architecture, observe no initial provider calls, request roadmap, review and approve it, generate/approve batch plan, generate bodies, then verify next-batch gate. Assert status and stored records as well as page text. Assert failed draft preview is escaped and clearly unapproved, failed attempts show repair round/count/usage and old generic errors still render. GET never calls provider or mutates stage/repair state. Missing CSRF/confirmation rejects actions.
- [ ] **Run focused tests, record RED.**
- [ ] **GREEN: Build routes and templates.** Stage setup uses label “剧情阶段总体架构”; keep “独立单章测试” separate. Show chapter count proposal, compact roadmap, stage/book ordinals, pending versus approved progress and revision differences. Buttons explicitly state model calls; approve buttons don't call models. Show latest work draft separately from accepted body and expose pause reasons. Display unknown usage as unknown. Provide explicit reuse-old-input action which creates a new versioned run and leaves old workflows unchanged; never retry on page load.
- [ ] **Run focused tests and full suite.** Add README usage flow and explanation that short repair adds bounded calls, has no guarantee of literary quality, and remains author-gated. Record all acceptance results including schema upgrade from existing test DB fixtures. Do not restart runtime service or invoke live Qwen.
- [ ] **Commit:** `git commit -m "feat: expose stage planning and draft repair in author UI"`.

## Plan self-review

Spec coverage: Task 1 covers isolated repair, word limits, usage and semantic coverage; Task 2 covers stage roadmap/version/progress/rolling batches; Task 3 covers user gates, compatibility UI, and offline end-to-end acceptance. Shared workflow and prompt files are edited sequentially, never by parallel implementers. Task 2 consumes Task 1's version switch; Task 3 consumes both service APIs. Existing runtime data and server processes remain outside this implementation pass.

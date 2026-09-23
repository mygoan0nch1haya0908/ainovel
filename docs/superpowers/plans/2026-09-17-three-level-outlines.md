# Three-level outlines implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Provide whole-book outline, current plot-stage outline and optional first-chapter outline with real generation-context propagation.

**Architecture:** Reuse immutable OutlineVersion/OutlineNode JSON payloads and StoryStage. New hierarchical setup mode stores separate parent-child nodes; legacy stage/single requests and saved snapshots remain unchanged. Existing roadmap and batch approvals remain the author gates.

**Tech Stack:** FastAPI, SQLAlchemy, Jinja, pytest, existing JavaScript.

**Spec:** User-approved chat requirements: total plot outline, stage outline, optional single-chapter outline; automatic subdivision and existing five-chapter approval gates. Implementation clarification: the optional chapter input targets the first chapter of the current stage, explicitly labelled; arbitrary later-chapter edits are not part of this form.

## Global Constraints

- Preserve prior uncommitted default-v2 changes in README, routes, template and tests; do not discard them.
- Whole-book outline and current stage outline required in new hierarchical mode; chapter outline optional. Independent single-chapter test remains available and v2 remains checked by default.
- No model calls on setup, reuse, GET or approval. No live Qwen calls, automatic generation, publication, old-task mutation or remote Git.
- Maintain CSRF, input limits, atomic setup rollback, frozen snapshots and bounded context; no truncating required constraints.
- Existing database schema is sufficient; no migrations or new dependencies.

### Task 1: End-to-end hierarchy setup, planning context and UI

**Files:** src/ainovel/web/chapter_test_routes.py, templates/chapter_test.html, templates/stage.html, static/app.js; src/ainovel/services/stages.py; tests/test_chapter_test.py, tests/test_stages.py; README.md. Add focused tests/test_three_level_outlines.py as needed. If prompt handoff requires orchestrator.py, change only its relevant stage context assembly.

**Interfaces:** add POST field book_outline (12000-character bounded string), setup_mode=hierarchical. Preserve legacy setup_mode stage/single_chapter handling. Store book payload book_outline, stage node stage-1 under book with stage_architecture, optional chapter-1 under stage-1 with chapter_outline/goal/ending_hook and stage_ordinal=1. Preserve provisional-ending under book. Do not duplicate chapter prose into book summary. _confirmed_input and reuse read new hierarchy with legacy fallback; old stage text must not be invented as a whole-book outline.

- [ ] RED: actual create_chapter_test_app form defaults hierarchical, exposes all three labels, validates missing book/stage but accepts absent chapter, saves separate node hierarchy atomically with no ModelAttempt. Test exemplar fields book_outline='全书调查失踪王朝，最终公开真相', stage_architecture='第一阶段调查邮局，找到失踪名单', chapter_outline='' and verify no chapter hint node. Provided chapter hint must persist with ordinal1 and reach first-node context only, not second/later batch nodes.
- [ ] RED: mock provider through real StageService captures distinct book/stage/optional chapter input in frozen snapshot; later batch context never mistakes first-chapter hint as a global instruction. Legacy snapshots still work; reuse retains all levels but never mutates source; oversize/CSRF/rollback tests remain green.
- [ ] GREEN: add bounded field/mode validation, parent-child outline creation, confirmed/reuse handling and existing stage redirect. Centralize any new hierarchy context extraction in one small helper within stages.py. Roadmap planner receives authoritative whole-book and current-stage outlines and optional first-node instruction; writer receives relevant node only. Explicitly instruct hierarchy precedence and ask for conflicts before author approval, never silently override upper outline; do not promise perfect semantic conflict detection or introduce unbounded automatic calls.
- [ ] GREEN: default hierarchical UI, distinct single mode, only relevant fields shown using minimal mode-switch JS; server remains authoritative with JS disabled. Hierarchical chapter field labelled '单章大纲（可选，当前阶段第一章）'; empty hint explains AI subdivision. Keep hints/whole/stage visible in confirmed input and stage page, escaped. Required states match selected mode. Preserve explicit paid-generation notices.
- [ ] Verify focused suites, then full offline suite; update README with actual flow and semantic limitations. Record RED/GREEN output and self-review in .superpowers/three-level-report.md. No commit/push required until controller review; return full changed-file list and concerns.

**Test command:** PYTHONPATH=D:/ainovel/.worktrees/qwen-adapter/src; D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short. Clear live provider flags/keys and AINOVEL_DATABASE_URL in test subprocess environment only. Baseline531passed1skip1knownwarning.

## Self-review

One cohesive task owns setup, persistence, context and rendering; one independent review follows. No competing writers. Legacy compatibility and optional hint scoping are the primary risks. Final runtime restart is controller-owned and must first verify no model attempt is RUNNING.

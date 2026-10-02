# Task 1 implementation report

## Status

Implemented versioned durable bounded chapter repair and exact-evidence chapter coverage for generation version 2. Legacy `WorkflowService.start(...)` remains generation version 1 by default.

## RED evidence

All commands used the required environment: `PYTHONPATH=D:/ainovel/.worktrees/qwen-adapter/src`, the existing `phase2-orchestration-context/.venv`, cleared live-provider/database environment variables, strict markers, isolated `--basetemp`, and no live model calls.

1. Initial v2 workflow/repair test:
   - Command: `python -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp-task1-red' -q --tb=short tests/test_draft_repair.py`
   - Result: `1 failed, 1 warning`; expected missing-feature failure: `WorkflowService.start() got an unexpected keyword argument 'generation_version'`.
2. Runtime integration boundary:
   - Command: focused `tests/test_draft_repair.py` with `.pytest-tmp-task1-green1`.
   - Result: `1 failed`; v2 plan remained `PLANNING` because the legacy schema rejected scenes.
3. Independent repair/protocol accounting boundary:
   - Command: focused `tests/test_draft_repair.py` with `.pytest-tmp-task1-green2`.
   - Result: `1 failed`; no third writer call because legacy `MAX_STEP_ATTEMPTS=2` coupled repairs to protocol retries.
4. Budget and evidence boundaries:
   - Command: focused `tests/test_draft_repair.py` with `.pytest-tmp-task1-red2`.
   - Result: `2 failed, 5 passed`; missing pre-call workflow budget pause and blank evidence was treated as protocol retry rather than review pause.
5. Demo compatibility boundary:
   - Command: focused `tests/test_draft_repair.py` with `.pytest-tmp-task1-red3`.
   - Result: `1 failed, 9 passed`; `DemoFakeProvider` still emitted a v1 plan for a v2 request.
6. Protocol/semantic independence and usage boundary:
   - Command: focused `test_v2_protocol_retry_does_not_consume_an_extra_semantic_repair` with `.pytest-tmp-task1-red4`.
   - Result: `1 failed`; durable `repair_pending` reservation was missing.
7. Budget ordering boundary:
   - Command: focused `test_v2_budget_rejection_does_not_spend_an_undispatched_repair` with `.pytest-tmp-task1-red5`.
   - Result: `1 failed`; repair count advanced before budget preflight.
8. Read API boundary:
   - Command: focused first repair test with `.pytest-tmp-task1-red6`.
   - Result: `1 failed`; immutable `list_for_workflow` view API was missing.
9. Downstream coverage report boundary:
   - Command: focused v2 batch-review coverage test with `.pytest-tmp-task1-red7`.
   - Result: `1 failed`; reviewer report still contained legacy literal coverage fields instead of evidenced coverage.

## GREEN and verification evidence

- First end-to-end repair/coverage GREEN: `tests/test_draft_repair.py`, `.pytest-tmp-task1-green3` -> `1 passed, 1 warning`.
- Expanded repair/budget/evidence GREEN: `tests/test_draft_repair.py`, `.pytest-tmp-task1-green4` -> `7 passed, 1 warning`.
- Demo/snapshot/lease GREEN: `tests/test_draft_repair.py`, `.pytest-tmp-task1-green5` -> `10 passed, 1 warning`.
- Protocol independence GREEN: focused test, `.pytest-tmp-task1-green6` -> `1 passed, 1 warning`.
- Budget ordering GREEN: focused test, `.pytest-tmp-task1-green7` -> `1 passed, 1 warning`.
- Final Task 1 focused GREEN: `tests/test_draft_repair.py`, `.pytest-tmp-task1-green8` -> `13 passed, 1 warning`.
- Migration/readiness GREEN: `tests/test_orchestration_schema.py` plus head/round-trip acceptance tests -> `18 passed, 1 warning`; final schema-only rerun -> `16 passed, 1 warning`.
- Related legacy regression GREEN: orchestrator/workflows/prompts/provider/demo tests -> `159 passed, 1 warning`; combined focused run -> `189 passed, 1 warning`.
- Required full offline suite (run once after final code changes):
  - Command: `python -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short`
  - Result: `457 passed, 1 skipped, 1 warning in 54.04s`.
  - Skip: optional live Ollama coverage, as expected. Warning: known Starlette `BlockingPortal` deprecation only.
- `git diff --check`: no whitespace errors; only the repository's Windows LF-to-CRLF notices.

## Implemented interfaces for Task 2/3

- `WorkflowService.start(..., generation_version=2)` explicitly opts in; omitted value remains v1.
- `GenerationWorkflow.generation_version` migrates existing rows to server-default `1`.
- V2 frozen snapshots are created through `PromptService.snapshot_versioned(workflow_id, prompt_bodies, schemas, parameters)` without activating or modifying v1/custom prompts.
- `ChapterDraftRepair` is uniquely keyed by `writing_step_id` and stores `latest_attempt_id`, `latest_payload`, `visible_count`, durable `repair_count`, `repair_pending`, and `draft_revision`.
- `DraftRepairService.list_for_workflow(workflow_id)` returns immutable `ChapterDraftRepairView` records containing ordinal/title/body/count/round/revision/attempt metadata for author-facing display.
- `VALIDATING_CHAPTER` is the persisted, budgeted v2 coverage step; `chapter_coverage` artifacts retain goal/hook verdicts and excerpts. Valid promotion creates `chapter_draft`; invalid/forged/missing/negative evidence pauses without promotion.
- Request metadata includes string `generation_version`; `DemoFakeProvider` uses it to preserve v1 output and emit v2 scenes/coverage when requested.
- V2 finite ceilings are stored in `model_call_limit`, `total_input_token_limit`, and `total_output_token_limit`; `model_calls_used` is incremented before each provider call. Configured per-call maxima are conservatively reserved against remaining total token capacity.

## Changed files

- `src/ainovel/agents/contracts.py`
- `src/ainovel/agents/prompts.py`
- `src/ainovel/agents/runner.py`
- `src/ainovel/db.py`
- `src/ainovel/models/__init__.py`
- `src/ainovel/models/workflow.py`
- `src/ainovel/providers/demo.py`
- `src/ainovel/providers/diagnostics.py`
- `src/ainovel/services/draft_repair.py` (new)
- `src/ainovel/services/prompts.py`
- `src/ainovel/services/workflows.py`
- `src/ainovel/workflows/orchestrator.py`
- `alembic/versions/0003_draft_repair.py` (new)
- `tests/test_draft_repair.py` (new)
- `tests/test_foundation_acceptance.py`
- `tests/test_orchestration_schema.py`

## Self-review

- Verified 2799 -> 3386 -> 5200 repair payloads use the latest complete body and exact gaps 1701/1114.
- Short/full/malformed responses remain distinct: only strict nonblank v2 work drafts are repairable; malformed/extra-field payloads remain protocol failures and do not create repair state.
- Work drafts are isolated as `chapter_work_draft` artifacts plus the latest keyed repair state. No raw body enters error details/diagnostics. A formal `chapter_draft` appears only after deterministic excerpt validation.
- Repair count is reserved durably before provider dispatch, but only after the budget preflight. `repair_pending` lets a protocol retry reuse the same semantic repair reservation; monotonic attempt numbers and stale completion fences are retained.
- Short-draft and malformed-response token metadata is persisted when available. Every v2 call increments the durable total-call counter before provider execution.
- V2 final deterministic checks retain title/body, 4500–6000 visible length, plan ordinal, continuity, and repeated-block validation; literal goal/hook inclusion checks remain unchanged only for v1.
- Migration upgrade/downgrade/re-upgrade and true 0002-row defaulting to v1 are covered. Active custom v1 prompt selection is unchanged by v2 snapshot creation.
- No Task 2 stage-roadmap or Task 3 route/template behavior was implemented.

## Concerns / explicit conservative behavior

- Total token preflight reserves each snapshotted call's configured maximum, not an estimate of likely actual usage. This can pause conservatively even when previous actual usage was low; it deliberately rejects rather than trims required repair text.
- The finite v2 ceilings include both allowed protocol attempts for planning, each initial/repair writer dispatch, coverage, summarization, and batch review. They are intentionally hard upper bounds, not user-facing cost estimates.
- Valid coverage is evidence that the model-supplied excerpt occurs exactly in the body; it is not an objective semantic proof, consistent with the approved spec.

## Review fix round 1 (2026-09-15)

### Findings and root causes

1. An expired v2 writing lease failed its running `ModelAttempt`, but recovery did not increment the separate `protocol_failure_count` used by the v2 dispatch gate. Repeated abandoned calls could therefore exceed the two-attempt protocol limit.
2. The orchestrator validated a successfully returned response inside the provider-error `try` block, but business-validation `ResponseFailure` objects do not carry a response. The failure path consequently discarded token usage already present on `run.response`.

### RED evidence

- Command: `python -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp-task1-review-red-control' -q --tb=short tests/test_draft_repair.py -k 'initial_writer_expired_attempts or pending_repair_expired_attempts or expired_claim_without_model_attempt or business_rejected_writer_response'`
- Result: `4 failed, 1 passed, 13 deselected, 1 warning in 1.82s`.
  - Initial and pending-repair cases both dispatched a third writer and raised another stale-completion conflict.
  - Oversized and repeated responses both stored `None` rather than `211/322` input/output usage.
  - The no-`ModelAttempt` expired-claim control passed, confirming recovery must not charge request/context construction stalls.

### GREEN and verification evidence

- Focused review regressions: same selection with `.pytest-tmp-task1-review-green` -> `5 passed, 13 deselected, 1 warning in 1.15s`.
- Related v1/v2 regression command: `python -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp-task1-review-related' -q --tb=short tests/test_draft_repair.py tests/test_workflows.py tests/test_orchestrator.py` -> `119 passed, 1 warning in 16.73s`.
- Fresh full offline suite: `python -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp-task1-review-full' -q --tb=short` -> `462 passed, 1 skipped, 1 warning in 53.57s`.
- Skip remains the optional live Ollama test; warning remains the known Starlette `BlockingPortal` deprecation.

### Review-fix changed files

- `src/ainovel/services/workflows.py`
- `src/ainovel/workflows/orchestrator.py`
- `tests/test_draft_repair.py`
- `.superpowers/sdd/2026-09-14-stage-planning-and-draft-repair/task-1-report.md`

### Review-fix self-review and concerns

- Recovery charges one v2 writing protocol failure only when that expired step has an actual `RUNNING` model attempt. The workflow/step revision and lease predicates remain the write fence; monotonic `attempt_count` and `attempt_number` are unchanged.
- A pending semantic repair reservation remains pending across expired provider attempts, so retrying an abandoned call cannot spend an extra repair round.
- Provider/parse failures still use the response attached to the exception. Only post-response business-validation failures use the already-returned `run.response`.
- V1 attempt-limit behavior is unchanged and is included in the 119-test related regression run.
- No plan/spec deviation or new concern was introduced by this fix round.

# Secure model profiles: acceptance and final fixes

Base: `cfabb0faa59f5a113db50606dca6294a16c0af63` (clean checkout before this wave).

## Changes

- `model_profile_routes.py`: bounded async form parsing remains on the event loop; the complete synchronous mutation/diagnostic action now runs in a worker with a session opened and closed there. The resolver retains its independent session. Both model listing and synthetic connection testing are off the event loop. Existing CSRF, explicit confirmation, no-store responses, fixed errors and secret-free output remain.
- Project workflow and stage roadmap creation render all validation errors with the submitted profile version, provider, model and other non-secret form values. Enabled historical versions remain selectable; disabled, revoked and missing versions get the selected unavailable marker and cannot create a task on retry. Consent is cleared. Explicit legacy fake/Qwen choices are preserved. The selector script retains the submitted model on initial retry-page load while keeping normal change behavior.
- The connection diagnostic accepts only a dictionary with exactly `ok: True` where the value is the literal boolean, rejecting numeric and string lookalikes.

## RED/GREEN evidence

Every pytest invocation used this process-only environment prelude in PowerShell:

```powershell
$names = @('AINOVEL_OPENAI_API_KEY','AINOVEL_QWEN_API_KEY','OPENAI_API_KEY','QWEN_API_KEY','DASHSCOPE_API_KEY','AINOVEL_DATABASE_URL')
foreach ($name in $names) { Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue }
$env:AINOVEL_ALLOW_REAL_OPENAI = 'false'
$env:AINOVEL_ALLOW_REAL_QWEN = 'false'
$env:AINOVEL_RUN_OLLAMA_TESTS = '0'
$env:PYTHONPATH = 'D:/ainovel/.worktrees/qwen-adapter/src'
```

Commands below ran from `D:/ainovel/.worktrees/qwen-adapter`, with `D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe` as Python:

```powershell
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short tests/test_compatible_provider.py -k requires_literal_boolean_true
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short tests/test_model_profile_web.py -k blocked_diagnostic_allows_health_and_revoke
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short tests/test_model_profile_web.py -k creation_error_retains
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short tests/test_model_profile_web.py -k selector_javascript_preserves
```

Before implementation, the strict test failed for `1` and `1.0` (2 failed, 2 passed); both blocked-diagnostic tests failed because zero of two concurrent health/revoke requests completed; all 14 form round-trip cases failed; the real Node selector test failed because `saved-model` replaced `submitted-model` on load. After implementation, each targeted test passed.

Final focused command:

```powershell
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short tests/test_model_profile_web.py tests/test_compatible_provider.py tests/test_workflow_web.py tests/test_stage_web.py tests/test_chapter_test.py tests/test_three_level_outlines.py
```

Result: **156 passed, 1 known Starlette/AnyIO deprecation warning**, 0 failures, in 19.76 seconds. `git diff --check` exited 0; Git emitted only local LF-to-CRLF normalization warnings.

## Scoped self-review

- Confirmed no SQLAlchemy Session crosses from async parsing to the worker or between concurrent requests. The diagnostic test uses a blocking fake transport and verifies health plus revoke finish before release, for both list and test actions. The transport is in-memory and makes no external request.
- Confirmed public-only context is used for retry display. Historical enabled selections remain selected; disabled, revoked and missing versions use a visibly unavailable option. Retrying them remains rejected without creating a workflow or roadmap. Correcting unrelated errors requires renewed consent and binds the same version. Raw model IDs remain in form values and are escaped by Jinja or assigned via `textContent`.
- Confirmed the provider diagnostic rejects `1`, `1.0` and strings while accepting literal `true`; no payload/credential serialization or fixed-error path changed.
- No live key, network call, runtime database, service restart, push, or browser visual claim. The full repository suite is left to the main agent after this focused review.

## Final acceptance

The main agent ran the full offline suite on product commit `afc143b8e7ce4d769e2d1e1155ffacbc113dbd53`, with the same process-only environment above, from the isolated qwen-adapter worktree:

```powershell
& 'D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe' -m pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short
```

Result: **751 passed, 1 skipped, 1 warning in 86.50 seconds**, exit 0. The skip is the existing opt-in live model test; the warning is the pre-existing Starlette/AnyIO BlockingPortal deprecation.

Independent task reviews and scoped fix reviews are complete. Final whole-branch review covered `992bcbb..cfabb0f` using risk-based integration passes; its two Important findings and one Minor finding were fixed in `afc143b`. Final scoped re-review found all three addressed, no new Critical/Important breakage, and no open findings. This document relocation and acceptance appendix change documentation only after that tested product commit.

Implemented entry: `/model-profiles`. Supports protected credential storage, immutable configuration versions, explicit model-list/connection-test operations, and selection for project, single-chapter and hierarchical/stage flows. Offline acceptance exercised author plan/body approval, revision isolation, revocation, real shipped JavaScript, and sentinel scans of HTML, logs, database, prompts, artifacts and official chapter serialization.

The existing runtime database/service has NOT been upgraded or restarted. Before enabling the new page there, obtain author authorization, verify no active model call, create a consistent SQLite backup, migrate only that exact isolated database to `0006_model_profile_bindings`, then restart and check local readiness. Do not automatically import an existing key or make a paid call. No new push or merge was performed.

## Recorded decisions and remaining verification boundaries

1. A changed normalized target requires newly entered credentials rather than copying an existing key. This chooses the stricter approved option; its cost is key re-entry.
2. Keyless loopback-to-loopback revisions may stay keyless because local credentials are optional. The network loopback checks remain essential; an incorrect boundary would permit an unintended local destination.
3. Model identifiers stay exact raw data, with HTML escaping only at template/textContent boundaries. Pre-escaping changes protocol identifiers; bypassing the rendering boundary would risk unsafe display.
4. Existing environment-provider SDK transports were preserved, not redesigned. Their legacy risks are not covered by the new pinned-profile transport protections.
5. No export endpoint exists; acceptance covers actual official-chapter serialization. A future exporter needs its own privacy acceptance.
6. No live provider, billing or production-key check was performed without author authorization. Real-provider interoperability still requires author testing.
7. Actual browser layout QA remains unperformed after two browser-controller failures. Template/TestClient/Node verification does not rule out visual or accessibility defects.
8. The existing Starlette/AnyIO warning is left for dependency maintenance. Future dependency compatibility needs follow-up.

Already admitted/in-progress requests may finish after revocation; subsequent dispatches recheck authorization and stop. Neither ciphertext deletion nor revocation erases old backups, recalls provider-received content, or guarantees process-memory erasure.

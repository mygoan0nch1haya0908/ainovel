# Secure model profiles implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Execute tasks sequentially with test gates.

**Goal:** Visual Base URL/key/model configuration with encrypted credentials and immutable generation bindings.

**Architecture:** Separate public profile versions, current-user DPAPI vault, pinned outbound transport and generic Chat Completions adapter. Integrate through existing workflow/stage services, preserving legacy providers. Network enforcement occurs at the socket boundary, not just a preflight URL check.

**Tech Stack:** Python/ctypes/SQLite/SQLAlchemy/Alembic, FastAPI/Jinja, existing pytest/JavaScript. Prefer standard-library socket/ssl/http.client for pinned transport rather than undocumented SDK hooks.

**Spec:** `docs/superpowers/specs/2026-09-23-secure-model-profiles-design.md`, confirmed by user.

## Global Constraints

- 保存、替换密钥、停用、删除需要 POST + CSRF；GET 与保存不访问服务商。
- 验证失败不回显提交的密钥；成功后输入框为空，只显示“已配置/未配置”，不显示密钥尾部。
- Windows 当前用户 DPAPI 加密密钥；应用库只保存不透明凭据引用。
- 非 Windows 平台或加密失败时拒绝保存密钥，绝不回退明文或硬编码加密密钥。
- URL 不允许 userinfo、query、fragment；远程 HTTPS，本机显式授权且仅 loopback；TLS 验证不可关闭。
- 禁用环境代理、跨目标重定向和隐式 SDK 重试。Reject all redirects for the first implementation.
- 配置修改不静默切换旧任务；撤销后暂停，不能回退环境默认模型。
- 默认测试完全离线；真实获取列表或连接测试必须由作者在页面主动操作。
- Preserve existing uncommitted three-level/default-v2 changes; record initial diff, never reset/stash. Existing isolated worktree is D:/ainovel/.worktrees/qwen-adapter.
- No live key import, automatic remote operations, runtime migration or real model test during implementation.

## Review Focus

1. DNS changes between validation and connection: pin approved IP and preserve original TLS hostname. Task2 connector-boundary tests.
2. Provider echoes a key in model IDs, errors or generated text: reject unsafe response before display/persistence. Task2/4 sentinel scans.
3. Vault write succeeds but metadata commit fails, or deletion crashes: compensate writes, commit revocation before cleanup, fail closed. Task1 fault injection.
4. Cached provider reused after revocation: authorization check before every dispatch, no fallback. Task3 cache/race tests.
5. Malformed multipart or browser autofill causes validation echo: empty secret field on every render and fixed errors. Task4 malformed-input tests.

## Shared interfaces and limits

```python
@dataclass(frozen=True)
class ProfileInput:
    name: str
    base_url: str
    connection_kind: str  # remote or loopback
    model_name: str
    context_limit: int = 32000
    output_limit: int = 12000

class SecretVault(Protocol):
    def put(self, secret: str) -> str: ...
    def get(self, reference: str) -> str: ...
    def delete(self, reference: str) -> None: ...

class ModelProfileService:
    def create(self, values: ProfileInput, *, api_key: str | None) -> ProfileView: ...
    def revise(self, profile_id: str, values: ProfileInput, *, api_key: str | None = None,
               keep_existing_key: bool = False) -> ProfileView: ...
    def set_enabled(self, version_id: str, enabled: bool) -> None: ...
    def revoke(self, profile_id: str) -> None: ...
    def list_public(self) -> list[ProfileView]: ...
    def get_public(self, version_id: str) -> ProfileView: ...
    def resolve_for_call(self, version_id: str) -> ResolvedProfile: ...
```

ProfileView exposes profile_id/version_id/name/base_url/kind/model/limits/enabled/revoked/has_key, never key/ciphertext/credential retrieval reference. ResolvedProfile is internal, repr-redacted, never serialized. Vault defaults to D:/ainovel/.worktrees/qwen-adapter/.superpowers/runtime/model-profiles/credentials.db, with injectable test path and Windows current-user file protection; failure to establish protection blocks secret persistence.

Bounds: name120, URL2048, model255, key8192 characters; context1..32000, output1..12000 and output<=context. Remote key required, local optional. Same-target key reuse requires explicit checkbox; changed target requires newly entered key. No URL secrets/custom headers/proxy configuration. Reject metadata equal to submitted key; warn against putting keys in ordinary prose.

Transport limits: connect10 seconds, total request deadline<=180 seconds or lower workflow limit; models<=1MiB/200 unique IDs/255 chars each; generation<=2MiB; synthetic connection test<=64KiB/max_output_tokens128. No redirects, automatic retries, proxy environment or decompression. Bound streaming reads and close connections. List results do not establish model capabilities.

### Task 1: Encrypted vault and immutable profile metadata

**Files:** create security/secret_vault.py, models/model_profile.py, services/model_profiles.py, providers/endpoint_policy.py under src/ainovel; migration0005_model_profiles.py; modify model exports/db readiness; tests/test_model_profiles.py, test_secret_vault.py, test_model_profile_migration.py.

**Interfaces:** implement shared contracts. normalize_endpoint(value, connection_kind) returns Endpoint(base_url,host,port,path,kind) with syntax checks only, zero DNS during save. Task2 adds resolve-and-validate network methods.

- [ ] RED real session tests for public DTO/SQL secrecy, immutable revisions, enable/revoke, bounds, absent remote key and invalid URL. Use a test-only vault, not a production plaintext fallback.

```python
created = service.create(remote_values, api_key="SENTINEL_KEY_42")
assert created.has_key
assert "SENTINEL_KEY_42" not in repr(created)
new = service.revise(created.profile_id, changed_model, keep_existing_key=True)
assert new.version_id != created.version_id
assert service.get_public(created.version_id).model_name == remote_values.model_name
```

- [ ] Run focused RED; add synthetic Windows DPAPI round-trip and unsupported-platform rejection tests. Platform skip applies only to the Windows integration test.
- [ ] GREEN current-user CryptProtectData/CryptUnprotectData with bounded buffers, LocalFree cleanup and fixed errors. Store ciphertext separately with restrictive access. Each new version receives a fresh credential ref, even when explicitly reusing a key.
- [ ] Implement metadata transactions; compensate vault insertion on SQL failure. Commit revocation before deleting ciphertext; cleanup failure remains revoked. Re-enabling cannot resurrect destroyed credentials. Document backup limits and already-sent requests.
- [ ] Test injected commit/encryption/delete failures and vault restart persistence. Migration from0004 preserves old data; no key in SQL dump/audit/logs. Run focused/full offline suites, then commit only owned files: feat: add encrypted versioned model profiles.

### Task 2: Pinned transport and compatible provider

**Files:** create providers/safe_transport.py, providers/compatible.py; extend endpoint_policy.py; tests/test_endpoint_policy.py, test_safe_transport.py, test_compatible_provider.py.

**Interfaces:** SafeTransport.request_json(endpoint, method, relative_path, *, api_key, payload=None, timeout_seconds, max_response_bytes) -> dict. Relative paths restricted to models/chat/completions. CompatibleProvider implements ModelProvider plus list_models() -> tuple[str,...] and test_connection() -> ProviderDiagnostic. Constructor/capabilities/diagnose are local only.

- [ ] RED private/metadata/mixed DNS/IPv4-mapped IPv6, userinfo/query/fragment, invalid ports, redirects, proxy env, malformed/oversized/chunked responses. Use a loopback test server and injected resolver/socket boundary, not external network.

```python
resolver.answers = ["93.184.216.34"]
transport.request_json(endpoint, "GET", "models", api_key=sentinel,
                       timeout_seconds=10, max_response_bytes=1048576)
assert connector.addresses == [("93.184.216.34", 443)]
assert connector.tls_server_names == [endpoint.host]
assert connector.verify_certificates is True
```

- [ ] GREEN validate every resolved address, then connect only to approved numeric IP, wrapping TLS with original hostname and sending correct Host. Never re-resolve during connect. If pinning/certificate checks cannot be maintained, block networking rather than weakening validation.
- [ ] Enforce total/read/connect deadlines and response limits even without Content-Length; reject redirects/encoding, close on every path. Standard-library transport ignores proxy variables. No insecure retry after certificate failure.
- [ ] Implement generic Chat Completions JSON Object requests with schema instruction and local validation; do not send Qwen-only thinking options to arbitrary providers. Retain valid usage on invalidJSON/refusal/truncation; errors fixed and body-free.
- [ ] Test key echoed in IDs/errors/success text/structured fields; unsafe results are rejected before persistence, not merely hidden in UI. Sanitize usage/response IDs. Model list exposes bounded escaped IDs only; test uses synthetic input/max128 tokens, never novel context.
- [ ] Focused/full offline tests and commit: feat: add guarded compatible model transport.

### Task 3: Immutable profile binding in generation

**Files:** workflow/stage models, registry, WorkflowService, StageService, orchestrator, app wiring; new services/provider_resolution.py; migration0006_model_profile_bindings.py; tests/test_profile_workflows.py, test_profile_binding_migration.py.

**Interfaces:** nullable model_profile_version_id FK on GenerationWorkflow and StageRoadmapVersion. Existing start/propose_roadmap accept optional keyword model_profile_version_id=None. New provider name compatible requires a version; legacy providers reject mismatched profile binding. ProviderResolver.resolve(provider_name, model_name, *, model_profile_version_id=None) -> ModelProvider; returned proxy checks enabled/revoked on every dispatch, including cache reuse.

- [ ] RED workflow uses versionA, edit createsB, old task remainsA; revokeA blocks cached retries with no network. Stage proposal and next batch inherit the exact approved binding.

```python
workflow = workflows.start(project_id, "compatible", version_a.model_name, 1,
    budgets, generation_version=2, model_profile_version_id=version_a.version_id)
service.set_enabled(version_a.version_id, False)
orchestrator.advance(workflow.id)
assert network.calls == []
assert session.get(GenerationWorkflow, workflow.id).model_profile_version_id == version_a.version_id
```

- [ ] GREEN bind within existing ownership transaction; snapshots contain safe metadata only. Model/version mismatch rejects; no fallback to environment credentials. Resolver applies consistently to workflow, stage, diagnostics and retry paths.
- [ ] Clamp role budgets to profile/system ceilings; preserve overall attempts/Token limits and existing overflow pauses. Do not assume small output ceilings can generate a full chapter.
- [ ] Cover revision/revoke races, cached instances, stage retries, secret-free snapshots/artifacts/audits and legacy providers. Already-sent request cannot be recalled, but subsequent dispatches must stop. Add impacted-task counts for revocation UI.
- [ ] Upgrade/downgrade fixtures and readiness0006, focused/full regression; commit feat: bind generation to immutable model profiles.

### Task 4: Visual configuration and privacy acceptance

**Files:** new web/model_profile_routes.py, templates/model_profiles.html and model_profile.html; app/chapter_test wiring; base/project/chapter_test/stage/workflow templates and routes; README; tests/test_model_profile_web.py and existing web suites.

**Interfaces:** GET /model-profiles and /model-profiles/{id}; POST /model-profiles create, /check local validation, /{id}/revise, /{id}/revoke, /versions/{version_id}/enabled, /versions/{version_id}/models, /versions/{version_id}/test. All under /model-profiles with CSRF; external/destructive operations require explicit confirmation.

- [ ] RED actual TestClient tests for masking, no-network GET/save, missing CSRF/confirmation, malformed multipart, duplicate secret fields, oversized input and provider error echoes.

```python
response = client.post("/model-profiles", data={**valid_form, "api_key": sentinel,
                                              "name": "", "csrf_token": csrf})
assert response.status_code == 422
assert sentinel not in response.text
assert sentinel not in captured_logs
assert network.calls == []
```

- [ ] GREEN DTO-only pages, Cache-Control:no-store; password autocomplete=new-password and never populated value. No localStorage/cookies/session secrets, third-party scripts or raw service HTML. Fixed request-validation errors avoid FastAPI body echoes. Check action performs no DNS.
- [ ] Add profile version selection to single/hierarchical setup, project starts and stage proposals. Display destination and model plus author consent before sending novel context. Stage batch inherits approved binding; changing provider requires a new approved proposal. Preserve environment Qwen/fake choices.
- [ ] Model list is explicit POST, failure leaves manual model entry usable; connection test warns possible cost. Revoke shows affected tasks and backup/in-flight limitations. No automatic call on page load or save.
- [ ] End-to-end fake service save/list/select/plan/approve/write/approve, edit binding isolation and revoke pause; sentinel scans across DB/logs/HTML/audit/snapshots/exports. Bound and escape malicious model IDs. Verify actual JS behavior separately from browser layout.
- [ ] README covers DPAPI account coupling, authorized provider receives key/prompt, local-only deployment and all limits. Full offline suite, independent security review and fix findings before delivery. Commit feat: expose secure model profile configuration.

## Verification and rollout

Use D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe with explicit PYTHONPATH=D:/ainovel/.worktrees/qwen-adapter/src and pytest -o 'addopts=--strict-markers --basetemp=.pytest-tmp' -q --tb=short. Clear live provider flags/keys and runtime DB overrides in test subprocess only. Never parallelize suites sharing basetemp. Start with fresh baseline; previous542passed1skip known warning is not fresh evidence.

After implementation and review, request authorization for the actual testing DB upgrade/restart; first verify no RUNNING calls and create a consistent SQLite backup. Migrate only the exact isolated database. Never import the existing real key automatically. Verify GET readiness/pages without outbound calls. No push/merge without user request.

## Self-review

All spec requirements map to tasks: storage/lifecycle1, URL/socket/provider2, immutable execution3, visual/privacy4. Each Review Focus item is pinned by corresponding task tests. Shared app/model/routes files are edited sequentially. No architectural weakening is permitted to avoid a difficult security test. Execution method and written plan require user confirmation before product code changes.

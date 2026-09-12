# AI Novel Studio

## Local setup

AI Novel Studio stores projects, outlines, prompt snapshots, workflow state, and
candidate batches locally. By default, its SQLite data is stored in
`./ainovel.db` in the directory where the server is started. The built-in Fake
Provider is the default deterministic workflow demonstration and makes no model
or network request.

Create the environment, install dependencies, apply the required Alembic migration, and start the server:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\uvicorn ainovel.app:app --host 127.0.0.1 --reload
```

Alembic upgrade is required before starting the application. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The server accepts only `127.0.0.1`, `localhost`, and the test host. Always bind
Uvicorn to `127.0.0.1`; do not expose it on a LAN or the public internet because
the local workbench does not provide network-user authentication. Browser
mutations use signed, session-backed CSRF tokens. By default, a cryptographically
random session secret is generated for each server process; set
`AINOVEL_SESSION_SECRET` when sessions must survive a local restart.

## Migrations

For an existing local database, apply all available migrations before running the application:

```powershell
.\.venv\Scripts\alembic upgrade head
```

Pytest is configured to keep temporary test artifacts in the worktree-local `.pytest-tmp` directory, which Git ignores.

## Health checks

- `GET /health` is a process liveness check and always preserves the response `{"status":"ready"}` while the application can serve requests.
- `GET /ready` is a readiness check. It verifies database connectivity and that the database revision is at Alembic head, returning HTTP 503 with a diagnostic when the schema is absent or outdated.

## Tests

The default suite is offline and requires neither Ollama nor an OpenAI API key:

```powershell
Remove-Item Env:AINOVEL_RUN_OLLAMA_TESTS -ErrorAction SilentlyContinue
Remove-Item Env:AINOVEL_OLLAMA_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:AINOVEL_ALLOW_REAL_OPENAI -ErrorAction SilentlyContinue
Remove-Item Env:AINOVEL_OPENAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:AINOVEL_DATABASE_URL -ErrorAction SilentlyContinue
.\.venv\Scripts\python -m pytest -q
```

## Fake workflow demo

Apply migrations and start the loopback-only workbench:

```powershell
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\uvicorn ainovel.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), create or open a project,
approve its creation constitution and official outline, then choose Provider
`fake`, model `demo`, and 1–5 chapters. Run to the plan gate, approve the plan,
run to the candidate-content gate, and use the existing candidate-batch approval
before synchronizing the workflow decision.

Fake output verifies orchestration, persistence, crash-safe gates, and approval
boundaries. It does **not** evaluate or demonstrate literary quality.

See [Phase 2 operational boundaries](docs/phase-two-operational-boundaries.md)
for claim ownership, strict summary encoding, cross-batch approved memory,
model budgets, and cancellation of paused workflows.

## Ollama diagnosis and optional local smoke test

Ollama is optional. AI Novel Studio never downloads or removes a model. Inspect
the already installed local models and diagnose the loopback service with:

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

With the loopback server running, the project page can diagnose Provider
`ollama` using an installed model name. To opt into the structured-output smoke
test, copy an exact model name from `ollama list` and run:

```powershell
$env:AINOVEL_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:AINOVEL_RUN_OLLAMA_TESTS = "1"
$env:AINOVEL_OLLAMA_MODEL = "<installed-model-name>"
.\.venv\Scripts\python -m pytest tests\test_ollama_live.py -v
Remove-Item Env:AINOVEL_OLLAMA_BASE_URL, Env:AINOVEL_RUN_OLLAMA_TESTS, Env:AINOVEL_OLLAMA_MODEL
```

The smoke test contacts only the configured `127.0.0.1` service, performs one
diagnostic and one short JSON-Schema request, and never pulls a model or calls a
cloud endpoint.

## Explicit Qwen opt-in

Qwen uses Alibaba Cloud's OpenAI-compatible Beijing endpoint and defaults to
model `qwen-flash`. It is a remote API: AI Novel Studio does not download a
model, and requests may incur external charges. Real requests remain disabled
unless both an API key and `AINOVEL_ALLOW_REAL_QWEN=true` are present.

```powershell
$secureQwenKey = Read-Host "DashScope API key" -AsSecureString
$qwenCredential = [pscredential]::new("unused", $secureQwenKey)
$env:AINOVEL_QWEN_API_KEY = $qwenCredential.GetNetworkCredential().Password
$env:AINOVEL_ALLOW_REAL_QWEN = "true"
$env:AINOVEL_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
.\.venv\Scripts\uvicorn ainovel.app:app --host 127.0.0.1 --port 8000
Remove-Item Env:AINOVEL_QWEN_API_KEY, Env:AINOVEL_ALLOW_REAL_QWEN, Env:AINOVEL_QWEN_BASE_URL
```

`DASHSCOPE_API_KEY` is accepted as a fallback; `AINOVEL_QWEN_API_KEY` takes
precedence when both are set. Keep credentials in the process environment, not
in files, logs, screenshots, or the database. Restart an already-running server
after changing environment variables so the process inherits them.

The project-page Qwen diagnosis checks configuration only; it is explicitly not
an online connectivity test. Qwen is requested with JSON Object mode and
thinking disabled. JSON Object mode is not strict schema enforcement, so the
application still validates every result against its Pydantic contract. The
adapter retains a 16,000-token context ceiling and a 4,000-token output ceiling;
the latter is generally too small for this application's full-length chapter
contract, so use Qwen here only for smaller structured steps unless budgets and
product requirements are separately revisited.

Any live smoke validation must be explicitly opted in and limited to one small,
synthetic structured request. It must not create, approve, or publish novel
chapters and must not be part of the default offline test suite.

## Explicit OpenAI opt-in

Real OpenAI requests are disabled unless both the opt-in and an API key are
present. The following PowerShell reads the key without echoing it and keeps the
server loopback-only. Real requests may incur API charges.

```powershell
$secureOpenAIKey = Read-Host "OpenAI API key" -AsSecureString
$openAICredential = [pscredential]::new("unused", $secureOpenAIKey)
$env:AINOVEL_OPENAI_API_KEY = $openAICredential.GetNetworkCredential().Password
$env:AINOVEL_ALLOW_REAL_OPENAI = "true"
.\.venv\Scripts\uvicorn ainovel.app:app --host 127.0.0.1 --port 8000
Remove-Item Env:AINOVEL_OPENAI_API_KEY, Env:AINOVEL_ALLOW_REAL_OPENAI
```

Do not place credentials in the database, command history, README, screenshots,
or logs. If either environment variable is absent, OpenAI remains unavailable
without making application startup fail.

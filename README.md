# AI Novel Studio

## Local setup

Phase one stores projects, outlines, and candidate batches locally. It does not call a model API. By default, its SQLite data is stored in `./ainovel.db` in the directory where the server is started.

Create the environment, install dependencies, apply the required Alembic migration, and start the server:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\uvicorn ainovel.app:app --host 127.0.0.1 --reload
```

Alembic upgrade is required before starting the application. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The Phase 1 server accepts only `127.0.0.1`, `localhost`, and the test host. Do not expose it on a LAN or the public internet: network exposure requires authentication, which Phase 1 does not provide. Browser mutations use signed, session-backed CSRF tokens. By default, a cryptographically random session secret is generated for each server process; set `AINOVEL_SESSION_SECRET` when sessions must survive a local restart.

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

```powershell
.\.venv\Scripts\python -m pytest -q
```

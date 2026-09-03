# AI Novel Studio

## Local setup

Phase one stores projects, outlines, and candidate batches locally. It does not call a model API. By default, its SQLite data is stored in `./ainovel.db` in the directory where the server is started.

Create the environment, install dependencies, apply the required Alembic migration, and start the server:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\uvicorn ainovel.app:app --reload
```

Alembic upgrade is required before starting the application. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Migrations

For an existing local database, apply all available migrations before running the application:

```powershell
.\.venv\Scripts\alembic upgrade head
```

Pytest is configured to keep temporary test artifacts in the worktree-local `.pytest-tmp` directory, which Git ignores.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

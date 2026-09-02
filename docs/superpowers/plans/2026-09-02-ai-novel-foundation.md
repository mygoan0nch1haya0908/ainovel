# AI Novel Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable local Web foundation that safely separates candidate writing from official novel state and supports versioned outlines plus atomic batch approval.

**Architecture:** Use FastAPI with server-rendered Jinja pages, SQLAlchemy 2 over SQLite, and service-layer transactions. HTTP routes call application services; services own validation and commits; ORM models only map persisted state. This phase uses no model API, so every workflow is deterministic and testable.

**Tech Stack:** Python 3.14.2, FastAPI, SQLAlchemy 2, Pydantic 2, Alembic, Jinja2, python-multipart, pytest, pytest-cov, HTTPX

**Spec:** `docs/superpowers/specs/2026-09-02-ai-novel-agent-design.md`

## Global Constraints

- Store all project data locally in SQLite.
- A batch contains 1—5 chapters.
- A chapter must contain 4500—6000 visible body characters before approval.
- Visible character count includes Han characters, digits, letters, and punctuation; it excludes whitespace, title, analysis, and reports.
- Candidate chapters and state deltas cannot change official state before author approval.
- Batch approval must commit chapters, outline version, state deltas, and audit event atomically.
- Published chapters cannot be overwritten by ordinary edit or approval paths.
- Use service-layer methods as the only state-changing interface.
- Run tests with a temporary SQLite database; tests must not write to the user database.
- No real model calls in this phase.

---

## File Structure

```text
pyproject.toml                         package metadata and dependencies
.gitignore                             local database, environment and cache exclusions
src/ainovel/__init__.py                package marker
src/ainovel/config.py                  typed local settings
src/ainovel/app.py                     FastAPI application factory
src/ainovel/db.py                      engine, session factory and SQLite settings
src/ainovel/models/base.py             SQLAlchemy base and shared timestamps
src/ainovel/models/project.py          novel project and constitution version
src/ainovel/models/outline.py          outline version and outline nodes
src/ainovel/models/batch.py            candidate batch and chapter records
src/ainovel/models/audit.py            append-only approval/rejection log
src/ainovel/services/projects.py       project creation and lookup
src/ainovel/services/outlines.py       immutable outline version operations
src/ainovel/services/batches.py        candidate editing, approval and rejection
src/ainovel/services/counting.py       visible character counting
src/ainovel/web/routes.py              browser routes
src/ainovel/web/templates/base.html    shared local UI shell
src/ainovel/web/templates/index.html   project list and create form
src/ainovel/web/templates/project.html project, outline and batch dashboard
src/ainovel/static/app.js              small local interaction helpers
tests/conftest.py                      isolated app and SQLite fixtures
tests/test_health.py                   application boot test
tests/test_projects.py                 project workflow tests
tests/test_outlines.py                 outline version tests
tests/test_batches.py                  candidate and atomic approval tests
tests/test_web.py                      local Web workflow tests
alembic.ini                            migration configuration
alembic/env.py                         migration environment
alembic/versions/0001_foundation.py    initial schema
```

---

### Task 1: Package Scaffold and Application Factory

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/ainovel/__init__.py`
- Create: `src/ainovel/config.py`
- Create: `src/ainovel/app.py`
- Create: `tests/test_health.py`

**Interfaces:**
- Produces: `ainovel.app.create_app(database_url: str | None = None) -> FastAPI`
- Produces: `ainovel.config.Settings(database_url: str, app_name: str)`

- [ ] **Step 1: Initialize Git and declare the Python package**

Run:

```powershell
git init
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "ainovel"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
  "fastapi>=0.115",
  "jinja2>=3.1",
  "python-multipart>=0.0.20",
  "sqlalchemy>=2.0",
  "alembic>=1.14",
  "pydantic-settings>=2.7",
  "uvicorn>=0.34",
]

[project.optional-dependencies]
test = [
  "httpx>=0.28",
  "pytest>=8.3",
  "pytest-cov>=6.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
```

Create `.gitignore` with:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.coverage
htmlcov/
*.pyc
*.db
*.db-shm
*.db-wal
.env
```

Install the editable package:

```powershell
.\.venv\Scripts\python -m pip install -e ".[test]"
```

- [ ] **Step 2: Write the failing health test**

Create `tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from ainovel.app import create_app


def test_health_returns_ready() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
```

- [ ] **Step 3: Run the test and confirm the expected failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_health.py -v
```

Expected: collection fails because `ainovel.app` does not exist.

- [ ] **Step 4: Implement the application factory**

Create `src/ainovel/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AINOVEL_", env_file=".env")

    app_name: str = "AI Novel Studio"
    database_url: str = "sqlite+pysqlite:///./ainovel.db"
```

Create `src/ainovel/app.py`:

```python
from fastapi import FastAPI

from ainovel.config import Settings


def create_app(database_url: str | None = None) -> FastAPI:
    settings = Settings(database_url=database_url) if database_url else Settings()
    app = FastAPI(title=settings.app_name)
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    return app


app = create_app()
```

Create an empty `src/ainovel/__init__.py`.

- [ ] **Step 5: Run the test and commit**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_health.py -v
git add pyproject.toml .gitignore src tests/test_health.py
git commit -m "chore: scaffold local novel application"
```

Expected: one test passes and the commit succeeds.

---

### Task 2: SQLite Session Boundary and Initial Migration

**Files:**
- Create: `src/ainovel/db.py`
- Create: `src/ainovel/models/__init__.py`
- Create: `src/ainovel/models/base.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `tests/conftest.py`
- Modify: `src/ainovel/app.py`

**Interfaces:**
- Produces: `create_engine_for_url(database_url: str) -> Engine`
- Produces: `create_session_factory(engine: Engine) -> sessionmaker[Session]`
- Produces: `get_session(request: Request) -> Iterator[Session]`

- [ ] **Step 1: Write a failing isolated-database test fixture**

Create `tests/conftest.py` with fixtures that construct a file-backed temporary SQLite database so separate TestClient threads share one database:

```python
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ainovel.app import create_app


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def client(database_url: str) -> Iterator[TestClient]:
    with TestClient(create_app(database_url)) as test_client:
        yield test_client
```

Add to `tests/test_health.py`:

```python
def test_app_exposes_session_factory(client: TestClient) -> None:
    assert client.app.state.session_factory is not None
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_health.py::test_app_exposes_session_factory -v
```

Expected: FAIL because `session_factory` is not attached to application state.

- [ ] **Step 3: Implement database creation and lifecycle**

Create `src/ainovel/models/base.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
```

Create `src/ainovel/db.py`:

```python
from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def create_engine_for_url(database_url: str) -> Engine:
    engine = create_engine(database_url, future=True)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_sqlite_constraints(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session
```

Update `create_app` to create the engine and session factory, create tables during the test-phase startup using `Base.metadata.create_all(engine)`, and dispose the engine during shutdown. Import `ainovel.models` before `create_all` so all tables register.

- [ ] **Step 4: Add Alembic configuration**

Configure `alembic/env.py` to import `Base.metadata` and read `AINOVEL_DATABASE_URL`, falling back to `sqlite+pysqlite:///./ainovel.db`. Confirm the migration environment loads before schema revisions exist:

```powershell
.\.venv\Scripts\alembic current
```

Expected: Alembic starts without an import or configuration error and reports no current revision.

- [ ] **Step 5: Run tests and commit the database boundary**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_health.py -v
git add src/ainovel/db.py src/ainovel/models src/ainovel/app.py tests/conftest.py tests/test_health.py alembic.ini alembic
git commit -m "feat: add isolated sqlite session boundary"
```

Expected: health tests pass and no database file appears under `tests/`.

---

### Task 3: Novel Project and Constitution Versions

**Files:**
- Create: `src/ainovel/models/project.py`
- Create: `src/ainovel/services/__init__.py`
- Create: `src/ainovel/services/projects.py`
- Create: `tests/test_projects.py`
- Modify: `src/ainovel/models/__init__.py`

**Interfaces:**
- Produces: `ProjectService.create(title: str, target_chars_min: int, target_chars_max: int) -> NovelProject`
- Produces: `ProjectService.get(project_id: str) -> NovelProject`
- Produces: `ProjectService.add_constitution(project_id: UUID, content: dict[str, object], author_approved: bool) -> ConstitutionVersion`

- [ ] **Step 1: Write failing project service tests**

Create `tests/test_projects.py`:

```python
from ainovel.services.projects import ProjectService


def test_create_project_uses_confirmed_length_range(session) -> None:
    project = ProjectService(session).create("星海问道", 2_000_000, 5_000_000)

    assert project.title == "星海问道"
    assert project.target_chars_min == 2_000_000
    assert project.target_chars_max == 5_000_000
    assert project.official_outline_version_id is None


def test_project_rejects_reversed_length_range(session) -> None:
    service = ProjectService(session)

    with pytest.raises(ValueError, match="minimum target must not exceed maximum"):
        service.create("错误项目", 5_000_000, 2_000_000)
```

Extend `tests/conftest.py` with a `session` fixture that yields a session from `client.app.state.session_factory` and rolls it back after each test.

Use this exact fixture:

```python
@pytest.fixture
def session(client: TestClient):
    with client.app.state.session_factory() as db_session:
        yield db_session
        db_session.rollback()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_projects.py -v
```

Expected: collection fails because `ProjectService` does not exist.

- [ ] **Step 3: Implement project models and service**

Define `NovelProject` with UUID string primary key, title, target range, current official outline version ID, current constitution version ID, and timestamps. Define `ConstitutionVersion` with project ID, monotonically increasing version number, JSON content, approval flag, and timestamp.

Implement the service boundary:

```python
class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, title: str, target_chars_min: int, target_chars_max: int) -> NovelProject:
        normalized = title.strip()
        if not normalized:
            raise ValueError("title is required")
        if target_chars_min > target_chars_max:
            raise ValueError("minimum target must not exceed maximum")
        project = NovelProject(
            id=str(uuid4()),
            title=normalized,
            target_chars_min=target_chars_min,
            target_chars_max=target_chars_max,
        )
        self.session.add(project)
        self.session.commit()
        return project
```

`add_constitution` must allocate the next version number inside one transaction and update `current_constitution_version_id` only when `author_approved` is true.

Add a reusable project fixture to `tests/conftest.py` after `ProjectService` exists:

```python
@pytest.fixture
def project(session):
    return ProjectService(session).create("测试小说", 2_000_000, 5_000_000)
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_projects.py -v
.\.venv\Scripts\python -m pytest -q
```

Expected: all project and health tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/ainovel/models src/ainovel/services tests
git commit -m "feat: add versioned novel projects"
```

---

### Task 4: Immutable Outline Versions and Queryable Tree

**Files:**
- Create: `src/ainovel/models/outline.py`
- Create: `src/ainovel/services/outlines.py`
- Create: `tests/test_outlines.py`
- Modify: `src/ainovel/models/__init__.py`

**Interfaces:**
- Consumes: `NovelProject.id`
- Produces: `OutlineService.create_candidate(project_id: str, nodes: list[OutlineNodeInput], reason: str) -> OutlineVersion`
- Produces: `OutlineService.approve(version_id: str) -> OutlineVersion`
- Produces: `OutlineService.get_tree(version_id: str) -> list[OutlineNodeView]`
- Produces: `OutlineService.compare(from_version_id: str, to_version_id: str) -> OutlineDiff`

- [ ] **Step 1: Define input/output schemas in the failing test**

Create `tests/test_outlines.py`:

```python
from ainovel.services.outlines import OutlineNodeInput, OutlineService


def test_candidate_outline_does_not_replace_official_project_version(session, project) -> None:
    service = OutlineService(session)
    candidate = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="initial outline",
    )

    session.refresh(project)
    assert candidate.status == "candidate"
    assert project.official_outline_version_id is None


def test_approved_outline_becomes_official_without_deleting_candidate_history(session, project) -> None:
    service = OutlineService(session)
    first = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="initial outline",
    )

    approved = service.approve(first.id)

    session.refresh(project)
    assert approved.status == "official"
    assert project.official_outline_version_id == approved.id
    assert service.get_tree(approved.id)[0].title == "全书总纲"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_outlines.py -v
```

Expected: collection fails because outline models and service do not exist.

- [ ] **Step 3: Implement immutable outline storage**

Define `OutlineVersion` with project ID, integer version number, status (`candidate`, `official`, `superseded`, `rejected`), base version ID, reason and timestamp. Define `OutlineNode` with version ID, stable key, parent key, kind, title, order, status, JSON payload and author-locked flag.

`create_candidate` copies no ORM objects from the official tree. It writes a complete snapshot from `OutlineNodeInput` values and records `base_version_id` from the project. Validate unique keys, valid parent keys and exactly one root.

`approve` marks the previously official version `superseded`, marks the selected version `official`, and updates the project pointer in one transaction.

`compare` returns added, removed and changed stable node keys by comparing canonical dictionaries that exclude database IDs and timestamps.

Add the approved outline fixture used by later tasks:

```python
@pytest.fixture
def official_outline(session, project):
    service = OutlineService(session)
    candidate = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="test outline",
    )
    return service.approve(candidate.id)
```

- [ ] **Step 4: Add failure cases and run tests**

Add this parameterized test for duplicate keys, missing parents and a second root:

```python
@pytest.mark.parametrize(
    "nodes",
    [
        [
            OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0),
            OutlineNodeInput(key="book", parent_key="book", kind="volume", title="重复", order=1),
        ],
        [OutlineNodeInput(key="stage", parent_key="missing", kind="stage", title="阶段", order=0)],
        [
            OutlineNodeInput(key="root-a", parent_key=None, kind="book", title="根A", order=0),
            OutlineNodeInput(key="root-b", parent_key=None, kind="book", title="根B", order=1),
        ],
    ],
)
def test_invalid_outline_tree_is_not_persisted(session, project, nodes) -> None:
    service = OutlineService(session)
    before = service.count_versions(project.id)

    with pytest.raises(ValueError):
        service.create_candidate(project.id, nodes, reason="invalid tree")

    assert service.count_versions(project.id) == before
```

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_outlines.py -v
```

Expected: all outline tests pass and the failed writes leave row counts unchanged.

- [ ] **Step 5: Commit**

```powershell
git add src/ainovel/models/outline.py src/ainovel/models/__init__.py src/ainovel/services/outlines.py tests/test_outlines.py
git commit -m "feat: add immutable outline versions"
```

---

### Task 5: Candidate Batches, Chapters, and Visible Character Counting

**Files:**
- Create: `src/ainovel/models/batch.py`
- Create: `src/ainovel/services/counting.py`
- Create: `src/ainovel/services/batches.py`
- Create: `tests/test_batches.py`
- Modify: `src/ainovel/models/__init__.py`

**Interfaces:**
- Produces: `count_visible_characters(body: str) -> int`
- Produces: `BatchService.create(project_id: str, outline_version_id: str, planned_chapters: int) -> WritingBatch`
- Produces: `BatchService.save_candidate_chapter(batch_id: str, ordinal: int, title: str, body: str, state_delta: dict[str, object]) -> Chapter`
- Produces: `BatchService.reject(batch_id: str, reason: str) -> WritingBatch`

- [ ] **Step 1: Write failing count and batch tests**

Create `tests/test_batches.py`:

```python
import pytest

from ainovel.services.batches import BatchService
from ainovel.services.counting import count_visible_characters


def test_visible_count_excludes_whitespace_but_includes_punctuation_and_ascii() -> None:
    assert count_visible_characters(" 甲，A1\n乙。 ") == 6


@pytest.mark.parametrize("planned", [0, 6])
def test_batch_size_must_be_between_one_and_five(session, project, official_outline, planned) -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        BatchService(session).create(project.id, official_outline.id, planned)


def test_saved_candidate_chapter_does_not_become_official(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, "初临", "甲" * 4500, {"time": "+1 day"})

    assert chapter.status == "candidate"
    assert chapter.visible_char_count == 4500
    assert batch.status == "draft"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_batches.py -v
```

Expected: collection fails because batch services do not exist.

- [ ] **Step 3: Implement exact visible character counting**

Create `src/ainovel/services/counting.py`:

```python
def count_visible_characters(body: str) -> int:
    return sum(1 for character in body if not character.isspace())
```

The service receives body only, so chapter titles and reports cannot enter the count.

- [ ] **Step 4: Implement candidate batch persistence**

Define `WritingBatch` with project ID, base outline version ID, planned chapter count, status (`draft`, `ready_for_review`, `approved`, `rejected`), and timestamps. Define `Chapter` with batch ID, ordinal, title, body, visible count, status (`candidate`, `official`, `published`), state delta JSON and official chapter number.

Enforce database uniqueness on `(batch_id, ordinal)` and `(project_id, official_chapter_number)` when the number is present. `save_candidate_chapter` validates ordinal range and stores the computed visible count. `reject` changes only the batch status; it never deletes official data or the rejected candidate record.

- [ ] **Step 5: Add boundary tests, run, and commit**

Add explicit boundary tests. Saving is allowed at any length so authors can retain incomplete drafts, while review readiness rejects out-of-range bodies:

```python
@pytest.mark.parametrize("length", [4500, 6000])
def test_boundary_length_chapter_can_be_marked_ready(session, project, official_outline, length) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "边界", "甲" * length, {})

    ready = service.mark_ready(batch.id)

    assert ready.status == "ready_for_review"


@pytest.mark.parametrize("length", [4499, 6001])
def test_out_of_range_chapter_remains_saved_but_cannot_be_ready(
    session, project, official_outline, length
) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, "越界", "甲" * length, {})

    with pytest.raises(ValueError, match="4500.*6000"):
        service.mark_ready(batch.id)

    assert service.get_chapter(chapter.id).status == "candidate"
```

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_batches.py -v
git add src/ainovel/models/batch.py src/ainovel/models/__init__.py src/ainovel/services/counting.py src/ainovel/services/batches.py tests/test_batches.py
git commit -m "feat: add isolated candidate writing batches"
```

Expected: all batch and counting tests pass.

---

### Task 6: Atomic Approval, Rejection Audit, and Published Protection

**Files:**
- Create: `src/ainovel/models/audit.py`
- Modify: `src/ainovel/services/batches.py`
- Modify: `src/ainovel/models/__init__.py`
- Modify: `tests/test_batches.py`

**Interfaces:**
- Produces: `BatchService.mark_ready(batch_id: str) -> WritingBatch`
- Produces: `BatchService.approve(batch_id: str, approved_outline_version_id: str, actor: str = "author") -> WritingBatch`
- Produces: `BatchService.publish_chapter(chapter_id: str) -> Chapter`
- Produces: `BatchService.replace_candidate_body(chapter_id: str, body: str) -> Chapter`

- [ ] **Step 1: Write the failing atomicity test**

Append to `tests/test_batches.py`:

```python
def test_failed_approval_keeps_every_chapter_candidate(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    service.save_candidate_chapter(batch.id, 1, "一", "甲" * 4500, {"seq": 1})
    service.save_candidate_chapter(batch.id, 2, "二", "乙" * 4500, {"seq": 2})
    service.mark_ready(batch.id)

    @event.listens_for(session, "before_commit", once=True)
    def fail_commit(_session) -> None:
        raise RuntimeError("simulated commit failure")

    with pytest.raises(RuntimeError, match="simulated commit failure"):
        service.approve(batch.id, official_outline.id)

    session.rollback()
    session.expire_all()
    chapters = service.list_chapters(batch.id)
    assert [chapter.status for chapter in chapters] == ["candidate", "candidate"]
    assert service.get(batch.id).status != "approved"


def test_published_chapter_cannot_be_replaced(session, approved_chapter) -> None:
    service = BatchService(session)
    service.publish_chapter(approved_chapter.id)

    with pytest.raises(PermissionError, match="published chapters are frozen"):
        service.replace_candidate_body(approved_chapter.id, "新正文")
```

Import `event` from SQLAlchemy in the test module. Add this fixture to `tests/conftest.py`:

```python
@pytest.fixture
def approved_chapter(session, project, official_outline):
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "已批准章", "甲" * 4500, {})
    service.mark_ready(batch.id)
    service.approve(batch.id, official_outline.id)
    return service.list_chapters(batch.id)[0]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_batches.py -v
```

Expected: FAIL because approval and published protection are not implemented.

- [ ] **Step 3: Implement append-only audit events**

Define `AuditEvent` with project ID, entity type, entity ID, action, actor, JSON details and timestamp. Do not expose update or delete methods for this model.

Record `batch_created`, `batch_ready`, `batch_rejected`, `batch_approved` and `chapter_published`. Validation errors are returned to the caller and application logs; they do not create a second database transaction.

- [ ] **Step 4: Implement transactional approval**

Inside `approve`:

1. Load the batch and all chapters using one session.
2. Require `ready_for_review` status and exactly `planned_chapters` distinct ordinals.
3. Require every body count in the inclusive 4500—6000 range.
4. Require the approved outline version to belong to the same project.
5. Allocate consecutive official chapter numbers after the current maximum.
6. Mark all chapters `official` and the batch `approved`.
7. Point the project to the approved outline version.
8. Insert one `batch_approved` audit record containing chapter IDs and state deltas.
9. Commit once after all mutations are staged.

Catch validation errors before mutation when possible. On database errors, call `session.rollback()` and re-raise. `replace_candidate_body` permits only `candidate` chapters; `publish_chapter` permits only `official` chapters.

- [ ] **Step 5: Run rollback, protection, and full tests**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_batches.py -v
.\.venv\Scripts\python -m pytest -q --cov=ainovel --cov-report=term-missing
```

Expected: approval failure leaves no partial official chapters; published replacement raises `PermissionError`; the full suite passes.

- [ ] **Step 6: Commit**

```powershell
git add src/ainovel/models/audit.py src/ainovel/models/__init__.py src/ainovel/services/batches.py tests/test_batches.py
git commit -m "feat: approve writing batches atomically"
```

---

### Task 7: Minimal Local Author Dashboard

**Files:**
- Create: `src/ainovel/web/__init__.py`
- Create: `src/ainovel/web/routes.py`
- Create: `src/ainovel/web/templates/base.html`
- Create: `src/ainovel/web/templates/index.html`
- Create: `src/ainovel/web/templates/project.html`
- Create: `src/ainovel/static/app.js`
- Create: `tests/test_web.py`
- Modify: `src/ainovel/app.py`

**Interfaces:**
- Consumes: `ProjectService`, `OutlineService`, `BatchService`
- Produces: `GET /`, `POST /projects`, `GET /projects/{project_id}`
- Produces: `POST /projects/{project_id}/batches`, `POST /batches/{batch_id}/ready`, `POST /batches/{batch_id}/approve`, `POST /batches/{batch_id}/reject`

- [ ] **Step 1: Write failing Web workflow tests**

Create `tests/test_web.py`:

```python
def test_home_page_creates_and_opens_project(client) -> None:
    response = client.post(
        "/projects",
        data={"title": "万界行舟", "target_chars_min": "2000000", "target_chars_max": "5000000"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    project_page = client.get(response.headers["location"])
    assert project_page.status_code == 200
    assert "万界行舟" in project_page.text
    assert "候选批次" in project_page.text


def test_invalid_project_form_returns_422_without_creating_project(client) -> None:
    response = client.post(
        "/projects",
        data={"title": "", "target_chars_min": "5000000", "target_chars_max": "2000000"},
    )

    assert response.status_code == 422
    assert "项目名称不能为空" in response.text
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_web.py -v
```

Expected: both tests fail because the routes do not exist.

- [ ] **Step 3: Implement server-rendered routes**

Mount `StaticFiles(directory="src/ainovel/static")` at `/static`. Configure Jinja templates. Route handlers use `Depends(get_session)` and services; they do not query ORM models directly.

The index page contains the project form and project list. The project page displays:

- current constitution and outline version numbers;
- official chapter count and visible character total;
- candidate batches with status;
- approval/rejection forms;
- audit event list;
- a clear notice that model generation is introduced in phase two.

Validation failures return the same form with HTTP 422 and Chinese error text. Successful mutations use HTTP 303 redirects to prevent form resubmission.

- [ ] **Step 4: Add local UI shell and accessibility checks**

`base.html` must declare `lang="zh-CN"`, include UTF-8, use visible `<label>` elements for every input, and render status messages in an element with `role="status"`. Keep `app.js` dependency-free; it only asks for confirmation before approve/reject form submission.

Add assertions for these markers in `tests/test_web.py`.

- [ ] **Step 5: Run tests and manual smoke test**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_web.py -v
.\.venv\Scripts\uvicorn ainovel.app:app --reload
```

Open `http://127.0.0.1:8000`, create one project, open its page, and confirm no JavaScript console error. Stop the server after the check.

- [ ] **Step 6: Commit**

```powershell
git add src/ainovel/app.py src/ainovel/web src/ainovel/static tests/test_web.py
git commit -m "feat: add local author dashboard"
```

---

### Task 8: Complete Migration, Acceptance Test, and Operator Documentation

**Files:**
- Modify: `alembic/versions/0001_foundation.py`
- Create: `tests/test_foundation_acceptance.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all foundation services and routes
- Produces: reproducible database creation and one end-to-end foundation acceptance test

- [ ] **Step 1: Write the failing acceptance test**

Create `tests/test_foundation_acceptance.py` that executes this sequence through services:

```python
def test_author_can_approve_one_batch_without_candidate_leakage(session) -> None:
    project = ProjectService(session).create("山海铸天", 2_000_000, 5_000_000)
    outline_service = OutlineService(session)
    outline = outline_service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="author approved initial outline",
    )
    outline_service.approve(outline.id)

    batch_service = BatchService(session)
    batch = batch_service.create(project.id, outline.id, 1)
    candidate = batch_service.save_candidate_chapter(
        batch.id, 1, "山门之外", "山" * 4500, {"timeline_days": 1}
    )
    assert candidate.status == "candidate"

    batch_service.mark_ready(batch.id)
    batch_service.approve(batch.id, outline.id)

    session.expire_all()
    official = batch_service.list_chapters(batch.id)[0]
    assert official.status == "official"
    assert official.official_chapter_number == 1
    assert batch_service.list_audit_events(project.id)[-1].action == "batch_approved"
```

- [ ] **Step 2: Run acceptance test before completing the migration**

Run:

```powershell
.\.venv\Scripts\python -m pytest tests/test_foundation_acceptance.py -v
```

Expected: the test exposes any missing fixture or service integration before release.

- [ ] **Step 3: Generate and inspect the complete initial migration**

Delete no existing user database. On the empty development database only, generate the revision from current metadata or write it explicitly so it creates every foundation table, foreign key, unique constraint and index. Verify both directions against a temporary path:

```powershell
$env:AINOVEL_DATABASE_URL = "sqlite+pysqlite:///./migration-check.db"
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\alembic downgrade base
.\.venv\Scripts\alembic upgrade head
Remove-Item -LiteralPath '.\migration-check.db'
Remove-Item Env:\AINOVEL_DATABASE_URL
```

Before the `Remove-Item`, resolve the target with `Resolve-Path '.\migration-check.db'` and confirm it is exactly `D:\ainovel\migration-check.db`.

- [ ] **Step 4: Document exact local commands**

Create `README.md` containing the following sections and commands:

```text
# AI Novel Studio

## Local setup

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -e ".[test]"
    .\.venv\Scripts\alembic upgrade head
    .\.venv\Scripts\uvicorn ainovel.app:app --reload

Open http://127.0.0.1:8000.

## Tests

    .\.venv\Scripts\python -m pytest -q

Phase one stores projects, outlines and candidate batches locally. It does not call a model API.
```

Render the indented command blocks in the final README as fenced PowerShell blocks:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\alembic upgrade head
.\.venv\Scripts\uvicorn ainovel.app:app --reload
```

- [ ] **Step 5: Run the complete verification suite**

Run:

```powershell
.\.venv\Scripts\python -m pytest -q --cov=ainovel --cov-report=term-missing
.\.venv\Scripts\python -m compileall -q src tests
git status --short
```

Expected: all tests pass, compilation succeeds, and Git shows only the intended plan or documentation changes not yet committed.

- [ ] **Step 6: Commit the phase-one release**

```powershell
git add alembic README.md tests/test_foundation_acceptance.py
git commit -m "test: verify foundation approval workflow"
```

Expected: a clean foundation implementation with no real model dependency.

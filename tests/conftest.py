from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ainovel.app import create_app

from ainovel.models import Base

@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def client(database_url: str) -> Iterator[TestClient]:
    app = create_app(database_url)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session(client: TestClient):
    with client.app.state.session_factory() as db_session:
        yield db_session
        db_session.rollback()


@pytest.fixture
def project(session):
    from ainovel.services.projects import ProjectService
    return ProjectService(session).create("测试小说", 2_000_000, 5_000_000)


@pytest.fixture
def official_outline(session, project):
    from ainovel.services.outlines import OutlineNodeInput, OutlineService

    service = OutlineService(session)
    candidate = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="test outline",
    )
    return service.approve(candidate.id)


@pytest.fixture
def approved_chapter(session, project, official_outline):
    from ainovel.services.batches import BatchService

    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "已批准章", "甲" * 4500, {})
    service.mark_ready(batch.id)
    service.approve(batch.id, official_outline.id)
    return service.list_chapters(batch.id)[0]
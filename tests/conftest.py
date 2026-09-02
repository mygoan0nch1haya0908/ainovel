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

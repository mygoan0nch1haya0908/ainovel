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

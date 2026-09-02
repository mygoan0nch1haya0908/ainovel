from fastapi.testclient import TestClient

from ainovel.app import create_app
from ainovel.models import Base


def test_health_returns_ready() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_app_exposes_session_factory(client: TestClient) -> None:
    assert client.app.state.session_factory is not None

def test_default_app_does_not_create_schema(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("production app must not create schema")

    monkeypatch.setattr(Base.metadata, "create_all", fail_if_called)

    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200

def test_explicit_url_app_does_not_create_schema(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("application factories must not create schemas")

    monkeypatch.setattr(Base.metadata, "create_all", fail_if_called)

    with TestClient(create_app("sqlite+pysqlite:///explicit.db")) as client:
        assert client.get("/health").status_code == 200

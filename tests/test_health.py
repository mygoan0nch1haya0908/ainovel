from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from ainovel.app import create_app
from ainovel.models import Base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_explicit_url_app_does_not_create_schema(monkeypatch, tmp_path: Path) -> None:
    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("application factories must not create schemas")

    monkeypatch.setattr(Base.metadata, "create_all", fail_if_called)

    database_url = f"sqlite+pysqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    with TestClient(create_app(database_url)) as client:
        assert client.get("/health").status_code == 200


def test_ready_returns_503_when_schema_is_absent(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'absent.db').as_posix()}"

    with TestClient(create_app(database_url)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "database schema is absent or unavailable",
    }


def test_ready_returns_503_when_schema_is_outdated(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'outdated.db').as_posix()}"
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO alembic_version (version_num) VALUES ('old_revision')"
            )
    finally:
        engine.dispose()

    with TestClient(create_app(database_url)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "database schema is not at Alembic head",
    }


def test_ready_returns_200_at_alembic_head(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'ready.db').as_posix()}"
    monkeypatch.setenv("AINOVEL_DATABASE_URL", database_url)
    command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

    with TestClient(create_app(database_url)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

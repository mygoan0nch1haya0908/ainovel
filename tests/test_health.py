from fastapi.testclient import TestClient

from ainovel.app import create_app


def test_health_returns_ready() -> None:
    client = TestClient(create_app("sqlite+pysqlite:///:memory:"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

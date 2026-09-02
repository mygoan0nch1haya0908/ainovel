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

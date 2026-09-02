from contextlib import asynccontextmanager

from fastapi import FastAPI

from ainovel.config import Settings
from ainovel.db import create_engine_for_url, create_session_factory


def create_app(database_url: str | None = None) -> FastAPI:
    settings = Settings(database_url=database_url) if database_url else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine.dispose()

    engine = create_engine_for_url(settings.database_url)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    return app


app = create_app()

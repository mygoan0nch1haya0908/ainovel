from contextlib import asynccontextmanager
from pathlib import Path
from secrets import token_urlsafe

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ainovel.config import Settings
from ainovel.db import create_engine_for_url, create_session_factory, database_readiness


def create_app(database_url: str | None = None) -> FastAPI:
    settings = Settings(database_url=database_url) if database_url else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine.dispose()

    engine = create_engine_for_url(settings.database_url)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    session_secret = settings.session_secret or token_urlsafe(32)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.csrf_signer = URLSafeSerializer(session_secret, salt="ainovel-csrf")
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        same_site="strict",
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
        name="static",
    )

    from ainovel.web.routes import router as web_router

    app.include_router(web_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/ready")
    def ready() -> object:
        is_ready, detail = database_readiness(engine)
        if not is_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": detail},
            )
        return {"status": "ready"}

    return app


app = create_app()

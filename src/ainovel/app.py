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

from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from secrets import token_urlsafe
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from itsdangerous import URLSafeSerializer
from openai import OpenAI
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ainovel.agents.runner import AgentRunner
from ainovel.config import Settings
from ainovel.db import create_engine_for_url, create_session_factory, database_readiness
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.ollama import OllamaProvider
from ainovel.providers.openai import OpenAIProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.workflows.orchestrator import WorkflowOrchestrator


PROVIDER_CONTEXT_WINDOW_CEILING = 16_000
PROVIDER_OUTPUT_TOKEN_CEILING = 4_000


def _loopback_provider_url(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or host is None:
        raise ValueError("Ollama base URL must be an HTTP loopback URL")
    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if not is_loopback:
        raise ValueError("Ollama base URL must use a loopback host")
    return value.rstrip("/")


def _default_provider_registry(settings: Settings) -> ProviderRegistry:
    ollama_base_url = _loopback_provider_url(settings.ollama_base_url)
    api_key = (
        settings.openai_api_key.get_secret_value()
        if settings.openai_api_key is not None
        else None
    )
    allow_openai = bool(settings.allow_real_openai and api_key)

    def ollama_provider() -> OllamaProvider:
        return OllamaProvider(
            httpx.Client(timeout=settings.provider_timeout_seconds),
            ollama_base_url,
            context_window_limit=PROVIDER_CONTEXT_WINDOW_CEILING,
            max_output_tokens_limit=PROVIDER_OUTPUT_TOKEN_CEILING,
        )

    def openai_provider() -> OpenAIProvider:
        client_options: dict[str, object] = {
            "api_key": api_key or "not-configured",
            "timeout": settings.provider_timeout_seconds,
        }
        if settings.openai_base_url:
            client_options["base_url"] = settings.openai_base_url
        return OpenAIProvider(
            OpenAI(**client_options),
            allow_real_calls=allow_openai,
            context_window_limit=PROVIDER_CONTEXT_WINDOW_CEILING,
            max_output_tokens_limit=PROVIDER_OUTPUT_TOKEN_CEILING,
        )

    return ProviderRegistry(
        {
            "fake": DemoFakeProvider,
            "ollama": ollama_provider,
            "openai": openai_provider,
        }
    )


def create_app(
    database_url: str | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> FastAPI:
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
    app.state.provider_registry = provider_registry or _default_provider_registry(settings)
    orchestrator = WorkflowOrchestrator(
        app.state.session_factory,
        app.state.provider_registry,
        AgentRunner(),
    )
    app.state.orchestrator_factory = lambda: orchestrator
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
    from ainovel.web.workflow_routes import router as workflow_router

    app.include_router(web_router)
    app.include_router(workflow_router)

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

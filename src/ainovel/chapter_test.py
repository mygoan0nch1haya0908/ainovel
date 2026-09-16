from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import FastAPI

from ainovel.app import _default_provider_registry, create_app
from ainovel.config import Settings
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.workflows import WorkflowBudgets


CHAPTER_TEST_CONTEXT_WINDOW = 32_000
CHAPTER_TEST_OUTPUT_CEILING = 12_000
CHAPTER_TEST_REQUEST_TIMEOUT_SECONDS = 180.0
CHAPTER_TEST_DATABASE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".superpowers"
    / "runtime"
    / "chapter-test"
    / "chapter-test.db"
)
CHAPTER_TEST_BUDGETS = WorkflowBudgets(
    planner_input=16_000,
    planner_output=4_000,
    writer_input=32_000,
    writer_output=12_000,
    summarizer_input=16_000,
    summarizer_output=4_000,
    reviewer_input=32_000,
    reviewer_output=4_000,
)


def create_chapter_test_app(
    database_url: str | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> FastAPI:
    """Build the loopback-only isolated stage-planning and chapter test app."""
    if database_url is None:
        CHAPTER_TEST_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        resolved_database_url = (
            f"sqlite+pysqlite:///{CHAPTER_TEST_DATABASE_PATH.as_posix()}"
        )
    else:
        resolved_database_url = database_url

    settings = Settings(database_url=resolved_database_url)
    registry = provider_registry or _default_provider_registry(
        settings,
        qwen_context_window_ceiling=CHAPTER_TEST_CONTEXT_WINDOW,
        qwen_output_token_ceiling=CHAPTER_TEST_OUTPUT_CEILING,
    )
    app = create_app(
        resolved_database_url,
        provider_registry=registry,
        orchestrator_request_timeout_seconds=CHAPTER_TEST_REQUEST_TIMEOUT_SECONDS,
    )
    app.title = "AI Novel Studio · Author Planning TEST"
    app.state.chapter_test_mode = True
    app.state.chapter_test_database_path = str(
        Path(app.state.engine.url.database).resolve()
    )
    app.state.workflow_budgets = CHAPTER_TEST_BUDGETS
    app.state.required_workflow_chapters = 1
    app.state.chapter_test_consumed_submission_tokens = set()
    app.state.chapter_test_submission_lock = Lock()

    from ainovel.web.chapter_test_routes import router

    app.include_router(router)
    return app

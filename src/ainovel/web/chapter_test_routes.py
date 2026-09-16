from __future__ import annotations

import logging
from secrets import compare_digest, token_urlsafe

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.db import get_session
from ainovel.models.outline import OutlineNode, OutlineVersion
from ainovel.models.project import ConstitutionVersion, NovelProject
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService
from ainovel.services.stages import StageService
from ainovel.services.workflows import WorkflowBudgets, WorkflowService
from ainovel.web.routes import templates
from ainovel.web.security import csrf_token, require_csrf


router = APIRouter()
logger = logging.getLogger(__name__)
SUBMISSION_SESSION_KEY = "chapter_test_submission_token"
FIELD_LIMITS = {
    "project_title": 120,
    "setting_style": 8_000,
    "provisional_ending": 4_000,
    "chapter_outline": 12_000,
    "chapter_title": 200,
    "chapter_goal": 1_000,
    "chapter_hook": 1_000,
    "stage_architecture": 12_000,
    "model_name": 128,
}
FIELD_LABELS = {
    "project_title": "项目名称",
    "setting_style": "设定与文风",
    "provisional_ending": "暂定结局",
    "chapter_outline": "章节提纲",
    "chapter_title": "章节标题",
    "chapter_goal": "章节目标",
    "chapter_hook": "章末钩子",
    "stage_architecture": "剧情阶段总体架构",
    "model_name": "模型名称",
}


class ChapterTestSetupFailure(RuntimeError):
    def __init__(self, stage: str, exception_type: str) -> None:
        self.stage = stage
        self.exception_type = exception_type
        super().__init__("chapter test setup failed")


def _empty_form() -> dict[str, str]:
    return {
        "project_title": "",
        "setting_style": "",
        "provisional_ending": "",
        "chapter_outline": "",
        "chapter_title": "",
        "chapter_goal": "",
        "chapter_hook": "",
        "stage_architecture": "",
        "setup_mode": "single_chapter",
        "model_name": "qwen-flash",
        "repair_mode": "",
        "author_confirm": "",
    }


def _new_submission_token(request: Request) -> str:
    token = token_urlsafe(32)
    request.session[SUBMISSION_SESSION_KEY] = token
    return token


def _confirmed_input(
    session: Session, project_id: str | None
) -> dict[str, object] | None:
    if not project_id:
        return None
    project = session.get(NovelProject, project_id)
    if project is None:
        return None
    constitution = (
        session.get(ConstitutionVersion, project.current_constitution_version_id)
        if project.current_constitution_version_id is not None
        else None
    )
    outline = (
        session.get(OutlineVersion, project.official_outline_version_id)
        if project.official_outline_version_id is not None
        else None
    )
    nodes = []
    if outline is not None:
        nodes = session.scalars(
            select(OutlineNode)
            .where(OutlineNode.outline_version_id == outline.id)
            .order_by(OutlineNode.order, OutlineNode.stable_key)
        ).all()
    root = next((node for node in nodes if node.stable_key == "book"), None)
    ending = next(
        (node for node in nodes if node.kind == "provisional_ending"), None
    )
    chapter = next((node for node in nodes if node.stable_key == "chapter-1"), None)
    return {
        "project": project,
        "setting_style": (
            constitution.content.get("setting_style", "")
            if constitution is not None
            else ""
        ),
        "provisional_ending": ending.payload.get("text", "") if ending else "",
        "chapter_outline": (
            chapter.payload.get("chapter_outline", "")
            if chapter is not None
            else root.payload.get("chapter_outline", "") if root is not None else ""
        ),
        "chapter_title": chapter.title if chapter is not None else "",
        "chapter_goal": chapter.payload.get("goal", "") if chapter else "",
        "chapter_hook": chapter.payload.get("ending_hook", "") if chapter else "",
        "stage_architecture": (
            root.payload.get("stage_architecture", "") if root is not None else ""
        ),
    }


def _render(
    request: Request,
    session: Session,
    values: dict[str, str],
    submission_token: str,
    *,
    error: str | None = None,
    status_code: int = 200,
    project_id: str | None = None,
    reuse_notice: str | None = None,
) -> object:
    return templates.TemplateResponse(
        request,
        "chapter_test.html",
        {
            "request": request,
            "csrf_token": csrf_token(request),
            "submission_token": submission_token,
            "chapter_form": values,
            "error": error,
            "projects": ProjectService(session).list_projects(),
            "confirmed_input": _confirmed_input(session, project_id),
            "reuse_notice": reuse_notice,
            "database_path": request.app.state.chapter_test_database_path,
            "field_limits": FIELD_LIMITS,
        },
        status_code=status_code,
    )


def _validation_error(values: dict[str, str]) -> str | None:
    if values.get("setup_mode") not in {"single_chapter", "stage"}:
        return "请选择独立单章测试或剧情阶段总体架构"
    required = (
        ("project_title", "项目名称不能为空"),
        ("setting_style", "设定与文风不能为空"),
        ("provisional_ending", "暂定结局不能为空"),
        ("model_name", "模型名称不能为空"),
    )
    for field, message in required:
        if not values[field].strip():
            return message
    if values["setup_mode"] == "single_chapter" and not values["chapter_outline"].strip():
        return "章节提纲不能为空"
    if values["setup_mode"] == "stage" and not values["stage_architecture"].strip():
        return "剧情阶段总体架构不能为空"
    for field, maximum in FIELD_LIMITS.items():
        if len(values[field]) > maximum:
            return f"{FIELD_LABELS[field]}不能超过 {maximum} 个字符"
    if values["author_confirm"] != "yes":
        return (
            "请勾选作者确认，确认设定、暂定结局与剧情阶段总体架构"
            if values["setup_mode"] == "stage"
            else "请勾选作者确认，确认设定、暂定结局与章节提纲"
        )
    if values.get("repair_mode", "") not in {"", "yes"}:
        return "短稿修补模式选项无效"
    return None


def _consume_submission_token(request: Request, submitted: str) -> bool:
    expected = request.session.get(SUBMISSION_SESSION_KEY)
    if not isinstance(expected, str) or not submitted:
        return False
    lock = request.app.state.chapter_test_submission_lock
    consumed = request.app.state.chapter_test_consumed_submission_tokens
    with lock:
        if submitted in consumed or not compare_digest(submitted, expected):
            return False
        consumed.add(submitted)
        request.session.pop(SUBMISSION_SESSION_KEY, None)
        return True


def _outline_nodes(values: dict[str, str]) -> list[OutlineNodeInput]:
    if values.get("setup_mode") == "stage":
        return [
            OutlineNodeInput(
                key="book",
                parent_key=None,
                kind="book",
                title="阶段规划总纲",
                order=0,
                payload={"stage_architecture": values["stage_architecture"].strip()},
            ),
            OutlineNodeInput(
                key="provisional-ending",
                parent_key="book",
                kind="provisional_ending",
                title="暂定结局",
                order=1,
                payload={"text": values["provisional_ending"].strip()},
            ),
        ]
    chapter_payload = {
        "chapter_outline": values["chapter_outline"].strip(),
    }
    if values["chapter_goal"].strip():
        chapter_payload["goal"] = values["chapter_goal"].strip()
    if values["chapter_hook"].strip():
        chapter_payload["ending_hook"] = values["chapter_hook"].strip()
    chapter_title = values["chapter_title"].strip() or "第一章"
    return [
        OutlineNodeInput(
            key="book",
            parent_key=None,
            kind="book",
            title="单章测试总纲",
            order=0,
            payload={"chapter_outline": values["chapter_outline"].strip()},
        ),
        OutlineNodeInput(
            key="provisional-ending",
            parent_key="book",
            kind="provisional_ending",
            title="暂定结局",
            order=1,
            payload={"text": values["provisional_ending"].strip()},
        ),
        OutlineNodeInput(
            key="chapter-1",
            parent_key="book",
            kind="current_stage_goal",
            title=chapter_title,
            order=2,
            payload=chapter_payload,
        ),
    ]


def _create_workflow_atomically(
    request: Request,
    values: dict[str, str],
) -> tuple[str, str]:
    connection = None
    transaction = None
    session = None
    stage = "connect"
    primary_failure = None
    result = None
    try:
        connection = request.app.state.engine.connect()
        stage = "transaction_begin"
        transaction = connection.begin()
        if connection.dialect.name == "sqlite":
            # Python's legacy sqlite transaction mode does not emit BEGIN for a
            # SELECT.  An explicit outer transaction keeps service SAVEPOINT
            # commits inside this unit so a late failure can roll it all back.
            stage = "sqlite_begin"
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        stage = "session_create"
        session = Session(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
            join_transaction_mode="create_savepoint",
        )
        stage = "project_create"
        project = ProjectService(session).create(
            values["project_title"], 2_000_000, 5_000_000
        )
        stage = "constitution_create"
        ProjectService(session).add_constitution(
            project.id,
            {"setting_style": values["setting_style"].strip(), "test_entry": True},
            author_approved=True,
        )
        stage = "outline_create"
        candidate = OutlineService(session).create_candidate(
            project.id,
            _outline_nodes(values),
            reason=("作者确认的剧情阶段总体架构输入" if values.get("setup_mode") == "stage"
                    else "作者确认的单章测试输入"),
        )
        stage = "outline_approve"
        OutlineService(session).approve(candidate.id)
        if values.get("setup_mode") == "stage":
            stage = "stage_create"
            story_stage = StageService(session).create(
                project.id, values["stage_architecture"].strip(), "author"
            )
            project_id, target_id = project.id, story_stage.id
        else:
            stage = "budget_validate"
            budgets = request.app.state.workflow_budgets
            if not isinstance(budgets, WorkflowBudgets):
                raise RuntimeError("chapter test workflow budgets are invalid")
            stage = "workflow_start"
            start_args = (project.id, "qwen", values["model_name"], 1, budgets)
            if values.get("repair_mode") == "yes":
                workflow = WorkflowService(session).start(
                    *start_args, generation_version=2
                )
            else:
                workflow = WorkflowService(session).start(*start_args)
            project_id, target_id = project.id, workflow.id
        stage = "session_close"
        session.close()
        session = None
        stage = "transaction_commit"
        transaction.commit()
        result = (project_id, target_id)
    except Exception as error:
        primary_failure = ChapterTestSetupFailure(stage, type(error).__name__)

    cleanup_failure = None
    if session is not None:
        try:
            session.close()
        except Exception as error:
            cleanup_failure = ChapterTestSetupFailure(
                "session_cleanup", type(error).__name__
            )
    if transaction is not None:
        try:
            if transaction.is_active:
                transaction.rollback()
        except Exception as error:
            if cleanup_failure is None:
                cleanup_failure = ChapterTestSetupFailure(
                    "transaction_rollback", type(error).__name__
                )
    if connection is not None:
        try:
            connection.close()
        except Exception as error:
            if cleanup_failure is None:
                cleanup_failure = ChapterTestSetupFailure(
                    "connection_cleanup", type(error).__name__
                )

    if primary_failure is not None:
        raise primary_failure from None
    if result is None:
        if cleanup_failure is not None:
            raise cleanup_failure from None
        raise ChapterTestSetupFailure("result_validation", "RuntimeError") from None
    if cleanup_failure is not None:
        # The outer commit succeeded; cleanup cannot undo the persisted setup.
        logger.warning(
            "chapter_test_setup_cleanup_failed event_id=%s stage=%s exception_type=%s",
            token_urlsafe(12),
            cleanup_failure.stage,
            cleanup_failure.exception_type,
        )
    return result


def _setup_failure_response(
    request: Request,
    session: Session,
    error: ChapterTestSetupFailure,
) -> object:
    event_id = token_urlsafe(12)
    logger.error(
        "chapter_test_setup_failed event_id=%s stage=%s exception_type=%s",
        event_id,
        error.stage,
        error.exception_type,
    )
    return _render(
        request,
        session,
        _empty_form(),
        _new_submission_token(request),
        error=f"测试项目创建失败；未保留半完成设置，请重试。参考编号：{event_id}",
        status_code=500,
    )


@router.get("/chapter-test")
def chapter_test_page(
    request: Request,
    project_id: str | None = None,
    session: Session = Depends(get_session),
) -> object:
    return _render(
        request,
        session,
        _empty_form(),
        _new_submission_token(request),
        project_id=project_id,
    )


@router.post("/chapter-test/reuse/{project_id}")
def reuse_chapter_test_input(
    project_id: str,
    request: Request,
    reuse_confirm: str = Form(""),
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    confirmed = _confirmed_input(session, project_id)
    if confirmed is None:
        return _render(
            request,
            session,
            _empty_form(),
            _new_submission_token(request),
            error="找不到可复用的已确认输入",
            status_code=404,
        )
    if reuse_confirm != "yes":
        return _render(
            request,
            session,
            _empty_form(),
            _new_submission_token(request),
            error="请确认只把旧输入预填到新任务表单",
            status_code=422,
            project_id=project_id,
        )
    project = confirmed["project"]
    values = {
        "project_title": project.title,
        "setting_style": str(confirmed["setting_style"]),
        "provisional_ending": str(confirmed["provisional_ending"]),
        "chapter_outline": str(confirmed["chapter_outline"]),
        "chapter_title": str(confirmed["chapter_title"]),
        "chapter_goal": str(confirmed["chapter_goal"]),
        "chapter_hook": str(confirmed["chapter_hook"]),
        "stage_architecture": str(confirmed["stage_architecture"]),
        "setup_mode": (
            "stage" if confirmed["stage_architecture"] else "single_chapter"
        ),
        "model_name": "qwen-flash",
        "repair_mode": "",
        "author_confirm": "",
    }
    return _render(
        request,
        session,
        values,
        _new_submission_token(request),
        project_id=project_id,
        reuse_notice="这是新任务的预填表单；原任务、提示词快照、修补次数和状态均未修改。",
    )


@router.post("/chapter-test")
def create_chapter_test(
    request: Request,
    project_title: str = Form(""),
    setting_style: str = Form(""),
    provisional_ending: str = Form(""),
    chapter_outline: str = Form(""),
    chapter_title: str = Form(""),
    chapter_goal: str = Form(""),
    chapter_hook: str = Form(""),
    model_name: str = Form("qwen-flash"),
    setup_mode: str = Form("single_chapter"),
    stage_architecture: str = Form(""),
    repair_mode: str = Form(""),
    author_confirm: str = Form(""),
    submission_token: str = Form(""),
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    values = {
        "project_title": project_title,
        "setting_style": setting_style,
        "provisional_ending": provisional_ending,
        "chapter_outline": chapter_outline,
        "chapter_title": chapter_title,
        "chapter_goal": chapter_goal,
        "chapter_hook": chapter_hook,
        "model_name": model_name,
        "setup_mode": setup_mode,
        "stage_architecture": stage_architecture,
        "repair_mode": repair_mode,
        "author_confirm": author_confirm,
    }
    error = _validation_error(values)
    if error is not None:
        current_token = request.session.get(SUBMISSION_SESSION_KEY)
        if not isinstance(current_token, str):
            current_token = _new_submission_token(request)
        return _render(
            request,
            session,
            values,
            current_token,
            error=error,
            status_code=422,
        )
    if not _consume_submission_token(request, submission_token):
        return _render(
            request,
            session,
            values,
            _new_submission_token(request),
            error="此表单已提交或已过期，请检查后重新确认",
            status_code=409,
        )
    try:
        _project_id, target_id = _create_workflow_atomically(request, values)
    except ChapterTestSetupFailure as error:
        return _setup_failure_response(
            request,
            session,
            error=error,
        )
    destination = "stages" if setup_mode == "stage" else "workflows"
    return RedirectResponse(f"/{destination}/{target_id}", status_code=303)

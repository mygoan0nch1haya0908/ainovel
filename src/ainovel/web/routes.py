from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import select

from ainovel.db import get_session
from ainovel.models.workflow import GenerationWorkflow
from ainovel.models.batch import Chapter
from ainovel.web.presentation import WORKFLOW_LABELS
from ainovel.services.batches import BatchService
from ainovel.services.outlines import OutlineService
from ainovel.services.projects import ProjectService
from ainovel.web.security import csrf_token, require_csrf

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

BATCH_STATUS_LABELS = {
    "draft": "草稿", "ready_for_review": "待审批", "approved": "已批准", "rejected": "已驳回",
}
AUDIT_ACTION_LABELS = {
    "batch_created": "创建批次", "batch_ready": "提交审批", "batch_approved": "批准批次",
    "batch_rejected": "驳回批次", "chapter_published": "发布章节",
    "workflow_started": "启动生成", "workflow_cancelled": "取消生成",
    "plan_approved": "确认章节计划", "plan_rejected": "驳回章节计划",
}


def _batch_readiness(batch, chapters, active_batch_id):
    complete = (
        len(chapters) == batch.planned_chapters
        and {chapter.ordinal for chapter in chapters} == set(range(1, batch.planned_chapters + 1))
        and all(4500 <= chapter.visible_char_count <= 6000 for chapter in chapters)
    )
    return {
        "count": len(chapters),
        "can_submit": complete and batch.status == "draft" and batch.id == active_batch_id,
    }


def _project_context(request: Request, session: Session, project_id: str) -> dict[str, object]:
    try:
        summary = ProjectService(session).summary(project_id)
        official_outline = OutlineService(session).current_official_for_project(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    batches = BatchService(session)
    active_workflow = (
        session.get(GenerationWorkflow, summary.project.active_workflow_id)
        if summary.project.active_workflow_id is not None
        else None
    )
    if active_workflow is not None and active_workflow.project_id != summary.project.id:
        active_workflow = None
    batch_rows = batches.list_for_project(project_id)
    chapter_rows = session.execute(
        select(Chapter.batch_id, Chapter.ordinal, Chapter.visible_char_count)
        .where(Chapter.project_id == project_id)
    ).all()
    by_batch = {}
    for chapter in chapter_rows:
        by_batch.setdefault(chapter.batch_id, []).append(chapter)
    history = session.scalars(
        select(GenerationWorkflow).where(GenerationWorkflow.project_id == project_id)
        .order_by(GenerationWorkflow.created_at.desc(), GenerationWorkflow.id.desc()).limit(20)
    ).all()
    return {
        "request": request, "summary": summary, "official_outline": official_outline,
        "csrf_token": csrf_token(request),
        "batches": batch_rows,
        "batch_info": {batch.id: _batch_readiness(batch, by_batch.get(batch.id, []), summary.project.active_batch_id) for batch in batch_rows},
        "workflow_history": history,
        "workflow_labels": WORKFLOW_LABELS,
        "statistics": batches.official_chapter_statistics(project_id),
        "audit_events": batches.list_audit_events(project_id),
        "active_workflow": active_workflow,
        "chapter_test_mode": bool(
            getattr(request.app.state, "chapter_test_mode", False)
        ),
        "batch_status_labels": BATCH_STATUS_LABELS, "audit_action_labels": AUDIT_ACTION_LABELS,
    }


def _project_page(
    request: Request,
    session: Session,
    project_id: str,
    error: str | None = None,
    status_code: int = 200,
    diagnostic: dict[str, object] | None = None,
) -> object:
    context = _project_context(request, session, project_id)
    context["error"] = error
    context["diagnostic"] = diagnostic
    return templates.TemplateResponse(request, "project.html", context, status_code=status_code)


def _index_error(request: Request, session: Session, values: dict[str, str], error: str) -> object:
    return templates.TemplateResponse(request, "index.html", {
        "request": request, "projects": ProjectService(session).list_projects(),
        "project_form": values, "error": error, "csrf_token": csrf_token(request),
    }, status_code=422)


@router.get("/")
def index(request: Request, session: Session = Depends(get_session)) -> object:
    return templates.TemplateResponse(request, "index.html", {
        "request": request, "projects": ProjectService(session).list_projects(),
        "project_form": {"title": "", "target_chars_min": "", "target_chars_max": ""}, "error": None,
        "csrf_token": csrf_token(request),
    })


@router.post("/projects")
def create_project(request: Request, title: str = Form(""), target_chars_min: str = Form(""), target_chars_max: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    values = {"title": title, "target_chars_min": target_chars_min, "target_chars_max": target_chars_max}
    if not title.strip():
        return _index_error(request, session, values, "项目名称不能为空")
    try:
        minimum, maximum = int(target_chars_min), int(target_chars_max)
    except ValueError:
        return _index_error(request, session, values, "目标字数必须是整数")
    if minimum <= 0 or maximum <= 0:
        return _index_error(request, session, values, "目标字数必须为正整数")
    if minimum > maximum:
        return _index_error(request, session, values, "目标字数下限不能大于上限")
    try:
        project = ProjectService(session).create(title, minimum, maximum)
    except (ValueError, PermissionError):
        return _index_error(request, session, values, "项目数据无效")
    return RedirectResponse(f"/projects/{project.id}", status_code=303)


@router.get("/projects/{project_id}")
def project_page(project_id: str, request: Request, session: Session = Depends(get_session)) -> object:
    return _project_page(request, session, project_id)


@router.post("/projects/{project_id}/batches")
def create_batch(project_id: str, request: Request, planned_chapters: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    try:
        official_outline = OutlineService(session).current_official_for_project(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    if official_outline is None:
        return _project_page(request, session, project_id, "请先创建并确认官方大纲后再创建批次", 422)
    try:
        planned = int(planned_chapters)
    except ValueError:
        return _project_page(request, session, project_id, "计划章节数必须是整数", 422)
    try:
        BatchService(session).create(project_id, official_outline.id, planned)
    except (ValueError, PermissionError):
        return _project_page(request, session, project_id, "批次创建失败，请检查计划章节数", 422)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


def _batch_project_id(batch_id: str, session: Session) -> str:
    try:
        return BatchService(session).get(batch_id).project_id
    except ValueError as error:
        raise HTTPException(status_code=404, detail="批次不存在") from error


@router.get("/batches/{batch_id}")
def batch_preview(batch_id: str, request: Request, session: Session = Depends(get_session)) -> object:
    project_id = _batch_project_id(batch_id, session)
    service = BatchService(session)
    batch = service.get(batch_id)
    chapters = service.list_chapters(batch_id)
    project = ProjectService(session).get(project_id)
    source = session.get(GenerationWorkflow, batch.source_workflow_id) if batch.source_workflow_id else None
    if source is not None and source.project_id != project_id:
        source = None
    return templates.TemplateResponse(request, "batch.html", {
        "request": request, "project": project, "batch": batch, "chapters": chapters,
        "info": _batch_readiness(batch, chapters, project.active_batch_id),
        "source_workflow": source, "workflow_labels": WORKFLOW_LABELS,
        "csrf_token": csrf_token(request), "batch_status_labels": BATCH_STATUS_LABELS,
    })


@router.get("/batches/{batch_id}/ready")
def batch_ready_page(batch_id: str, session: Session = Depends(get_session)) -> object:
    _batch_project_id(batch_id, session)
    return RedirectResponse(f"/batches/{batch_id}", status_code=303)


@router.post("/batches/{batch_id}/ready")
def mark_batch_ready(batch_id: str, request: Request, session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    project_id = _batch_project_id(batch_id, session)
    try:
        BatchService(session).mark_ready(batch_id)
    except (ValueError, PermissionError):
        return _project_page(request, session, project_id, "批次尚未满足提交审批条件", 422)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/batches/{batch_id}/approve")
def approve_batch(batch_id: str, request: Request, session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    service = BatchService(session)
    try:
        batch = service.get(batch_id)
        service.approve(batch.id, batch.base_outline_version_id)
    except (ValueError, PermissionError):
        try:
            project_id = service.get(batch_id).project_id
        except ValueError as error:
            raise HTTPException(status_code=404, detail="批次不存在") from error
        return _project_page(request, session, project_id, "批次批准失败，内容或大纲状态已变化", 422)
    return RedirectResponse(f"/projects/{batch.project_id}", status_code=303)


@router.post("/batches/{batch_id}/reject")
def reject_batch(batch_id: str, request: Request, reason: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    project_id = _batch_project_id(batch_id, session)
    if not reason.strip():
        return _project_page(request, session, project_id, "请填写驳回原因", 422)
    try:
        BatchService(session).reject(batch_id, reason.strip())
    except (ValueError, PermissionError):
        return _project_page(request, session, project_id, "批次驳回失败，状态已变化", 422)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)

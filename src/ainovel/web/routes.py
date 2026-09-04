from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ainovel.db import get_session
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
}


def _project_context(request: Request, session: Session, project_id: str) -> dict[str, object]:
    try:
        summary = ProjectService(session).summary(project_id)
        official_outline = OutlineService(session).current_official_for_project(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    batches = BatchService(session)
    return {
        "request": request, "summary": summary, "official_outline": official_outline,
        "csrf_token": csrf_token(request),
        "batches": batches.list_for_project(project_id),
        "statistics": batches.official_chapter_statistics(project_id),
        "audit_events": batches.list_audit_events(project_id),
        "batch_status_labels": BATCH_STATUS_LABELS, "audit_action_labels": AUDIT_ACTION_LABELS,
    }


def _project_page(request: Request, session: Session, project_id: str, error: str | None = None, status_code: int = 200) -> object:
    context = _project_context(request, session, project_id)
    context["error"] = error
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

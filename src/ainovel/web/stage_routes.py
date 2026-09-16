from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.db import get_session
from ainovel.models.stage import StageModelAttempt, StageRoadmapVersion, StageWorkflow
from ainovel.models.workflow import GenerationWorkflow
from ainovel.providers.contracts import ProviderError
from ainovel.services.projects import ProjectService
from ainovel.services.stages import StageService
from ainovel.services.workflows import WorkflowBudgets
from ainovel.web.presentation import STAGE_ROADMAP_LABELS, WORKFLOW_LABELS
from ainovel.web.routes import _project_page, templates
from ainovel.web.security import csrf_token, require_csrf


router = APIRouter()


def _stage_context(request: Request, session: Session, stage_id: str) -> dict[str, object]:
    service = StageService(session)
    try:
        stage = service.get(stage_id)
        project = ProjectService(session).get(stage.project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="剧情阶段不存在") from error
    roadmaps = service.list_roadmaps(stage.id)
    roadmap_ids = [roadmap.id for roadmap in roadmaps]
    attempts = (
        session.scalars(
            select(StageModelAttempt)
            .where(StageModelAttempt.roadmap_id.in_(roadmap_ids))
            .order_by(StageModelAttempt.roadmap_id, StageModelAttempt.number)
        ).all()
        if roadmap_ids else []
    )
    attempts_by_roadmap: dict[str, list[StageModelAttempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_roadmap[attempt.roadmap_id].append(attempt)
    roadmap_rows = []
    for roadmap in roadmaps:
        difference = {}
        if roadmap.payload is not None and stage.approved_roadmap_id != roadmap.id:
            try:
                difference = service.roadmap_diff(stage.id, roadmap.id)
            except ValueError:
                difference = {}
        roadmap_rows.append({"roadmap": roadmap, "attempts": attempts_by_roadmap[roadmap.id], "difference": difference})
    approved = service.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
    estimated = approved.estimated_chapters if approved is not None else None
    proposed_estimate = next(
        (item.estimated_chapters for item in reversed(roadmaps) if item.payload is not None),
        None,
    )
    remaining = max(estimated - stage.confirmed_chapters, 0) if estimated is not None else None
    stage_workflows = session.scalars(
        select(StageWorkflow).where(StageWorkflow.stage_id == stage.id)
        .order_by(StageWorkflow.created_at, StageWorkflow.workflow_id)
    ).all()
    workflow_rows = []
    for mapping in stage_workflows:
        workflow = session.get(GenerationWorkflow, mapping.workflow_id)
        if workflow is not None:
            workflow_rows.append({"workflow": workflow, "mapping": mapping, "nodes": service.workflow_nodes(workflow.id)})
    active_workflow = session.get(GenerationWorkflow, project.active_workflow_id) if project.active_workflow_id else None
    return {
        "request": request, "csrf_token": csrf_token(request), "stage": stage,
        "project": project, "roadmap_rows": roadmap_rows, "approved_roadmap": approved,
        "estimated_chapters": estimated, "proposed_estimate": proposed_estimate,
        "remaining_chapters": remaining,
        "workflow_rows": workflow_rows, "active_workflow": active_workflow,
        "can_start_batch": approved is not None and bool(remaining) and project.active_workflow_id is None and project.active_batch_id is None,
        "stage_status_labels": STAGE_ROADMAP_LABELS, "workflow_labels": WORKFLOW_LABELS,
        "chapter_test_mode": bool(getattr(request.app.state, "chapter_test_mode", False)),
    }


def _stage_page(request: Request, session: Session, stage_id: str, error: str | None = None, status_code: int = 200) -> object:
    context = _stage_context(request, session, stage_id)
    context["error"] = error
    return templates.TemplateResponse(request, "stage.html", context, status_code=status_code)


@router.get("/stages/{stage_id}")
def stage_page(stage_id: str, request: Request, session: Session = Depends(get_session)) -> object:
    return _stage_page(request, session, stage_id)


@router.post("/projects/{project_id}/stages")
def create_stage(project_id: str, request: Request, architecture: str = Form(""), author_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    try:
        ProjectService(session).get(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    if not architecture.strip():
        return _project_page(request, session, project_id, "剧情阶段总体架构不能为空", 422)
    if author_confirm != "yes":
        return _project_page(request, session, project_id, "请确认剧情阶段总体架构", 422)
    try:
        stage = StageService(session).create(project_id, architecture.strip(), "author")
    except (ValueError, PermissionError):
        return _project_page(request, session, project_id, "剧情阶段创建失败，请检查项目状态", 422)
    return RedirectResponse(f"/stages/{stage.id}", status_code=303)


@router.post("/stages/{stage_id}/roadmaps")
def propose_roadmap(stage_id: str, request: Request, provider_name: str = Form(""), model_name: str = Form(""), architecture: str = Form(""), author_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    _stage_context(request, session, stage_id)
    if author_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认架构与模型设置", 422)
    if not request.app.state.provider_registry.contains(provider_name):
        return _stage_page(request, session, stage_id, "Provider 未配置", 422)
    if not model_name.strip():
        return _stage_page(request, session, stage_id, "模型名称不能为空", 422)
    try:
        StageService(session).propose_roadmap(stage_id, "author", provider_name, model_name.strip(), architecture=architecture.strip() or None)
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, "路线图提案创建失败，请检查项目状态", 422)
    return RedirectResponse(f"/stages/{stage_id}", status_code=303)


def _roadmap_for_stage(session: Session, stage_id: str, roadmap_id: str) -> StageRoadmapVersion:
    try:
        roadmap = StageService(session).roadmap(roadmap_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="路线图版本不存在") from error
    if roadmap.stage_id != stage_id:
        raise HTTPException(status_code=404, detail="路线图版本不存在")
    return roadmap


@router.post("/stages/{stage_id}/roadmaps/{roadmap_id}/generate")
def generate_roadmap(stage_id: str, roadmap_id: str, request: Request, model_call_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    roadmap = _roadmap_for_stage(session, stage_id, roadmap_id)
    if model_call_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认此次操作会调用模型并可能产生费用", 422)
    if not request.app.state.provider_registry.contains(roadmap.provider_name):
        return _stage_page(request, session, stage_id, "Provider 未配置", 422)
    try:
        provider = request.app.state.provider_registry.get(roadmap.provider_name)
        StageService(session).generate_roadmap(roadmap.id, provider)
    except (ValueError, PermissionError, ProviderError):
        return _stage_page(request, session, stage_id, "路线图生成当前无法执行", 422)
    return RedirectResponse(f"/stages/{stage_id}", status_code=303)


@router.post("/stages/{stage_id}/roadmaps/{roadmap_id}/approve")
def approve_roadmap(stage_id: str, roadmap_id: str, request: Request, approval_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    _roadmap_for_stage(session, stage_id, roadmap_id)
    if approval_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认批准此路线图版本", 422)
    try:
        StageService(session).approve_roadmap(stage_id, roadmap_id, "author")
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, "路线图批准失败，输入或项目状态已变化", 422)
    return RedirectResponse(f"/stages/{stage_id}", status_code=303)


@router.post("/stages/{stage_id}/batches")
def start_stage_batch(stage_id: str, request: Request, requested_chapters: str = Form(""), author_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    context = _stage_context(request, session, stage_id)
    if author_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认启动本批次（最多五章）", 422)
    try:
        count = int(requested_chapters)
    except ValueError:
        return _stage_page(request, session, stage_id, "批次章节数必须是整数", 422)
    approved = context["approved_roadmap"]
    if not isinstance(approved, StageRoadmapVersion):
        return _stage_page(request, session, stage_id, "请先批准路线图", 422)
    budgets = request.app.state.workflow_budgets
    if not isinstance(budgets, WorkflowBudgets):
        return _stage_page(request, session, stage_id, "工作流预算配置无效", 422)
    try:
        started = StageService(session).start_next_batch(stage_id, "author", approved.provider_name, approved.model_name, count, budgets=budgets)
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, "批次启动失败；请确认没有待审批批次、活动工作流或过期版本", 422)
    return RedirectResponse(f"/workflows/{started.workflow.id}", status_code=303)

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.db import get_session
from ainovel.models.prompt import WorkflowPromptSnapshot
from ainovel.models.project import NovelProject
from ainovel.models.batch import WritingBatch
from ainovel.models.workflow import (
    GenerationWorkflow,
    ModelAttempt,
    PlanDecision,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.services.projects import ProjectService
from ainovel.services.workflows import (
    DEFAULT_BUDGETS,
    EXECUTABLE_WORKFLOW_STATUSES,
    WorkflowService,
)
from ainovel.web.routes import _project_page, templates
from ainovel.web.security import csrf_token, require_csrf


router = APIRouter()


def _workflow_row(session: Session, workflow_id: str) -> GenerationWorkflow:
    workflow = session.get(GenerationWorkflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="工作流不存在")
    project = session.get(NovelProject, workflow.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return workflow


def _workflow_context(
    request: Request,
    session: Session,
    workflow_id: str,
) -> dict[str, object]:
    can_resume = WorkflowService(session).can_resume(workflow_id)
    can_cancel = WorkflowService(session).can_cancel(workflow_id)
    workflow = _workflow_row(session, workflow_id)
    project = session.get(NovelProject, workflow.project_id)
    assert project is not None
    steps = session.scalars(
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow.id)
        .order_by(WorkflowStep.position)
    ).all()
    step_by_id = {step.id: step for step in steps}
    attempts = session.scalars(
        select(ModelAttempt)
        .where(ModelAttempt.step_id.in_(step_by_id))
        .order_by(ModelAttempt.created_at, ModelAttempt.id)
    ).all() if step_by_id else []
    artifacts = session.scalars(
        select(WorkflowArtifact)
        .where(WorkflowArtifact.workflow_id == workflow.id)
        .order_by(WorkflowArtifact.created_at, WorkflowArtifact.id)
    ).all()
    snapshots = session.scalars(
        select(WorkflowPromptSnapshot)
        .where(WorkflowPromptSnapshot.workflow_id == workflow.id)
        .order_by(WorkflowPromptSnapshot.role)
    ).all()
    decisions = session.scalars(
        select(PlanDecision)
        .where(PlanDecision.workflow_id == workflow.id)
        .order_by(PlanDecision.created_at, PlanDecision.id)
    ).all()

    plan = next(
        (artifact for artifact in artifacts if artifact.kind == "batch_plan"),
        None,
    )
    plan_chapters = [] if plan is None else plan.payload.get("chapters", [])
    if not isinstance(plan_chapters, list):
        plan_chapters = []
    review_issues: list[str] = []
    for artifact in artifacts:
        if artifact.kind != "batch_review":
            continue
        issues = artifact.payload.get("issues", [])
        if isinstance(issues, list):
            review_issues.extend(issue for issue in issues if isinstance(issue, str))
    candidate_chapters = [
        {
            "title": artifact.payload.get("title", ""),
            "body": artifact.payload.get("body", artifact.text_content or ""),
            "visible_char_count": artifact.visible_char_count,
            "ordinal": artifact.ordinal,
        }
        for artifact in artifacts
        if artifact.kind == "chapter_draft"
        and isinstance(artifact.payload.get("title"), str)
        and isinstance(artifact.payload.get("body", artifact.text_content), str)
    ]

    return {
        "request": request,
        "workflow": workflow,
        "project": project,
        "steps": steps,
        "attempt_rows": [
            {"attempt": attempt, "step": step_by_id[attempt.step_id]}
            for attempt in attempts
        ],
        "artifacts": artifacts,
        "snapshots": snapshots,
        "decisions": decisions,
        "plan_chapters": plan_chapters,
        "review_issues": review_issues,
        "candidate_chapters": candidate_chapters,
        "csrf_token": csrf_token(request),
        "can_run": workflow.status in EXECUTABLE_WORKFLOW_STATUSES,
        "can_resume": can_resume,
        "can_cancel": can_cancel,
        "candidate_batch_id": workflow.candidate_batch_id or session.scalar(
            select(WritingBatch.id).where(
                WritingBatch.source_workflow_id == workflow.id,
                WritingBatch.project_id == workflow.project_id,
            )
        ),
        "is_paused": workflow.status.startswith("PAUSED_"),
        "chapter_test_mode": bool(
            getattr(request.app.state, "chapter_test_mode", False)
        ),
    }


def _workflow_page(
    request: Request,
    session: Session,
    workflow_id: str,
    error: str | None = None,
    status_code: int = 200,
) -> object:
    context = _workflow_context(request, session, workflow_id)
    context["error"] = error
    return templates.TemplateResponse(
        request,
        "workflow.html",
        context,
        status_code=status_code,
    )


def _start_error_message(error: ValueError) -> str:
    message = str(error)
    if message == "model name is required":
        return "模型名称不能为空"
    if message == "requested chapters must be between 1 and 5":
        return "计划章节数必须在 1 至 5 之间"
    if message == "project requires an approved constitution":
        return "请先确认创作宪法"
    if message == "project requires a current official outline":
        return "请先创建并确认官方大纲"
    if message in {"project already has an active workflow", "project already has an active batch"}:
        return "项目已有活动工作流或候选批次"
    return "工作流启动失败，请检查项目状态"


@router.get("/workflows/{workflow_id}")
def workflow_page(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> object:
    return _workflow_page(request, session, workflow_id)


@router.post("/projects/{project_id}/workflows")
def create_workflow(
    project_id: str,
    request: Request,
    provider_name: str = Form(""),
    model_name: str = Form(""),
    requested_chapters: str = Form(""),
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    try:
        ProjectService(session).get(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    try:
        count = int(requested_chapters)
    except ValueError:
        return _project_page(
            request, session, project_id, "计划章节数必须是整数", 422
        )
    required_count = getattr(request.app.state, "required_workflow_chapters", None)
    if required_count is not None and count != required_count:
        return _project_page(
            request, session, project_id, "单章测试仅允许生成 1 章", 422
        )
    if not request.app.state.provider_registry.contains(provider_name):
        return _project_page(request, session, project_id, "Provider 未配置", 422)
    try:
        workflow = WorkflowService(session).start(
            project_id,
            provider_name,
            model_name,
            count,
            getattr(request.app.state, "workflow_budgets", DEFAULT_BUDGETS),
        )
    except (ValueError, PermissionError) as error:
        return _project_page(
            request,
            session,
            project_id,
            _start_error_message(error),
            422,
        )
    return RedirectResponse(f"/workflows/{workflow.id}", status_code=303)


@router.post("/workflows/{workflow_id}/run")
def run_workflow(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    session.rollback()
    try:
        request.app.state.orchestrator_factory().run_until_blocked(workflow_id)
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id, "工作流当前无法运行", 422
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/workflows/{workflow_id}/plan/approve")
def approve_plan(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    try:
        WorkflowService(session).approve_plan(workflow_id, "author")
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id, "计划批准失败，状态已变化", 422
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/workflows/{workflow_id}/plan/reject")
def reject_plan(
    workflow_id: str,
    request: Request,
    reason: str = Form(""),
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    try:
        WorkflowService(session).reject_plan(workflow_id, reason, "author")
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id, "计划驳回失败，请填写原因并检查状态", 422
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/workflows/{workflow_id}/resume")
def resume_workflow(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    try:
        WorkflowService(session).resume(workflow_id)
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id, "此暂停状态不能恢复", 422
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/workflows/{workflow_id}/reconcile")
def reconcile_workflow(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    try:
        WorkflowService(session).reconcile_batch_decision(workflow_id)
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id, "候选批次对账失败，状态已变化", 422
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/workflows/{workflow_id}/cancel")
def cancel_workflow(
    workflow_id: str,
    request: Request,
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    _workflow_row(session, workflow_id)
    try:
        WorkflowService(session).cancel(workflow_id, "author")
    except (ValueError, PermissionError):
        return _workflow_page(
            request, session, workflow_id,
            "无法取消：状态或所有权已变化；已有候选批次请先完成正文审批并同步结果。", 422,
        )
    return RedirectResponse(f"/workflows/{workflow_id}", status_code=303)


@router.post("/projects/{project_id}/providers/diagnose")
def diagnose_provider(
    project_id: str,
    request: Request,
    provider_name: str = Form(""),
    model_name: str = Form(""),
    session: Session = Depends(get_session),
    _csrf: None = Depends(require_csrf),
) -> object:
    try:
        ProjectService(session).get(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="项目不存在") from error
    if not request.app.state.provider_registry.contains(provider_name):
        return _project_page(request, session, project_id, "Provider 未配置", 422)
    try:
        provider = request.app.state.provider_registry.get(provider_name)
    except Exception:
        return _project_page(
            request,
            session,
            project_id,
            diagnostic={
                "available": False,
                "message": "诊断失败：Provider 当前不可用或未配置",
                "models": (),
            },
        )

    try:
        diagnostic = provider.diagnose(model_name.strip() or None)
    except Exception:
        safe_result = {
            "available": False,
            "message": "诊断失败：Provider 当前不可用或未配置",
            "models": (),
        }
    else:
        safe_result = {
            "available": diagnostic.available,
            "message": (
                "诊断成功：Provider 可用"
                if diagnostic.available
                else "诊断失败：Provider 当前不可用或未配置"
            ),
            "models": diagnostic.models,
        }
    return _project_page(
        request,
        session,
        project_id,
        diagnostic=safe_result,
    )

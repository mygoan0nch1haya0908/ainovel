from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.db import get_session
from ainovel.services.llm_diagnostics import diagnostic_rows
from ainovel.agents.schema_diagnostics import safe_schema_issues, SCHEMA_ERROR_LABELS
from ainovel.models.audit import AuditEvent
from ainovel.models.stage import StageModelAttempt, StageRoadmapVersion, StageWorkflow
from ainovel.models.workflow import GenerationWorkflow
from ainovel.models.project import ConstitutionVersion
from ainovel.providers.contracts import ProviderError
from ainovel.providers.diagnostics import failure_code_detail
from ainovel.services.projects import ProjectService
from ainovel.services.stages import StageService, stage_context_report
from ainovel.services.stage_planning import PLOT_FORMAT, plot_point_ranges
from ainovel.services.workflows import WorkflowBudgets
from ainovel.web.presentation import STAGE_ROADMAP_LABELS, WORKFLOW_LABELS
from ainovel.web.routes import _project_page, templates
from ainovel.web.security import csrf_token, require_csrf
from ainovel.web.profile_selection import available_profiles, selected_profile, bound_profile, retry_selection_context


router = APIRouter()


def _planning_points(payload, confirmed=0):
    if not payload or payload.get('format') != PLOT_FORMAT:
        return []
    return [{**point, **span, 'confirmed_count': max(0, min(point['chapter_count'], confirmed - span['start'] + 1))}
            for point, span in zip(payload['points'], plot_point_ranges(payload), strict=True)]


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
    llm_diagnostics={(d.get('roadmap_id'),d.get('attempt_number')):d['diagnostic'] for d in diagnostic_rows(session,stage.id)}
    diagnostic_events = session.scalars(select(AuditEvent).where(
        AuditEvent.entity_id == stage.id,
        AuditEvent.action == 'stage_schema_validation_failed',
    )).all()
    diagnostics = {}
    for event in diagnostic_events:
        details = event.details
        if isinstance(details, dict):
            key = (details.get('roadmap_id'), details.get('attempt_number'))
            if isinstance(key[0], str) and type(key[1]) is int:
                diagnostics[key] = details.get('issues', [])
    for roadmap in roadmaps:
        difference = {}
        if roadmap.payload is not None and (stage.approved_roadmap_id != roadmap.id or roadmap.input_snapshot.get('revision')):
            try:
                difference = service.roadmap_diff(stage.id, roadmap.id)
            except ValueError:
                difference = {}
        profile = bound_profile(request, session, roadmap.model_profile_version_id)
        if 'points' in difference:
            revision = roadmap.input_snapshot.get('revision')
            previous = service.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            before = revision['source_payload'] if revision else previous.payload if previous else None
            difference['points'] = {'before': _planning_points(before), 'after': _planning_points(roadmap.payload)}
        report = None
        try:
            if profile:
                window, output = profile.context_limit, profile.output_limit
            else:
                caps = request.app.state.provider_registry.get(roadmap.provider_name).capabilities(roadmap.model_name)
                window, output = caps.context_window, caps.max_output_tokens
            report = stage_context_report(roadmap, window, output)
        except (ValueError, ProviderError):
            pass
        roadmap_rows.append({"roadmap": roadmap, "attempts": attempts_by_roadmap[roadmap.id], "difference": difference,
                             "profile": profile, "context_report": report,
                             'llm_diagnostics':{a.number:llm_diagnostics.get((roadmap.id,a.number)) for a in attempts_by_roadmap[roadmap.id]},
                             'schema_issues': {attempt.number: safe_schema_issues(
                                 diagnostics.get((roadmap.id, attempt.number), []),
                                 roadmap.prompt_snapshot.get('schema', {}))
                                 for attempt in attempts_by_roadmap[roadmap.id]},
                             'planning_view': {'format': roadmap.payload.get('format') if roadmap.payload else None,
                                               'points': _planning_points(roadmap.payload, stage.confirmed_chapters if stage.approved_roadmap_id == roadmap.id else 0)}})
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
        "model_profiles": available_profiles(request, session),
        "approved_profile": bound_profile(request, session, approved.model_profile_version_id) if approved else None,
        "project": project, "roadmap_rows": roadmap_rows, "approved_roadmap": approved,
        "estimated_chapters": estimated, "proposed_estimate": proposed_estimate,
        "remaining_chapters": remaining,
        'legacy_locked': bool(approved and approved.payload.get('format') != PLOT_FORMAT and stage.confirmed_chapters),
        "workflow_rows": workflow_rows, "active_workflow": active_workflow,
        "can_start_batch": approved is not None and bool(remaining) and project.active_workflow_id is None and project.active_batch_id is None,
        "stage_status_labels": STAGE_ROADMAP_LABELS, "workflow_labels": WORKFLOW_LABELS,
        "chapter_test_mode": bool(getattr(request.app.state, "chapter_test_mode", False)),
        "outline_hierarchy": service.outline_context(stage.id),
        "failure_code_detail": failure_code_detail,
        "schema_error_labels": SCHEMA_ERROR_LABELS,
        "current_constitution": session.get(ConstitutionVersion, project.current_constitution_version_id) if project.current_constitution_version_id else None,
        "constitution_labels": {"setting_style": "世界设定与文风", "provisional_ending": "暂定结局",
                                "genre": "小说类型", "voice": "叙述文风", "rules": "创作规则"},
    }


def _stage_page(request: Request, session: Session, stage_id: str, error: str | None = None, status_code: int = 200,
                roadmap_form: dict[str, str] | None = None, feedback_form: dict[str, str] | None = None) -> object:
    context = _stage_context(request, session, stage_id)
    context["error"] = error
    context["roadmap_form"] = roadmap_form or {}
    context['feedback_form'] = feedback_form or {}
    if roadmap_form is not None:
        context.update(retry_selection_context(request, session, roadmap_form))
    return templates.TemplateResponse(request, "stage.html", context, status_code=status_code)


@router.get("/stages/{stage_id}")
def stage_page(stage_id: str, request: Request, retry_roadmap: str = "", session: Session = Depends(get_session)) -> object:
    if retry_roadmap:
        previous = _roadmap_for_stage(session, stage_id, retry_roadmap)
        if previous.status != "PAUSED_CONTEXT_OVERFLOW":
            raise HTTPException(status_code=422, detail="仅本次发送给 AI 的内容过长记录可使用此入口")
        return _stage_page(request, session, stage_id, roadmap_form={
            "provider_name": previous.provider_name, "model_name": previous.model_name,
            "model_profile_version_id": previous.model_profile_version_id or "",
            "architecture": previous.architecture,
            'output_token_limit': str(previous.output_token_limit),
            'roadmap_format': PLOT_FORMAT if 'points' in previous.prompt_snapshot['schema'].get('properties', {}) else 'legacy',
        })
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
def propose_roadmap(stage_id: str, request: Request, provider_name: str = Form(""), model_name: str = Form(""), model_profile_version_id: str = Form(""), provider_consent: str = Form(""), architecture: str = Form(""), author_confirm: str = Form(""), roadmap_format: str = Form(PLOT_FORMAT), output_token_limit: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    _stage_context(request, session, stage_id)
    roadmap_form = {"provider_name": provider_name, "model_name": model_name,
                    'roadmap_format': roadmap_format,
                    'output_token_limit': output_token_limit,
                    "model_profile_version_id": model_profile_version_id,
                    "architecture": architecture, "author_confirm": author_confirm}
    if author_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认架构与模型设置", 422, roadmap_form)
    output_text = output_token_limit.strip()
    if output_text and (len(output_text) > 10 or not output_text.isascii() or not output_text.isdecimal()
                        or not 1 <= int(output_text) <= 2147483647):
        return _stage_page(request, session, stage_id, '规划输出预算必须是正整数 Token', 422, roadmap_form)
    requested_output = int(output_text) if output_text else None
    if model_profile_version_id:
        try:
            view = selected_profile(request, session, model_profile_version_id, provider_consent)
            provider_name, model_name = "compatible", view.model_name
        except ValueError:
            return _stage_page(request, session, stage_id, "请选择可用配置并确认发送小说内容", 422, roadmap_form)
    if not request.app.state.provider_registry.contains(provider_name):
        return _stage_page(request, session, stage_id, "尚未配置模型服务商", 422, roadmap_form)
    if not model_name.strip():
        return _stage_page(request, session, stage_id, "模型名称不能为空", 422, roadmap_form)
    try:
        capacity = {}
        if model_profile_version_id:
            window, maximum_output = view.context_limit, view.output_limit
        if not model_profile_version_id:
            caps = request.app.state.provider_registry.get(provider_name).capabilities(model_name.strip())
            capacity = {"context_window": caps.context_window, "max_output_tokens": caps.max_output_tokens}
            window, maximum_output = caps.context_window, caps.max_output_tokens
        if requested_output is not None and (requested_output > maximum_output or requested_output + 1024 >= window):
            return _stage_page(request, session, stage_id,
                f'输出预算超出可用容量：模型配置输出上限为 {maximum_output} Token，总上下文为 {window} Token，还须为输入和安全余量留空间。', 422, roadmap_form)
        StageService(session).propose_roadmap(stage_id, "author", provider_name, model_name if model_profile_version_id else model_name.strip(),
            requested_output_tokens=requested_output,
            roadmap_format=roadmap_format,
            architecture=architecture.strip() or None,
            model_profile_version_id=model_profile_version_id or None, **capacity)
    except (ValueError, PermissionError, ProviderError):
        return _stage_page(request, session, stage_id, "章节安排提案创建失败，请检查项目状态", 422, roadmap_form)
    return RedirectResponse(f"/stages/{stage_id}", status_code=303)


def _roadmap_for_stage(session: Session, stage_id: str, roadmap_id: str) -> StageRoadmapVersion:
    try:
        roadmap = StageService(session).roadmap(roadmap_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="章节安排版本不存在") from error
    if roadmap.stage_id != stage_id:
        raise HTTPException(status_code=404, detail="章节安排版本不存在")
    return roadmap


@router.post("/stages/{stage_id}/roadmaps/{roadmap_id}/generate")
def generate_roadmap(stage_id: str, roadmap_id: str, request: Request, model_call_confirm: str = Form(""), provider_consent: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    roadmap = _roadmap_for_stage(session, stage_id, roadmap_id)
    if roadmap.model_profile_version_id and provider_consent != "yes":
        return _stage_page(request, session, stage_id, "请确认将小说内容发送给此版本目标服务商", 422)
    if model_call_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认此次操作会调用模型并可能产生费用", 422)
    if not request.app.state.provider_registry.contains(roadmap.provider_name):
        return _stage_page(request, session, stage_id, "尚未配置模型服务商", 422)
    try:
        provider = (
            None if roadmap.provider_name == "compatible"
            else request.app.state.provider_resolver.resolve(
                roadmap.provider_name, roadmap.model_name,
                model_profile_version_id=roadmap.model_profile_version_id,
            )
        )
        StageService(session, provider_resolver=request.app.state.provider_resolver).generate_roadmap(
            roadmap.id, provider
        )
    except (ValueError, PermissionError, ProviderError):
        return _stage_page(request, session, stage_id, "章节安排生成当前无法执行", 422)
    return RedirectResponse(f"/stages/{stage_id}", status_code=303)


@router.post('/stages/{stage_id}/roadmaps/{roadmap_id}/feedback')
def roadmap_feedback(stage_id: str, roadmap_id: str, request: Request, feedback: str = Form(''),
                     stage_revision: str = Form(''), feedback_confirm: str = Form(''),
                     session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    _roadmap_for_stage(session, stage_id, roadmap_id)
    form = {'roadmap_id': roadmap_id, 'feedback': feedback}
    if not feedback.strip() or feedback_confirm != 'yes':
        return _stage_page(request, session, stage_id, '请填写修改意见并确认保存；保存不会调用模型。', 422, feedback_form=form)
    try:
        revision = int(stage_revision)
        new = StageService(session).revise_roadmap(stage_id, roadmap_id, feedback, 'author', expected_revision=revision)
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, '暂时不能修订：请确认无进行中的写作或待审批正文，并刷新核对最新安排和模型配置。意见已保留在下方。', 422, feedback_form=form)
    return RedirectResponse(f'/stages/{stage_id}#planning-{new.id}', status_code=303)


@router.post("/stages/{stage_id}/roadmaps/{roadmap_id}/approve")
def approve_roadmap(stage_id: str, roadmap_id: str, request: Request, approval_confirm: str = Form(""), session: Session = Depends(get_session), _csrf: None = Depends(require_csrf)) -> object:
    _roadmap_for_stage(session, stage_id, roadmap_id)
    if approval_confirm != "yes":
        return _stage_page(request, session, stage_id, "请确认批准此章节安排版本", 422)
    try:
        StageService(session).approve_roadmap(stage_id, roadmap_id, "author")
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, "章节安排批准失败，输入或项目状态已变化", 422)
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
        return _stage_page(request, session, stage_id, "请先批准章节安排", 422)
    budgets = request.app.state.workflow_budgets
    if not isinstance(budgets, WorkflowBudgets):
        return _stage_page(request, session, stage_id, "工作流预算配置无效", 422)
    try:
        started = StageService(session).start_next_batch(stage_id, "author", approved.provider_name, approved.model_name, count, budgets=budgets)
    except (ValueError, PermissionError):
        return _stage_page(request, session, stage_id, "批次启动失败；请确认没有待审批批次、活动工作流或过期版本", 422)
    return RedirectResponse(f"/workflows/{started.workflow.id}", status_code=303)

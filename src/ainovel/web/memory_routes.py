"""Local author controls; GET/preview never commits or dispatches a model."""
import json

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.agents.memory_contracts import MemoryEntryInput
from ainovel.db import get_session
from ainovel.models import GenerationWorkflow, MemoryCardVersion, NovelProject, StoryMemoryEntry, WorkflowStep, MemoryExtractionJob
from ainovel.services.context_policy import ContextPolicyService
from ainovel.services.scoped_context import ScopedContextService, active_policy
from ainovel.services.story_memory import StoryMemoryService, digest
from ainovel.models import ProjectLLMClaim
from ainovel.services.project_llm_guard import ProjectLLMGuard
from ainovel.web.routes import templates
from ainovel.web.security import csrf_token, require_csrf

router = APIRouter()


def _project(session, project_id):
    project = session.get(NovelProject, project_id)
    if project is None:
        raise HTTPException(404, '项目不存在')
    return project


def _memory_page(request, session, project_id, *, error=None, status=200, submitted=None):
    service = StoryMemoryService(session)
    cards = session.scalars(select(MemoryCardVersion).where(MemoryCardVersion.project_id == project_id).order_by(MemoryCardVersion.version_number.desc())).all()
    sources = service.sources(project_id)
    approved_card = next((c for c in cards if c.status == 'APPROVED'), None)
    pending = session.scalars(select(StoryMemoryEntry).where(
        StoryMemoryEntry.project_id == project_id, StoryMemoryEntry.state_scope.like('pending:%'))).all()
    return templates.TemplateResponse(request=request, name='memory.html', context={
        'llm_claim': session.get(ProjectLLMClaim, project_id),
        'profiles': request.app.state.model_profile_service_factory(session).list_public(),
        'extraction_jobs': session.scalars(select(MemoryExtractionJob).where(MemoryExtractionJob.project_id == project_id).order_by(MemoryExtractionJob.created_at.desc())).all(),
        'project': _project(session, project_id), 'sources': sources, 'cards': cards,
        'card_blockers': {c.id: service.card_blockers(c) for c in cards}, 'source_fingerprint': digest(sources),
        'entries_json': submitted if submitted is not None else json.dumps(cards[0].entries if cards else [], ensure_ascii=False, indent=2),
        'pending': [row for row in pending if not service.pending_confirmed(row, approved_card)],
        'csrf_token': csrf_token(request), 'error': error}, status_code=status)


@router.get('/projects/{project_id}/memory')
def memory_page(project_id: str, request: Request, session: Session = Depends(get_session)):
    return _memory_page(request, session, project_id)


@router.post('/projects/{project_id}/memory/reconcile-call', dependencies=[Depends(require_csrf)])
def reconcile_call(project_id: str, owner_id: str = Form(...), author_confirm: str = Form(''),
                   session: Session = Depends(get_session)):
    _project(session, project_id)
    try:
        if author_confirm != 'yes':
            raise ValueError('请确认所有旧工作进程已停止，并已核对供应商用量')
        ProjectLLMGuard.reconcile(session, project_id, owner_id, 'author')
        session.commit()
    except ValueError as error:
        session.rollback()
        raise HTTPException(422, str(error)) from None
    return RedirectResponse(f'/projects/{project_id}/memory', 303)


@router.post('/projects/{project_id}/memory/cards', dependencies=[Depends(require_csrf)])
async def create_card(project_id: str, request: Request, session: Session = Depends(get_session)):
    _project(session, project_id)
    form = await request.form()
    submitted = form.get('entries_json')
    try:
        service = StoryMemoryService(session)
        if submitted is not None:
            entries = json.loads(submitted)
            if not isinstance(entries, list):
                raise ValueError('约束卡必须是条目列表')
        else:
            sources = service.sources(project_id)
            if form.get('source_fingerprint') != digest(sources):
                raise ValueError('原文已更新，请刷新后重新整理')
            entries = []
            for index, source in enumerate(sources):
                text = str(form.get(f'text_{index}', '')).strip()
                if not text:
                    continue
                entries.append(dict(kind=form.get(f'kind_{index}', 'rule'), text=text, source_refs=[source['ref']],
                    point_ids=[s.strip() for s in str(form.get(f'points_{index}', '')).split(',') if s.strip()],
                    entity_ids=[s.strip() for s in str(form.get(f'entities_{index}', '')).split(',') if s.strip()],
                    audience=form.get(f'audience_{index}', 'author_only'), author_locked=form.get(f'locked_{index}') == 'yes'))
        if not entries:
            return _memory_page(request, session, project_id,
                error='尚未填写任何精简内容，未创建空草稿。请展开下方原文，填写需要保留的规则或事实，再保存。原有设定不会自动变成约束卡。',
                status=422, submitted=submitted)
        service.create_card(project_id, [MemoryEntryInput.model_validate(e) for e in entries], 'author')
        session.commit()
    except (ValueError, TypeError):
        session.rollback()
        return _memory_page(request, session, project_id, error='保存失败：请检查条目类型、来源、章节范围或刷新已过期的原文。输入未自动批准。', status=422,
                            submitted=submitted if submitted is not None else json.dumps(entries if 'entries' in locals() else [], ensure_ascii=False))
    return RedirectResponse(f'/projects/{project_id}/memory', status_code=303)


@router.post('/projects/{project_id}/memory/cards/{card_id}/approve', dependencies=[Depends(require_csrf)])
def approve_card(project_id: str, card_id: str, request: Request, fingerprint: str = Form(...), author_confirm: str = Form(''), session: Session = Depends(get_session)):
    card = session.get(MemoryCardVersion, card_id)
    if card is None or card.project_id != project_id:
        raise HTTPException(404, '约束卡不属于此项目')
    try:
        if author_confirm != 'yes':
            raise ValueError('请确认已核对原文、规则覆盖和信息揭示范围')
        if not card.entries:
            raise ValueError('这版草稿没有任何约束条目，不能批准。请在下方「整理原文」填写精简内容并保存新版本；原文仍完整保留。')
        StoryMemoryService(session).approve_card(card_id, fingerprint, 'author')
        session.commit()
    except ValueError as error:
        session.rollback()
        return _memory_page(request, session, project_id, error=str(error), status=422)
    return RedirectResponse(f'/projects/{project_id}/memory', status_code=303)


@router.get('/workflows/{workflow_id}/context-preview')
def context_preview(workflow_id: str, request: Request, card_id: str | None = None, session: Session = Depends(get_session)):
    workflow = session.get(GenerationWorkflow, workflow_id)
    if workflow is None:
        raise HTTPException(404, '工作流不存在')
    cards = session.scalars(select(MemoryCardVersion).where(MemoryCardVersion.project_id == workflow.project_id).order_by(MemoryCardVersion.version_number.desc())).all()
    policy = active_policy(session, workflow_id)
    # An unbound preview may suggest a card but never activates it.
    selected_id = card_id or (policy.card_id if policy else None) or next((c.id for c in cards if c.status == 'APPROVED'), None)
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow_id, WorkflowStep.position == workflow.current_position))
    preview, error, conversion_blocker = None, None, None
    try:
        if step is None:
            raise ValueError('当前没有可预览的模型步骤')
        preview = ScopedContextService(session).preview(workflow_id, step.id, card_id=selected_id)
    except ValueError:
        error = '无法构建预览：请检查剧情点格式、约束卡所属项目、固定来源与步骤资料。旧章节节点流程请保留原任务，从阶段页创建剧情点规划后再建立新任务。'
    try:
        ContextPolicyService(session)._idle(workflow_id)
    except ValueError as exc:
        conversion_blocker = str(exc)
    return templates.TemplateResponse(request=request, name='context_preview.html', context={
        'workflow': workflow, 'cards': cards, 'selected_card_id': selected_id, 'policy': policy,
        'preview': preview, 'error': error, 'conversion_blocker': conversion_blocker, 'csrf_token': csrf_token(request)})


@router.post('/workflows/{workflow_id}/context-policy', dependencies=[Depends(require_csrf)])
def activate_context(workflow_id: str, card_id: str = Form(...), fingerprint: str = Form(...), author_confirm: str = Form(''), session: Session = Depends(get_session)):
    if author_confirm != 'yes':
        raise HTTPException(422, '请明确确认采用新输入模式；未发送模型请求')
    try:
        ContextPolicyService(session).activate(workflow_id, card_id, fingerprint, 'author')
        session.commit()
    except ValueError as error:
        session.rollback()
        raise HTTPException(422, str(error)) from None
    return RedirectResponse(f'/workflows/{workflow_id}/context-preview', status_code=303)


@router.post('/workflows/{workflow_id}/context-policy/revert', dependencies=[Depends(require_csrf)])
def revert_context(workflow_id: str, policy_version: int = Form(...), session: Session = Depends(get_session)):
    try:
        ContextPolicyService(session).revert(workflow_id, policy_version, 'author')
        session.commit()
    except ValueError as error:
        session.rollback()
        raise HTTPException(422, str(error)) from None
    return RedirectResponse(f'/workflows/{workflow_id}/context-preview', status_code=303)

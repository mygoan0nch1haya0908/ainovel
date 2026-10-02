import json
from uuid import uuid4
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from ainovel.db import get_session
from ainovel.models import MemoryExtractionJob, MemoryExtractionAuthorization, MemoryExtractionAttempt, MemoryExtractionChunk
from ainovel.services.memory_extraction import MemoryExtractionService
from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
from ainovel.web.routes import templates
from ainovel.web.security import require_csrf, csrf_token

router = APIRouter()


def task(session, job_id, project_id=None):
    job = session.get(MemoryExtractionJob, job_id)
    if job is None or (project_id is not None and project_id != job.project_id):
        raise HTTPException(404, '提取任务不属于此项目')
    return job


@router.post('/projects/{project_id}/memory/extractions', dependencies=[Depends(require_csrf)])
def prepare(project_id: str, profile_version_id: str = Form(...), pending_ids: list[str] = Form([]),
            session: Session = Depends(get_session)):
    try:
        job = MemoryExtractionService(session).prepare(project_id, profile_version_id, None, pending_ids, 'author')
    except ValueError:
        session.rollback()
        raise HTTPException(422, '无法准备：请检查模型档案、原文来源及可用上下文容量') from None
    return RedirectResponse(f'/memory-extractions/{job.id}', 303)


@router.get('/memory-extractions/{job_id}')
def page(job_id: str, request: Request, session: Session = Depends(get_session)):
    job = task(session, job_id)
    service = MemoryExtractionService(session)
    chunks = service.chunks(job.id)
    auth = session.scalar(select(MemoryExtractionAuthorization).where(MemoryExtractionAuthorization.job_id == job.id,
        MemoryExtractionAuthorization.status == 'ACTIVE').order_by(MemoryExtractionAuthorization.created_at.desc()))
    attempts = session.scalars(select(MemoryExtractionAttempt).join(MemoryExtractionChunk,
        MemoryExtractionChunk.id == MemoryExtractionAttempt.chunk_id).where(MemoryExtractionChunk.job_id == job.id)
        .order_by(MemoryExtractionAttempt.created_at)).all()
    return templates.TemplateResponse(request=request, name='memory_extraction.html', context=dict(
        job=job, chunks=chunks, auth=auth, attempts=attempts, authorization_id=str(uuid4()),
        output_budget=sum(c.snapshot['request']['max_output_tokens'] for c in chunks if c.status not in ('SUCCEEDED', 'REUSED')),
        csrf_token=csrf_token(request)))


@router.get('/memory-extractions/{job_id}/diff')
def diff(job_id: str, request: Request, session: Session = Depends(get_session)):
    job = task(session, job_id)
    try:
        preview = MemoryExtractionMergeService(session).preview(job_id)
    except ValueError:
        raise HTTPException(422, '候选来源校验未通过，请重新整理') from None
    return templates.TemplateResponse(request=request, name='memory_extraction_diff.html', context=dict(
        job=job, preview=preview, entries_json=json.dumps(preview['entries'], ensure_ascii=False, indent=2),
        csrf_token=csrf_token(request)))


@router.post('/memory-extractions/{job_id}/{action}', dependencies=[Depends(require_csrf)])
def action(job_id: str, action: str, request: Request, project_id: str = Form(...), revision: int = Form(...),
           authorization_id: str = Form(''), author_confirm: str = Form(''), fingerprint: str = Form(''),
           entries_json: str = Form('[]'), retained_obsolete_ids: list[str] = Form([]), session: Session = Depends(get_session)):
    job = task(session, job_id, project_id)
    service = MemoryExtractionService(session, request.app.state.provider_resolver)
    try:
        if action == 'authorize':
            if author_confirm != 'yes' or not authorization_id:
                raise ValueError('请确认模型、资料范围及本批调用预算')
            service.authorize(job_id, revision, authorization_id, 'author')
        elif action == 'step':
            service.run_next(job_id, authorization_id, expected_revision=revision)
        elif action == 'cancel':
            service.cancel(job_id, revision, 'author')
        elif action == 'interrupted':
            if author_confirm != 'yes':
                raise ValueError('请核对供应商用量后确认中断')
            service.mark_interrupted(job_id, revision, 'author')
        elif action == 'merge':
            if author_confirm != 'yes':
                raise ValueError('请先核对差异')
            entries = json.loads(entries_json)
            if not isinstance(entries, list):
                raise ValueError('条目必须为列表')
            MemoryExtractionMergeService(session).merge(job_id, revision, fingerprint, entries, 'author',
                retained_obsolete_ids=retained_obsolete_ids)
            return RedirectResponse(f'/projects/{project_id}/memory', 303)
        else:
            raise HTTPException(404, '操作不存在')
    except ValueError:
        session.rollback()
        raise HTTPException(422, '操作未执行：请刷新核对状态、资料版本、授权预算及待确认项；未自动重试') from None
    if action == 'step' and 'application/json' in request.headers.get('accept', ''):
        return JSONResponse(dict(status=job.status, revision=job.revision))
    return RedirectResponse(f'/memory-extractions/{job_id}', 303)

"""Serial, explicitly authorized extraction. Every dispatch has a durable attempt."""
from datetime import datetime, timezone
from copy import deepcopy
from uuid import uuid4
from sqlalchemy import select, update
from ainovel.models import (MemoryCardVersion, MemoryExtractionJob, MemoryExtractionChunk,
    MemoryExtractionAuthorization, MemoryExtractionAttempt, AuditEvent)
from ainovel.agents.runner import AgentRunner
from ainovel.agents.memory_extraction_contracts import ExtractionResult, identified_entries
from ainovel.providers.contracts import ModelRequest
from ainovel.providers.diagnostics import safe_failure_code
from ainovel.services.model_profiles import ModelProfileService
from ainovel.services.story_memory import digest
from ainovel.services.memory_extraction_sources import freeze_sources, build_chunks, RULE_VERSION
from ainovel.services.project_llm_guard import ProjectLLMGuard

PROCESS_STARTED = datetime.now(timezone.utc)


class MemoryExtractionService:
    def __init__(self, session, provider_resolver=None):
        self.session, self.provider_resolver = session, provider_resolver

    def latest_card(self, project_id):
        return self.session.scalar(select(MemoryCardVersion).where(MemoryCardVersion.project_id == project_id)
            .order_by(MemoryCardVersion.version_number.desc()))

    def get(self, job_id):
        job = self.session.get(MemoryExtractionJob, job_id)
        if job is None:
            raise ValueError('提取任务不存在')
        return job

    def lock(self, job_id):
        self.session.execute(update(MemoryExtractionJob).where(MemoryExtractionJob.id == job_id)
                             .values(revision=MemoryExtractionJob.revision))
        self.session.expire_all()
        return self.get(job_id)

    def chunks(self, job_id):
        return self.session.scalars(select(MemoryExtractionChunk).where(MemoryExtractionChunk.job_id == job_id)
                                   .order_by(MemoryExtractionChunk.ordinal)).all()

    def profile(self, version_id):
        view = ModelProfileService(self.session).get_public(version_id)
        if not view.enabled or view.revoked:
            raise ValueError('模型档案不可用')
        return dict(model=view.model_name, version_id=view.version_id, context_limit=view.context_limit,
                    output_limit=view.output_limit, base_url=view.base_url, timeout_seconds=120)

    def current(self, job):
        card = self.latest_card(job.project_id)
        return ((card.id if card else None) == job.base_card_id
                and digest(freeze_sources(self.session, job.project_id, job.snapshot['sources']['pending_ids'])) == job.source_fingerprint)

    def prepare(self, project_id, profile_version_id, base_card_id, pending_ids, actor):
        profile = self.profile(profile_version_id)
        card = self.latest_card(project_id)
        if base_card_id and (card is None or card.id != base_card_id):
            raise ValueError('请选择最新记忆卡作为整理基础')
        sources = freeze_sources(self.session, project_id, pending_ids)
        base = identified_entries(card.id, card.entries) if card else []
        chunks = build_chunks(sources, base, profile)
        if not chunks:
            raise ValueError('没有可整理的原文')
        job = MemoryExtractionJob(id=str(uuid4()), project_id=project_id,
            model_profile_version_id=profile_version_id, base_card_id=card.id if card else None,
            snapshot=dict(sources=sources, base=base, profile=profile), source_fingerprint=digest(sources),
            rule_version=RULE_VERSION, status='DRAFT')
        self.session.add(job)
        self.session.flush()
        for index, chunk in enumerate(chunks):
            cached = self.session.scalar(select(MemoryExtractionChunk).join(MemoryExtractionJob,
                MemoryExtractionJob.id == MemoryExtractionChunk.job_id).where(
                MemoryExtractionJob.project_id == project_id, MemoryExtractionJob.rule_version == RULE_VERSION,
                MemoryExtractionChunk.cache_key == chunk['cache_key'],
                MemoryExtractionChunk.status.in_(('SUCCEEDED', 'REUSED'))).order_by(MemoryExtractionChunk.created_at.desc()))
            self.session.add(MemoryExtractionChunk(id=str(uuid4()), job_id=job.id, ordinal=index,
                snapshot=chunk, cache_key=chunk['cache_key'], status='REUSED' if cached else 'PENDING',
                result=deepcopy(cached.result) if cached else None))
        self.session.flush()
        if all(c.status == 'REUSED' for c in self.chunks(job.id)):
            job.status = 'NEEDS_REVIEW'
        self.audit(job, 'memory_extraction_prepared', actor)
        self.session.commit()
        return job

    def authorize(self, job_id, expected_revision, authorization_id, actor):
        job = self.lock(job_id)
        existing = self.session.get(MemoryExtractionAuthorization, authorization_id)
        if existing:
            if existing.job_id != job_id:
                raise ValueError('授权不属于此任务')
            self.session.rollback()
            return {'id': existing.id}
        if job.revision != expected_revision or job.status not in ('DRAFT', 'PAUSED') or not self.current(job):
            raise ValueError('任务或资料已变化，不能授权；请重新准备或核对中断')
        if any(c.status in ('RUNNING', 'UNKNOWN') for c in self.chunks(job_id)):
            raise ValueError('请先核对中断请求')
        self.profile(job.model_profile_version_id)
        for chunk in self.chunks(job_id):
            if chunk.status == 'FAILED':
                chunk.status = 'PENDING'
        job.status, job.revision = 'READY', job.revision + 1
        auth = MemoryExtractionAuthorization(id=authorization_id, job_id=job.id, job_revision=job.revision)
        self.session.add(auth)
        self.audit(job, 'memory_extraction_authorized', actor)
        self.session.commit()
        return {'id': auth.id}

    def run_next(self, job_id, authorization_id, provider=None, *, expected_revision=None):
        job = self.lock(job_id)
        auth = self.session.get(MemoryExtractionAuthorization, authorization_id)
        if (not auth or auth.job_id != job.id or auth.status != 'ACTIVE' or auth.calls_used >= 8
                or job.status != 'READY' or (expected_revision is not None and job.revision != expected_revision)):
            raise ValueError('任务不能继续，授权已结束或请求重复')
        if not self.current(job):
            job.status = 'STALE'
            self.session.commit()
            return job
        profile = self.profile(job.model_profile_version_id)
        chunks = self.chunks(job_id)
        chunk = next((c for c in chunks if c.status == 'PENDING'), None)
        if chunk is None:
            job.status, auth.status = 'NEEDS_REVIEW', 'CLOSED'
            self.session.commit()
            return job
        if provider is None:
            if self.provider_resolver is None:
                raise ValueError('模型调用入口未配置')
            provider = self.provider_resolver.resolve('compatible', profile['model'],
                model_profile_version_id=job.model_profile_version_id)
        cap = provider.capabilities(profile['model'])
        request = ModelRequest(**chunk.snapshot['request'])
        if (request.max_output_tokens > cap.max_output_tokens or
                request.max_input_tokens + request.max_output_tokens + 1024 > cap.context_window):
            raise ValueError('模型容量变化，请重新准备任务')
        if auth.reserved_output_tokens + request.max_output_tokens > 512000:
            raise ValueError('本批输出预留预算不足')
        attempt_id = str(uuid4())
        ProjectLLMGuard.claim(self.session, job.project_id, attempt_id)
        job.status, job.revision, chunk.status = 'RUNNING', job.revision + 1, 'RUNNING'
        auth.calls_used += 1
        auth.reserved_output_tokens += request.max_output_tokens
        attempt = MemoryExtractionAttempt(id=attempt_id, chunk_id=chunk.id, authorization_id=auth.id,
            status='RUNNING', started_at=datetime.now(timezone.utc))
        self.session.add(attempt)
        self.session.commit()
        response, result, error_code = None, None, None
        try:
            run = AgentRunner().run_with_response(provider, request, ExtractionResult)
            response, result = run.response, run.result.model_dump()
            from ainovel.services.memory_extraction_merge import validate_chunk_result
            validate_chunk_result(chunk.snapshot, result)
            if ((response.input_tokens is not None and response.input_tokens > request.max_input_tokens)
                    or (response.output_tokens is not None and response.output_tokens > request.max_output_tokens)):
                error_code, result = 'memory_budget_exceeded', None
        except Exception as error:
            response = getattr(error, 'response', None) or response
            result = None
            error_code = safe_failure_code(error)
        job = self.lock(job_id)
        chunk = self.session.get(MemoryExtractionChunk, chunk.id)
        attempt = self.session.get(MemoryExtractionAttempt, attempt_id)
        auth = self.session.get(MemoryExtractionAuthorization, authorization_id)
        if attempt.status != 'RUNNING':
            raise ValueError('请求结果状态冲突')
        chunk.result, chunk.status = result, 'FAILED' if error_code else 'SUCCEEDED'
        attempt.status, attempt.error_code, attempt.ended_at = chunk.status, error_code, datetime.now(timezone.utc)
        attempt.input_tokens = response.input_tokens if response else None
        attempt.output_tokens = response.output_tokens if response else None
        job.error_code = error_code
        if job.status != 'CANCELLED':
            job.status = 'PAUSED' if error_code or auth.calls_used >= 8 else 'READY'
            if not self.current(job):
                job.status = 'STALE'
            elif all(c.status in ('SUCCEEDED', 'REUSED') for c in self.chunks(job_id)):
                job.status = 'NEEDS_REVIEW'
        if job.status != 'READY':
            auth.status = 'CLOSED'
        job.revision += 1
        ProjectLLMGuard.release(self.session, job.project_id, attempt_id)
        self.session.commit()
        return job

    def cancel(self, job_id, expected_revision, actor):
        job = self.lock(job_id)
        if job.revision != expected_revision or job.status == 'MERGED':
            raise ValueError('任务版本已变化')
        job.status, job.revision = 'CANCELLED', job.revision + 1
        for chunk in self.chunks(job_id):
            if chunk.status == 'PENDING':
                chunk.status = 'CANCELLED'
        self.audit(job, 'memory_extraction_cancelled', actor)
        self.session.commit()

    def mark_interrupted(self, job_id, expected_revision, actor):
        job = self.lock(job_id)
        if job.revision != expected_revision or job.status not in ('RUNNING', 'CANCELLED'):
            raise ValueError('任务状态不允许核对中断')
        attempts = self.session.scalars(select(MemoryExtractionAttempt).join(MemoryExtractionChunk,
            MemoryExtractionChunk.id == MemoryExtractionAttempt.chunk_id).where(
            MemoryExtractionChunk.job_id == job.id, MemoryExtractionAttempt.status == 'RUNNING')).all()
        if not attempts or any(a.started_at is None or a.started_at >= PROCESS_STARTED for a in attempts):
            raise ValueError('本服务启动后发出的请求尚未结束，不可释放')
        for attempt in attempts:
            attempt.status = 'UNKNOWN'
            attempt.error_code = 'process_interrupted_usage_unknown'
            self.session.get(MemoryExtractionChunk, attempt.chunk_id).status = 'FAILED'
            self.session.get(MemoryExtractionAuthorization, attempt.authorization_id).status = 'CLOSED'
            ProjectLLMGuard.release(self.session, job.project_id, attempt.id)
        job.status, job.revision = ('CANCELLED' if job.status == 'CANCELLED' else 'PAUSED'), job.revision + 1
        self.audit(job, 'memory_extraction_interruption_confirmed', actor)
        self.session.commit()

    def audit(self, job, action, actor):
        self.session.add(AuditEvent(id=str(uuid4()), project_id=job.project_id, entity_type='memory_extraction',
            entity_id=job.id, action=action, actor=actor, details={'revision': job.revision}))

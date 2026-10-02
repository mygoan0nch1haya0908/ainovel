"""Same-database project dispatch exclusion; no claims about other API clients."""
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import delete, inspect, update
from ainovel.models import NovelProject, ProjectLLMClaim
from ainovel.providers.contracts import ProviderUnavailable

PROCESS_STARTED = datetime.now(timezone.utc)


class ProjectLLMGuard:
    @staticmethod
    def claim(session, project_id, owner_id):
        session.execute(update(NovelProject).where(NovelProject.id == project_id).values(title=NovelProject.title))
        session.expire_all()
        if session.get(NovelProject, project_id) is None:
            raise ValueError('project not found')
        if session.get(ProjectLLMClaim, project_id) is not None:
            raise ProviderUnavailable('本项目已有模型调用或尚未核对的中断请求')
        session.add(ProjectLLMClaim(project_id=project_id, owner_id=owner_id))
        session.flush()

    @staticmethod
    def release(session, project_id, owner_id):
        session.execute(delete(ProjectLLMClaim).where(ProjectLLMClaim.project_id == project_id,
                                                     ProjectLLMClaim.owner_id == owner_id))
        session.flush()

    @staticmethod
    def reconcile(session, project_id, owner_id, actor):
        """Author confirmed all old workers stopped; never called automatically."""
        from ainovel.models import (StageModelAttempt, StageRoadmapVersion, StoryStage,
                                    ModelAttempt, WorkflowStep, GenerationWorkflow, AuditEvent)
        session.execute(update(NovelProject).where(NovelProject.id == project_id).values(title=NovelProject.title))
        session.expire_all()
        claim = session.get(ProjectLLMClaim, project_id)
        if claim is None or claim.owner_id != owner_id:
            raise ValueError('请求占用已变化，请刷新')
        created = claim.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created >= PROCESS_STARTED:
            raise ValueError('本服务启动后发出的请求不能标记为旧进程中断')
        code = 'process_interrupted_usage_unknown'
        stage_attempt = session.get(StageModelAttempt, owner_id)
        attempt = session.get(ModelAttempt, owner_id)
        if stage_attempt is not None:
            roadmap = session.get(StageRoadmapVersion, stage_attempt.roadmap_id)
            stage = session.get(StoryStage, roadmap.stage_id)
            if stage.project_id != project_id or stage_attempt.status != 'RUNNING':
                raise ValueError('规划请求状态已变化')
            stage_attempt.status, stage_attempt.error_code = 'PAUSED_PROVIDER', code
            stage_attempt.input_tokens = stage_attempt.output_tokens = None
            roadmap.actual_input_tokens = roadmap.actual_output_tokens = None
            if roadmap.status == 'RUNNING':
                roadmap.status = 'PAUSED_PROVIDER'
        elif attempt is not None:
            step = session.get(WorkflowStep, attempt.step_id)
            workflow = session.get(GenerationWorkflow, step.workflow_id)
            if workflow.project_id != project_id or not (attempt.status == 'RUNNING' or attempt.error_code == 'lease_expired'):
                raise ValueError('写作请求状态已变化')
            attempt.status, attempt.error_code = 'FAILED', code
            attempt.input_tokens = attempt.output_tokens = None
            attempt.error_detail = '作者确认旧服务已停止；供应商用量未知，未自动重试'
            if step.active_artifact_id is None:
                step.status = 'PAUSED'
                step.lease_owner = step.lease_expires_at = None
                step.revision += 1
            if workflow.status not in ('COMPLETED', 'CANCELLED', 'REJECTED', 'FAILED'):
                workflow.status = 'PAUSED_PROVIDER'
                workflow.last_error_code, workflow.last_error_detail = code, attempt.error_detail
                workflow.revision += 1
        else:
            raise ValueError('此占用不是规划或写作请求；请在对应整理任务核对')
        ProjectLLMGuard.release(session, project_id, owner_id)
        session.add(AuditEvent(id=str(uuid4()), project_id=project_id, entity_type='project_llm_claim',
            entity_id=owner_id, action='project_llm_interruption_confirmed', actor=actor,
            details={'usage': None, 'automatic_retry': False}))
        session.flush()


@contextmanager
def project_call(session_factory, project_id, owner_id):
    with session_factory() as session:
        # Older-schema migration tests retain their historical provider path.
        enabled = inspect(session.bind).has_table('project_llm_claims')
        if enabled:
            ProjectLLMGuard.claim(session, project_id, owner_id)
            session.commit()
    try:
        yield
    except BaseException as error:
        if not isinstance(error, Exception):
            raise  # process interruption retains durable uncertainty
        if enabled:
            with session_factory() as session:
                ProjectLLMGuard.release(session, project_id, owner_id)
                session.commit()
        raise
    else:
        if enabled:
            with session_factory() as session:
                ProjectLLMGuard.release(session, project_id, owner_id)
                session.commit()

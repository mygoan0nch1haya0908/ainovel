"""Explicit, compare-and-swap context conversion; never resumes or calls a model."""
from uuid import uuid4

from sqlalchemy import select, update

from ainovel.models import (AuditEvent, GenerationWorkflow, MemoryCardVersion, ModelAttempt,
                            NovelProject, WorkflowArtifact, WorkflowContextPolicy, WorkflowStep)
from ainovel.services.scoped_context import ScopedContextService, active_policy
from ainovel.services.story_memory import digest


class ContextPolicyService:
    def __init__(self, session):
        self.session = session

    def _idle(self, workflow_id):
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        if workflow is None:
            raise ValueError('workflow not found')
        project = self.session.get(NovelProject, workflow.project_id)
        if project.active_workflow_id != workflow_id or workflow.status not in ('PAUSED_CONTEXT_OVERFLOW', 'PLANNING'):
            raise ValueError('只能转换尚未调用的规划任务或输入超限暂停任务；请保留原记录，取消后从阶段页创建下一批。')
        steps = self.session.scalars(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow_id)).all()
        if (workflow.model_calls_used or workflow.candidate_batch_id or len(steps) != 1
                or steps[0].kind != 'PLANNING' or steps[0].attempt_count or steps[0].active_artifact_id
                or steps[0].lease_owner or steps[0].lease_expires_at or steps[0].status not in ('PAUSED', 'PENDING')
                or self.session.scalar(select(WorkflowArtifact.id).where(WorkflowArtifact.workflow_id == workflow_id).limit(1))
                or self.session.scalar(select(ModelAttempt.id).join(WorkflowStep, WorkflowStep.id == ModelAttempt.step_id).where(WorkflowStep.workflow_id == workflow_id).limit(1))):
            raise ValueError('已有模型请求、产物或运行租约，不能原地转换；请保留原记录，取消后从阶段页创建下一批。')
        return workflow, steps[0]

    def activate(self, workflow_id, card_id, expected_preview_fingerprint, actor):
        # Acquire the workflow write lock before re-reading preview dependencies.
        self.session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(revision=GenerationWorkflow.revision))
        self.session.expire_all()
        workflow, step = self._idle(workflow_id)
        service = ScopedContextService(self.session)
        preview = service.preview(workflow_id, step.id, card_id=card_id)
        if preview.preview_fingerprint != expected_preview_fingerprint:
            raise ValueError('预览已过期，请重新预览后确认')
        if preview.blockers:
            raise ValueError('; '.join(preview.blockers))
        previous = active_policy(self.session, workflow_id)
        card = self.session.get(MemoryCardVersion, card_id)
        return self._append(workflow, previous, 'scoped_story_v1', card_id,
            {'fingerprint': preview.source_fingerprint, 'card_hash': digest(card.entries)}, preview.preview_fingerprint, actor)

    def revert(self, workflow_id, expected_policy_version, actor):
        self.session.execute(update(GenerationWorkflow).where(GenerationWorkflow.id == workflow_id).values(revision=GenerationWorkflow.revision))
        self.session.expire_all()
        workflow, step = self._idle(workflow_id)
        previous = active_policy(self.session, workflow_id)
        if previous is None or previous.strategy != 'scoped_story_v1' or previous.version_number != expected_policy_version:
            raise ValueError('context policy revision conflict')
        self._append(workflow, previous, 'legacy', None, {}, digest({'revert': previous.id}), actor)

    def _append(self, workflow, previous, strategy, card_id, sources, preview_fingerprint, actor):
        revision = workflow.revision
        changed = self.session.execute(update(GenerationWorkflow).where(
            GenerationWorkflow.id == workflow.id, GenerationWorkflow.revision == revision,
            GenerationWorkflow.model_calls_used == 0).values(revision=revision + 1))
        if changed.rowcount != 1:
            raise ValueError('workflow context conversion conflict')
        if previous:
            previous.active = False
        policy = WorkflowContextPolicy(id=str(uuid4()), workflow_id=workflow.id,
            version_number=previous.version_number + 1 if previous else 1, strategy=strategy, card_id=card_id,
            source_versions=sources, previous_policy_id=previous.id if previous else None,
            preview_fingerprint=preview_fingerprint, active=True)
        self.session.add(policy)
        self.session.add(AuditEvent(id=str(uuid4()), project_id=workflow.project_id, entity_type='workflow',
            entity_id=workflow.id, action='context_policy_activated' if strategy != 'legacy' else 'context_policy_reverted',
            actor=actor, details={'policy_id': policy.id, 'strategy': strategy, 'previous_policy_id': policy.previous_policy_id,
                                  'previous_workflow_revision': revision, 'card_id': card_id}))
        self.session.flush()
        return policy

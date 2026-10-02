from uuid import uuid4
import pytest
from sqlalchemy import select
from test_orchestrator import ready_project
from test_model_profiles import MemoryVault
from ainovel.services.model_profiles import ModelProfileService, ProfileInput
from ainovel.services.projects import ProjectService
from ainovel.providers.contracts import ModelResponse, ProviderCapabilities, ProviderTimeout
from ainovel.models import MemoryExtractionAttempt, MemoryExtractionChunk, MemoryExtractionAuthorization


class ExtractionProvider:
    def __init__(self, action=None):
        self.requests = []
        self.action = action

    def capabilities(self, model):
        return ProviderCapabilities(140000, 64000, True, False, True, True)

    def generate(self, request):
        self.requests.append(request)
        if self.action:
            self.action()
        s = request.input_payload['source']
        return ModelResponse(dict(entries=[dict(kind='rule', text='精简约束', references=[
            dict(source_id=s['source_id'], start=s['start'], end=s['end'])])], unresolved=[]),
            None, 'test', 100, 50, 1)


def extraction_fixture(session, project):
    from ainovel.services.memory_extraction import MemoryExtractionService
    ProjectService(session).add_constitution(project.id, {'rule': '\n\n'.join(f'规则{i}' for i in range(9))}, True)
    profile = ModelProfileService(session, vault=MemoryVault()).create(
        ProfileInput('test', 'https://example.com/v1', 'remote', 'test', 140000, 32000), api_key='fake')
    service = MemoryExtractionService(session)
    job = service.prepare(project.id, profile.version_id, None, [], 'author')
    auth = str(uuid4())
    service.authorize(job.id, job.revision, auth, 'author')
    return service, job, auth


def test_nine_chunks_only_eight_calls(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    provider = ExtractionProvider()
    for _ in range(8):
        service.run_next(job.id, auth, provider)
    with pytest.raises(ValueError):
        service.run_next(job.id, auth, provider)
    assert len(provider.requests) == 8
    assert session.get(MemoryExtractionAuthorization, auth).calls_used == 8


def test_replayed_authorization_no_extra_calls(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    service.authorize(job.id, 1, auth, 'author')
    assert len(session.scalars(select(MemoryExtractionAuthorization)).all()) == 1


def test_timeout_keeps_unknown_usage_and_no_retry(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    def timeout():
        raise ProviderTimeout(source='client', phase='response_headers')
    provider = ExtractionProvider(timeout)
    service.run_next(job.id, auth, provider)
    with pytest.raises(ValueError):
        service.run_next(job.id, auth, provider)
    attempt = session.scalar(select(MemoryExtractionAttempt))
    assert attempt.input_tokens is None and attempt.output_tokens is None
    assert len(provider.requests) == 1


def test_crash_does_not_resend_running_chunk(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    def crash():
        raise KeyboardInterrupt()
    provider = ExtractionProvider(crash)
    with pytest.raises(KeyboardInterrupt):
        service.run_next(job.id, auth, provider)
    with pytest.raises(ValueError):
        service.run_next(job.id, auth, provider)
    assert len(provider.requests) == 1


def test_revoked_profile_no_dispatch(session, ready_project):
    from ainovel.models import ModelProfileVersion
    service, job, auth = extraction_fixture(session, ready_project)
    session.get(ModelProfileVersion, job.model_profile_version_id).revoked = True
    session.commit()
    provider = ExtractionProvider()
    with pytest.raises(ValueError):
        service.run_next(job.id, auth, provider)
    assert provider.requests == []


def test_same_project_writing_and_extract_exclusive(session, ready_project):
    from sqlalchemy.orm import Session
    from ainovel.services.project_llm_guard import ProjectLLMGuard
    from ainovel.providers.contracts import ProviderUnavailable
    ProjectLLMGuard.claim(session, ready_project.id, 'writing')
    session.commit()
    with Session(session.bind) as second:
        with pytest.raises(ProviderUnavailable):
            ProjectLLMGuard.claim(second, ready_project.id, 'extraction')


def test_actual_usage_overrun_pauses(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    class Overrun(ExtractionProvider):
        def generate(self, request):
            from dataclasses import replace
            return replace(super().generate(request), output_tokens=64001)
    service.run_next(job.id, auth, Overrun())
    assert job.status == 'PAUSED'
    assert session.scalar(select(MemoryExtractionAttempt)).output_tokens == 64001


def test_cancel_preserves_inflight_result(session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    provider = ExtractionProvider(lambda: service.cancel(job.id, job.revision, 'author'))
    service.run_next(job.id, auth, provider)
    assert job.status == 'CANCELLED'
    assert session.scalar(select(MemoryExtractionChunk).where(MemoryExtractionChunk.status == 'SUCCEEDED')).result


def test_stage_does_not_dispatch_while_extraction_claimed(session, ready_project):
    from ainovel.services.project_llm_guard import ProjectLLMGuard
    from ainovel.services.stages import StageService
    from ainovel.providers.fake import FakeProvider
    from test_orchestrator import response
    from test_plot_point_planning import plot_payload
    stages = StageService(session)
    stage = stages.create(ready_project.id, 'test')
    proposal = stages.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    ProjectLLMGuard.claim(session, ready_project.id, 'extraction')
    session.commit()
    provider = FakeProvider([response(plot_payload(), 1)])
    stages.generate_roadmap(proposal.id, provider)
    assert proposal.status == 'PAUSED_PROVIDER'


def test_cancelled_crash_can_reconcile_without_resuming(session, ready_project, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from ainovel.models import ProjectLLMClaim
    service, job, auth = extraction_fixture(session, ready_project)
    def crash():
        service.cancel(job.id, job.revision, 'author')
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        service.run_next(job.id, auth, ExtractionProvider(crash))
    monkeypatch.setattr('ainovel.services.memory_extraction.PROCESS_STARTED', datetime.now(timezone.utc) + timedelta(seconds=1))
    service.mark_interrupted(job.id, job.revision, 'author')
    assert job.status == 'CANCELLED'
    assert session.get(ProjectLLMClaim, ready_project.id) is None
    attempt = session.scalar(select(MemoryExtractionAttempt))
    assert attempt.status == 'UNKNOWN' and attempt.input_tokens is None


def test_stage_crash_claim_has_explicit_audited_recovery(session, ready_project, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from ainovel.services.project_llm_guard import ProjectLLMGuard
    from ainovel.services.stages import StageService
    from ainovel.models import ProjectLLMClaim, StageModelAttempt, AuditEvent
    stages = StageService(session)
    stage = stages.create(ready_project.id, 'test')
    proposal = stages.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    def crash():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        stages.generate_roadmap(proposal.id, ExtractionProvider(crash))
    claim = session.get(ProjectLLMClaim, ready_project.id)
    owner = claim.owner_id
    with pytest.raises(ValueError):
        ProjectLLMGuard.reconcile(session, ready_project.id, owner, 'author')
    monkeypatch.setattr('ainovel.services.project_llm_guard.PROCESS_STARTED', datetime.now(timezone.utc) + timedelta(seconds=1))
    ProjectLLMGuard.reconcile(session, ready_project.id, owner, 'author')
    session.commit()
    assert session.get(ProjectLLMClaim, ready_project.id) is None
    assert proposal.status == 'PAUSED_PROVIDER'
    attempt = session.get(StageModelAttempt, owner)
    assert attempt.input_tokens is None and attempt.error_code == 'process_interrupted_usage_unknown'
    assert session.scalar(select(AuditEvent).where(AuditEvent.action == 'project_llm_interruption_confirmed'))


def test_writing_claim_recovery_pauses_and_preserves_unknown_usage(session, ready_project, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from ainovel.services.project_llm_guard import ProjectLLMGuard
    from ainovel.services.workflows import WorkflowService, DEFAULT_BUDGETS
    from ainovel.models import ModelAttempt, WorkflowStep, ProjectLLMClaim
    workflow = WorkflowService(session).start(ready_project.id, 'fake', 'demo', 1, DEFAULT_BUDGETS)
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    step.status = 'RUNNING'
    attempt = ModelAttempt(id=str(uuid4()), step_id=step.id, attempt_number=1,
        status='RUNNING', request_digest='a' * 64)
    session.add(attempt)
    session.flush()
    ProjectLLMGuard.claim(session, ready_project.id, attempt.id)
    session.commit()
    monkeypatch.setattr('ainovel.services.project_llm_guard.PROCESS_STARTED', datetime.now(timezone.utc) + timedelta(seconds=1))
    ProjectLLMGuard.reconcile(session, ready_project.id, attempt.id, 'author')
    session.commit()
    assert workflow.status == 'PAUSED_PROVIDER' and step.status == 'PAUSED'
    assert attempt.input_tokens is None and attempt.output_tokens is None
    assert session.get(ProjectLLMClaim, ready_project.id) is None

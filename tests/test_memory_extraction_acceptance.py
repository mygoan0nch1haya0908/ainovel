from copy import deepcopy
from uuid import uuid4
from test_orchestrator import ready_project
from test_plot_point_stages import proposed_plot
from test_model_profiles import MemoryVault
from test_memory_extraction_execution import ExtractionProvider
from ainovel.services.projects import ProjectService
from ainovel.services.model_profiles import ModelProfileService, ProfileInput
from ainovel.services.memory_extraction import MemoryExtractionService
from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
from ainovel.services.story_memory import StoryMemoryService
from ainovel.services.scoped_context import ScopedContextService
from ainovel.models import WorkflowStep
from sqlalchemy import select


def test_long_sources_to_author_approved_scoped_memory(session, ready_project):
    from ainovel.services.stages import StageService, StageBudgets
    from ainovel.providers.fake import FakeProvider
    from test_orchestrator import response
    from test_plot_point_planning import plot_payload
    ProjectService(session).add_constitution(ready_project.id, {'rule': '设定原文，不得违背。' * 7000}, True)
    stages = StageService(session)
    stage = stages.create(ready_project.id, '调查')
    roadmap = stages.propose_roadmap(stage.id, 'author', 'fake', 'demo',
        budgets=StageBudgets(input_tokens=100000, total_input_tokens=200000))
    stages.generate_roadmap(roadmap.id, FakeProvider([response(plot_payload(), 1)]))
    stages.approve_roadmap(stage.id, roadmap.id, 'author')
    workflow = stages.start_next_batch(stage.id, 'author', 'fake', 'demo', 1).workflow
    profile = ModelProfileService(session, vault=MemoryVault()).create(
        ProfileInput('test', 'https://example.com/v1', 'remote', 'test', 140000, 64000), api_key='fake')
    service = MemoryExtractionService(session)
    job = service.prepare(ready_project.id, profile.version_id, None, [], 'author')
    assert len(service.chunks(job.id)) > 1
    provider = ExtractionProvider()
    while job.status in ('DRAFT', 'PAUSED'):
        auth = str(uuid4())
        service.authorize(job.id, job.revision, auth, 'author')
        for _ in range(8):
            service.run_next(job.id, auth, provider)
            if job.status != 'READY':
                break
    assert job.status == 'NEEDS_REVIEW'
    # Same sources/profile/base reuse success without any network dispatch.
    cached = service.prepare(ready_project.id, profile.version_id, None, [], 'author')
    assert all(c.status == 'REUSED' for c in service.chunks(cached.id))
    merge = MemoryExtractionMergeService(session)
    preview = merge.preview(job.id)
    edited = deepcopy(preview['entries'])
    edited[0]['text'], edited[0]['author_locked'] = '作者确认的全局规则', True
    card = merge.merge(job.id, job.revision, preview['fingerprint'], edited, 'author')
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    assert ScopedContextService(session).select(workflow.id, step.id, card_id=card.id).missing
    memory = StoryMemoryService(session)
    memory.approve_card(card.id, card.source_fingerprint, 'author')
    session.commit()
    assert ScopedContextService(session).select(workflow.id, step.id, card_id=card.id).missing == []
    # Manual editing must present stable IDs instead of null identifiers.
    assert all(e['entry_id'] for e in card.entries)


def test_changed_constitution_reuses_unmodified_stage_chunks(session, ready_project):
    from test_memory_extraction_execution import extraction_fixture
    stages, stage, roadmap = proposed_plot(session, ready_project)
    stages.approve_roadmap(stage.id, roadmap.id, 'author')
    service, job, auth = extraction_fixture(session, ready_project)
    provider = ExtractionProvider()
    while job.status == 'READY':
        service.run_next(job.id, auth, provider)
        if job.status == 'PAUSED':
            auth = str(uuid4())
            service.authorize(job.id, job.revision, auth, 'author')
    assert job.status == 'NEEDS_REVIEW'
    ProjectService(session).add_constitution(ready_project.id, {'rule': '新设定，阶段原文未改'}, True)
    updated = service.prepare(ready_project.id, job.model_profile_version_id, None, [], 'author')
    chunks = service.chunks(updated.id)
    assert any(c.status == 'REUSED' for c in chunks)
    assert all(c.status == 'REUSED' for c in chunks if c.snapshot['source']['ref']['source_type'] != 'constitution')
    expected_calls = sum(c.status == 'PENDING' for c in chunks)
    calls_before = len(provider.requests)
    auth = str(uuid4())
    service.authorize(updated.id, updated.revision, auth, 'author')
    while updated.status == 'READY':
        service.run_next(updated.id, auth, provider)
    assert len(provider.requests) - calls_before == expected_calls == 1

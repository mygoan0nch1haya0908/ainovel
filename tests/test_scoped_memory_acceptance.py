from sqlalchemy import select
from test_orchestrator import ready_project, session_factory, clock, make_orchestrator
from test_scoped_context_selection import scoped_fixture
from ainovel.models import StoryMemoryEntry, StoryStage, WorkflowStep, WorkflowArtifact
from ainovel.providers.demo import DemoFakeProvider
from ainovel.services.context_policy import ContextPolicyService
from ainovel.services.scoped_context import ScopedContextService
from ainovel.services.workflows import WorkflowService
from ainovel.services.batches import BatchService
from ainovel.services.stages import StageService


class CountingDemo(DemoFakeProvider):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


def test_five_chapter_batch_is_scoped_and_promotes_only_after_approval(session, ready_project, session_factory, clock):
    workflow, step, card, version = scoped_fixture(session, ready_project, 5)
    preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=card.id)
    ContextPolicyService(session).activate(workflow.id, card.id, preview.preview_fingerprint, 'author')
    session.commit()
    provider = CountingDemo()
    orchestrator = make_orchestrator(session_factory, provider, clock)
    assert orchestrator.advance(workflow.id).status == 'AWAITING_PLAN_APPROVAL'
    assert len(provider.requests) == 1
    WorkflowService(session, clock=clock).approve_plan(workflow.id, 'author')
    result = orchestrator.run_until_blocked(workflow.id)
    assert result.status == 'AWAITING_CONTENT_APPROVAL'
    # Planner + (writer, coverage, summarizer) * five + batch reviewer.
    assert len(provider.requests) == 17
    assert session.scalar(select(StoryMemoryEntry)) is None
    writing = [r for r in provider.requests if r.metadata['agent_role'] == 'chapter_writer']
    assert all(len(r.input_payload['stage']['points']) == 1 for r in writing)
    assert all('roadmap' not in r.input_payload['stage'] for r in provider.requests)
    assert all('project_constitution' not in r.input_payload for r in provider.requests)
    for ordinal, request in enumerate(writing, 1):
        prior = [i for i in request.input_payload['context_packet']['items'] if i['kind'] == 'candidate_summary']
        assert len(prior) == ordinal - 1
        assert all(i['approved'] is False for i in prior)
    BatchService(session).approve(result.candidate_batch_id, workflow.base_outline_version_id)
    session.expire_all()
    assert session.get(StoryStage, version.stage_id).confirmed_chapters == 5
    formal = session.scalars(select(StoryMemoryEntry).where(StoryMemoryEntry.state_scope == 'official')).all()
    assert len(formal) == 5 and all(e.kind == 'summary' for e in formal)
    assert session.scalars(select(StoryMemoryEntry).where(StoryMemoryEntry.state_scope.like('pending:%'))).all()
    # Publication changes metadata, not the approved source body.
    chapter = BatchService(session).list_chapters(result.candidate_batch_id)[0]
    BatchService(session).publish_chapter(chapter.id)
    WorkflowService(session, clock=clock).reconcile_batch_decision(workflow.id)
    next_workflow = StageService(session).start_next_batch(version.stage_id, 'author', 'fake', 'demo', 1).workflow
    next_step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == next_workflow.id))
    next_preview = ScopedContextService(session).preview(next_workflow.id, next_step.id, card_id=card.id)
    assert any('待分类' in b for b in next_preview.blockers)

import pytest

from test_orchestrator import ready_project, session_factory, clock, make_orchestrator
from test_plot_point_stages import proposed_plot
from ainovel.providers.demo import DemoFakeProvider
from ainovel.services.workflows import WorkflowService
from ainovel.services.batches import BatchService


def test_plot_rolling_five_then_one_approval(session, ready_project, session_factory, clock):
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    for previous, total, slots in [(0,5,['p1:1','p1:2','p2:1','p2:2','p2:3']), (5,6,['p2:4'])]:
        started = service.start_next_batch(stage.id, 'author', 'fake', 'demo')
        assert [n.node_id for n in started.nodes] == slots
        orchestrator = make_orchestrator(session_factory, DemoFakeProvider(), clock)
        assert orchestrator.advance(started.workflow.id).status == 'AWAITING_PLAN_APPROVAL'
        assert orchestrator.advance(started.workflow.id).status == 'AWAITING_PLAN_APPROVAL'
        assert service.get(stage.id).confirmed_chapters == previous
        WorkflowService(session, clock=clock).approve_plan(started.workflow.id, 'author')
        result = orchestrator.run_until_blocked(started.workflow.id, max_steps=50)
        assert result.status == 'AWAITING_CONTENT_APPROVAL'
        assert service.get(stage.id).confirmed_chapters == previous
        BatchService(session).approve(result.candidate_batch_id, ready_project.official_outline_version_id)
        assert service.get(stage.id).confirmed_chapters == total
        with pytest.raises(ValueError):
            BatchService(session).approve(result.candidate_batch_id, ready_project.official_outline_version_id)
        assert service.get(stage.id).confirmed_chapters == total
        WorkflowService(session, clock=clock).reconcile_batch_decision(started.workflow.id)
    with pytest.raises(ValueError, match='complete'):
        service.start_next_batch(stage.id, 'author', 'fake', 'demo')


def test_demo_new_stage_format(session, ready_project):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    proposal = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    result = service.generate_roadmap(proposal.id, DemoFakeProvider())
    assert result.status == 'PROPOSED'
    assert result.payload['format'] == 'plot_points_v1'
    assert result.estimated_chapters == 7


@pytest.mark.parametrize('decision', ['reject', 'fail'])
def test_plot_rejected_or_failed_batch_does_not_advance(session, ready_project, session_factory, clock, decision):
    from ainovel.providers.fake import FakeProvider
    from test_orchestrator import response
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    started = service.start_next_batch(stage.id, 'author', 'fake', 'demo', 1)
    if decision == 'reject':
        orchestrator = make_orchestrator(session_factory, DemoFakeProvider(), clock)
        assert orchestrator.advance(started.workflow.id).status == 'AWAITING_PLAN_APPROVAL'
        WorkflowService(session, clock=clock).approve_plan(started.workflow.id, 'author')
        result = orchestrator.run_until_blocked(started.workflow.id, max_steps=30)
        assert result.status == 'AWAITING_CONTENT_APPROVAL'
        BatchService(session).reject(result.candidate_batch_id, '重写')
        WorkflowService(session, clock=clock).reconcile_batch_decision(started.workflow.id)
    else:
        provider = FakeProvider([response({'wrong':'plan'}, 1), response({'wrong':'plan'}, 2)])
        result = make_orchestrator(session_factory, provider, clock).run_until_blocked(started.workflow.id)
        assert result.status == 'PAUSED_ATTEMPTS'
        WorkflowService(session, clock=clock).cancel(started.workflow.id, 'author')
    assert service.get(stage.id).confirmed_chapters == 0
    again = service.start_next_batch(stage.id, 'author', 'fake', 'demo', 1)
    assert [(n.node_id,n.stage_ordinal,n.book_ordinal) for n in again.nodes] == [('p1:1',1,1)]

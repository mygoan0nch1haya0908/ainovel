from copy import deepcopy

import pytest

from test_orchestrator import ready_project, response, session_factory, clock, make_orchestrator
from test_plot_point_planning import plot_payload
from test_stages import roadmap_payload
from ainovel.services.stages import StageService
from ainovel.providers.fake import FakeProvider


def proposed_plot(session, project, counts=(2, 4)):
    service = StageService(session)
    stage = service.create(project.id, '按剧情点安排调查与揭密')
    version = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    service.generate_roadmap(version.id, FakeProvider([response(plot_payload(counts), 1)]))
    assert version.status == 'PROPOSED'
    return service, stage, version


def test_new_proposal_freezes_plot_schema_without_call(session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    version = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    assert 'points' in version.prompt_snapshot['schema']['properties']
    assert version.status == 'PENDING' and version.attempts_used == 0


def test_plot_revision_locks_started_point_and_preserves_source(session, ready_project):
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    stage.confirmed_chapters = 1
    session.commit()
    revised = service.revise_roadmap(stage.id, version.id, '调整后续')
    assert revised.attempts_used == 0
    changed = deepcopy(version.payload)
    changed['points'][0]['goal'] = '改动已开始点'
    service.generate_roadmap(revised.id, FakeProvider([response(changed, 2)]))
    with pytest.raises(ValueError, match='started plot'):
        service.approve_roadmap(stage.id, revised.id, 'author')
    revised = service.revise_roadmap(stage.id, version.id, '只改未开始点')
    changed = deepcopy(version.payload)
    changed['points'][1]['chapter_count'] = 6
    service.generate_roadmap(revised.id, FakeProvider([response(changed, 3)]))
    service.approve_roadmap(stage.id, revised.id, 'author')
    assert revised.estimated_chapters == 8
    assert version.payload == plot_payload()


def test_legacy_frozen_request_and_feedback_stay_legacy(session, ready_project):
    from ainovel.agents.stage_contracts import StageRoadmapDraft
    service = StageService(session)
    stage = service.create(ready_project.id, '旧阶段')
    version = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    version.prompt_snapshot = {'body': '旧提示词', 'schema': StageRoadmapDraft.model_json_schema()}
    session.commit()
    service.generate_roadmap(version.id, FakeProvider([response(roadmap_payload(2), 1)]))
    assert version.status == 'PROPOSED'
    revision = service.revise_roadmap(stage.id, version.id, '调整旧版')
    assert 'nodes' in revision.prompt_snapshot['schema']['properties']
    service.generate_roadmap(revision.id, FakeProvider([response(roadmap_payload(2), 2)]))
    service.approve_roadmap(stage.id, revision.id, 'author')
    stage.confirmed_chapters = 1
    session.commit()
    with pytest.raises(ValueError, match='format'):
        service.propose_roadmap(stage.id, 'author', 'fake', 'demo')


def test_zero_confirmed_legacy_can_create_plot_proposal(session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '阶段')
    old = service.propose_roadmap(stage.id, 'author', 'fake', 'demo', roadmap_format='legacy')
    service.generate_roadmap(old.id, FakeProvider([response(roadmap_payload(2), 1)]))
    service.approve_roadmap(stage.id, old.id, 'author')
    new = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    service.generate_roadmap(new.id, FakeProvider([response(plot_payload(), 2)]))
    service.approve_roadmap(stage.id, new.id, 'author')
    assert old.payload == roadmap_payload(2)
    assert service.get(stage.id).approved_roadmap_id == new.id


def test_plot_batch_reserves_cross_point_slots_and_scopes_writer(session, ready_project):
    service, stage, version = proposed_plot(session, ready_project)
    with pytest.raises(ValueError, match='approved roadmap'):
        service.start_next_batch(stage.id, 'author', 'fake', 'demo')
    service.approve_roadmap(stage.id, version.id, 'author')
    started = service.start_next_batch(stage.id, 'author', 'fake', 'demo')
    assert [n.node_id for n in started.nodes] == ['p1:1','p1:2','p2:1','p2:2','p2:3']
    assert [n.stage_ordinal for n in started.nodes] == [1,2,3,4,5]
    assert [n.book_ordinal for n in started.nodes] == [1,2,3,4,5]
    assert stage.confirmed_chapters == 0
    context = service.workflow_context(started.workflow.id, include_roadmap=True)
    assert len(context['roadmap']['points']) == 2
    writer = service.workflow_context(started.workflow.id, 1)
    assert 'roadmap' not in writer
    assert [p['point_id'] for p in writer['points']] == ['p1']
    assert len(writer['slots']) == 1
    with pytest.raises(ValueError):
        service.revise_roadmap(stage.id, version.id, '不能在写作中修改')
    assert service.workflow_context(started.workflow.id, 1) == writer


def test_plot_final_short_batch(session, ready_project):
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    stage.confirmed_chapters = 5
    ready_project.next_official_chapter_number = 6
    session.commit()
    started = service.start_next_batch(stage.id, 'author', 'fake', 'demo')
    assert started.workflow.requested_chapters == 1
    assert [(n.node_id,n.stage_ordinal,n.book_ordinal) for n in started.nodes] == [('p2:4',6,6)]


def plot_plan(count=2):
    return {'chapters': [dict(ordinal=i, title=f'不同标题{i}', goal=f'局部目标{i}', ending_hook=f'钩子{i}',
                              slot_id=f'p1:{i}', point_id='p1', point_ordinal=i,
                              scenes=[dict(ordinal=1, description='细化行动', target_characters=5200)])
                         for i in range(1,count+1)]}


def test_plot_plan_refinement_gate_and_writer_context(session, ready_project, session_factory, clock):
    from ainovel.services.workflows import WorkflowService
    from ainovel.models import WorkflowStep
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    workflow = service.start_next_batch(stage.id, 'author', 'fake', 'demo', 2).workflow
    provider = FakeProvider([response(plot_plan(), 1)])
    orchestrator = make_orchestrator(session_factory, provider, clock)
    assert orchestrator.advance(workflow.id).status == 'AWAITING_PLAN_APPROVAL'
    assert len(provider.requests) == 1
    assert orchestrator.advance(workflow.id).status == 'AWAITING_PLAN_APPROVAL'
    assert len(provider.requests) == 1
    WorkflowService(session, clock=clock).approve_plan(workflow.id, 'author')
    writer = session.query(WorkflowStep).filter_by(workflow_id=workflow.id, kind='WRITING', ordinal=1).one()
    request = orchestrator._build_request(writer)
    assert request.input_payload['chapter_plan']['goal'] == '局部目标1'
    assert 'roadmap' not in request.input_payload['stage']
    assert len(request.input_payload['stage']['slots']) == 1
    assert stage.confirmed_chapters == 0


@pytest.mark.parametrize('field,value', [('slot_id','p2:1'),('point_id','p2'),('point_ordinal',2),('ordinal',2)])
def test_wrong_slot_rejected_at_artifact_boundary(session, ready_project, field, value):
    from ainovel.services.workflows import WorkflowService
    from ainovel.models import WorkflowStep
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    workflow = service.start_next_batch(stage.id, 'author', 'fake', 'demo', 2).workflow
    step = session.query(WorkflowStep).filter_by(workflow_id=workflow.id, kind='PLANNING').one()
    valid = plot_plan()
    assert WorkflowService(session)._artifact_values(workflow, step, valid).payload['chapters'][0]['goal'] == '局部目标1'
    bad = deepcopy(valid)
    bad['chapters'][0][field] = value
    with pytest.raises(ValueError):
        WorkflowService(session)._artifact_values(workflow, step, bad)


def test_plot_plan_wrong_count_and_scene_budget_rejected(session, ready_project):
    from ainovel.services.workflows import WorkflowService
    from ainovel.models import WorkflowStep
    service, stage, version = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, version.id, 'author')
    workflow = service.start_next_batch(stage.id, 'author', 'fake', 'demo', 2).workflow
    step = session.query(WorkflowStep).filter_by(workflow_id=workflow.id, kind='PLANNING').one()
    with pytest.raises(ValueError):
        WorkflowService(session)._artifact_values(workflow, step, plot_plan(1))
    bad = plot_plan()
    bad['chapters'][0]['scenes'][0]['target_characters'] = 4499
    with pytest.raises(ValueError):
        WorkflowService(session)._artifact_values(workflow, step, bad)

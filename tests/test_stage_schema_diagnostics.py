import json

from test_orchestrator import ready_project, response
from test_plot_point_planning import plot_payload
from ainovel.providers.fake import FakeProvider
from ainovel.services.stages import StageService
from ainovel.models.audit import AuditEvent
from test_stage_web import provider_registry, stage_provider


def test_stage_schema_diagnostics_persist_only_safe_paths_and_codes(session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    version = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    payload = plot_payload()
    del payload['points'][0]['chapter_count']
    payload['points'][1]['chapter_count'] = 'SECRET_BAD_VALUE'
    payload['SECRET_EXTRA_FIELD'] = 'SECRET_BODY'
    service.generate_roadmap(version.id, FakeProvider([response(payload, 1)]))
    assert version.status == 'PAUSED_INVALID' and version.payload is None
    event = session.query(AuditEvent).filter_by(action='stage_schema_validation_failed').one()
    assert event.details['roadmap_id'] == version.id
    assert event.details['attempt_number'] == 1
    assert {'loc':['points',0,'chapter_count'], 'type':'missing'} in event.details['issues']
    assert {'loc':['points',1,'chapter_count'], 'type':'int_type'} in event.details['issues']
    assert 'SECRET' not in json.dumps(event.details)
    assert version.actual_output_tokens == 50


def test_schema_diagnostics_are_bounded_and_never_echo_extra_keys():
    from ainovel.agents.schema_diagnostics import safe_schema_issues
    from ainovel.agents.stage_contracts import StagePlotRoadmapDraft
    issues = [{'loc':['SECRET_EXTRA'], 'type':'SECRET_TYPE', 'input':'SECRET', 'msg':'SECRET'}] * 100
    safe = safe_schema_issues(issues, StagePlotRoadmapDraft.model_json_schema())
    assert len(safe) == 8
    assert all(i == {'loc':['unknown_field'], 'type':'validation_error'} for i in safe)


def test_stage_page_displays_matching_attempt_fields_not_failed_values(client, session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    version = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    payload = plot_payload()
    del payload['points'][0]['chapter_count']
    service.generate_roadmap(version.id, FakeProvider([response(payload, 1)]))
    page = client.get(f'/stages/{stage.id}')
    assert page.status_code == 200
    assert 'points.0.chapter_count' in page.text
    assert '缺少必填字段' in page.text
    assert '字段校验详情' in page.text

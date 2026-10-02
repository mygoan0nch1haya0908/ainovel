from dataclasses import asdict
from sqlalchemy import select
import pytest
from test_orchestrator import ready_project
from test_scoped_context_selection import scoped_fixture
from ainovel.models import WorkflowPromptSnapshot
from ainovel.services.scoped_context import ScopedContextService
from ainovel.context import RequiredContextOverflow


def test_preview_and_request_share_budget_and_no_full_blobs(session, ready_project):
    workflow, step, card, version = scoped_fixture(session, ready_project)
    service = ScopedContextService(session)
    preview = service.preview(workflow.id, step.id, card_id=card.id)
    request = service.build_request(workflow.id, step.id, card_id=card.id)
    assert preview.blockers == [] and preview.deficit == 0
    assert request.max_input_tokens == workflow.planner_input_tokens
    assert 'project_constitution' not in request.input_payload
    assert 'official_outline_tree' not in request.input_payload
    assert 'roadmap' not in request.input_payload['stage']
    assert request.input_payload['context_packet']['items']
    assert workflow.model_calls_used == 0
    assert not session.dirty


def test_mandatory_overflow_reports_deficit_without_trimming(session, ready_project):
    workflow, step, card, version = scoped_fixture(session, ready_project)
    snapshot = session.scalar(select(WorkflowPromptSnapshot).where(WorkflowPromptSnapshot.workflow_id == workflow.id, WorkflowPromptSnapshot.role == 'batch_planner'))
    snapshot.parameters = {**snapshot.parameters, 'max_input_tokens': 100}
    session.commit()
    service = ScopedContextService(session)
    preview = service.preview(workflow.id, step.id, card_id=card.id)
    assert preview.capacity == 100
    assert preview.deficit == preview.estimated_tokens - 100 > 0
    assert all(i['selected'] for i in preview.items if i['required'])
    with pytest.raises(RequiredContextOverflow):
        service.build_request(workflow.id, step.id, card_id=card.id)
    assert workflow.model_calls_used == 0

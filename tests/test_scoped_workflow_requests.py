from uuid import uuid4
from sqlalchemy import select
from test_orchestrator import ready_project, session_factory, clock, make_orchestrator
from test_scoped_context_selection import scoped_fixture
from ainovel.models import ContextPacket, WorkflowContextPolicy, WorkflowPromptSnapshot
from ainovel.providers.fake import FakeProvider


def bind_for_test(session, workflow, card):
    session.add(WorkflowContextPolicy(id=str(uuid4()), workflow_id=workflow.id, version_number=1,
        strategy='scoped_story_v1', card_id=card.id, source_versions={'fingerprint': card.source_fingerprint}, preview_fingerprint='a' * 64))
    session.commit()


def test_orchestrator_uses_scoped_packet_and_saves_provenance(session, ready_project, session_factory, clock):
    workflow, step, card, version = scoped_fixture(session, ready_project)
    bind_for_test(session, workflow, card)
    provider = FakeProvider([])
    orchestrator = make_orchestrator(session_factory, provider, clock)
    request = orchestrator._build_request(step)
    assert 'project_constitution' not in request.input_payload
    assert 'roadmap' not in request.input_payload['stage']
    assert request.input_payload['context_packet']['items']
    session.expire_all()
    assert session.scalar(select(ContextPacket).where(ContextPacket.step_id == step.id)) is not None
    assert provider.requests == []


def test_local_overflow_never_dispatches_provider(session, ready_project, session_factory, clock):
    workflow, step, card, version = scoped_fixture(session, ready_project)
    bind_for_test(session, workflow, card)
    snapshot = session.scalar(select(WorkflowPromptSnapshot).where(WorkflowPromptSnapshot.workflow_id == workflow.id, WorkflowPromptSnapshot.role == 'batch_planner'))
    snapshot.parameters = {**snapshot.parameters, 'max_input_tokens': 100}
    session.commit()
    provider = FakeProvider([])
    result = make_orchestrator(session_factory, provider, clock).advance(workflow.id)
    assert result.status == 'PAUSED_CONTEXT_OVERFLOW'
    assert provider.requests == []

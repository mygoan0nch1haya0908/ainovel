import pytest
from sqlalchemy import select
from test_orchestrator import ready_project
from test_scoped_context_selection import scoped_fixture
from ainovel.models import AuditEvent, WorkflowContextPolicy
from ainovel.services.scoped_context import ScopedContextService


def paused_fixture(session, project):
    workflow, step, card, roadmap = scoped_fixture(session, project)
    workflow.status = 'PAUSED_CONTEXT_OVERFLOW'
    workflow.last_error_code = 'required_context_overflow'
    step.status = 'PAUSED'
    session.commit()
    return workflow, step, card


def test_conversion_preserves_pause_and_old_error_and_can_revert(session, ready_project):
    from ainovel.services.context_policy import ContextPolicyService
    workflow, step, card = paused_fixture(session, ready_project)
    preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=card.id)
    service = ContextPolicyService(session)
    policy = service.activate(workflow.id, card.id, preview.preview_fingerprint, 'author')
    session.commit()
    assert workflow.status == 'PAUSED_CONTEXT_OVERFLOW'
    assert workflow.last_error_code == 'required_context_overflow'
    assert workflow.model_calls_used == 0
    assert policy.strategy == 'scoped_story_v1'
    service.revert(workflow.id, policy.version_number, 'author')
    session.commit()
    active = session.scalar(select(WorkflowContextPolicy).where(WorkflowContextPolicy.active.is_(True)))
    assert active.strategy == 'legacy'
    assert len(session.scalars(select(AuditEvent).where(AuditEvent.action.like('context_policy%'))).all()) == 2


@pytest.mark.parametrize('change', ['revision', 'attempt', 'lease'])
def test_conversion_rejects_stale_or_used_workflow(session, ready_project, change):
    from ainovel.services.context_policy import ContextPolicyService
    from datetime import datetime, timezone, timedelta
    workflow, step, card = paused_fixture(session, ready_project)
    preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=card.id)
    if change == 'revision':
        workflow.revision += 1
    elif change == 'attempt':
        workflow.model_calls_used = 1
    else:
        step.lease_owner = 'worker'
        step.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=20)
    session.commit()
    with pytest.raises(ValueError):
        ContextPolicyService(session).activate(workflow.id, card.id, preview.preview_fingerprint, 'author')
    assert session.scalar(select(WorkflowContextPolicy)) is None


def test_conversion_rejects_source_change_after_preview(session, ready_project):
    from ainovel.services.context_policy import ContextPolicyService
    from ainovel.services.projects import ProjectService
    workflow, step, card = paused_fixture(session, ready_project)
    preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=card.id)
    ProjectService(session).add_constitution(ready_project.id, {'rule': 'changed'}, True)
    with pytest.raises(ValueError):
        ContextPolicyService(session).activate(workflow.id, card.id, preview.preview_fingerprint, 'author')

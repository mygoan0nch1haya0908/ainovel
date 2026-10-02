from copy import deepcopy
from uuid import uuid4

import pytest

from test_orchestrator import ready_project
from test_context_policy import paused_fixture
from ainovel.models import StoryMemoryEntry
from ainovel.services.context_policy import ContextPolicyService
from ainovel.services.scoped_context import ScopedContextService


def test_policy_snapshot_cannot_be_rewritten(session, ready_project):
    workflow, step, card = paused_fixture(session, ready_project)
    preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=card.id)
    policy = ContextPolicyService(session).activate(workflow.id, card.id, preview.preview_fingerprint, 'author')
    session.commit()
    policy.source_versions = {'fingerprint': 'tampered'}
    with pytest.raises(ValueError, match='immutable'):
        session.flush()


def test_confirmed_pending_record_hidden_without_deleting_history(client, session, ready_project):
    workflow, step, card = paused_fixture(session, ready_project)
    ref = deepcopy(card.entries[0]['source_refs'][0])
    # A smaller range inside the approved reference is also covered.
    ref['excerpt_end'] = ref['excerpt_start'] + 1
    ident = str(uuid4())
    session.add(StoryMemoryEntry(id=ident, project_id=ready_project.id, source_key='confirmed-change',
        kind='fact', text='UNIQUE_PENDING_MARKER', source_refs=[ref], entity_ids=[], point_ids=[],
        effective_from=1, audience='author_only', state_scope='pending:batch'))
    session.commit()
    page = client.get(f'/projects/{ready_project.id}/memory')
    assert page.status_code == 200
    assert 'UNIQUE_PENDING_MARKER' not in page.text
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=card.id)
    assert not any('待分类' in reason for reason in selected.missing)
    assert session.get(StoryMemoryEntry, ident) is not None


@pytest.mark.parametrize('invalid', ['draft', 'missing', 'stale'])
def test_unconfirmed_card_does_not_clear_pending(session, ready_project, invalid):
    from ainovel.agents.memory_contracts import MemoryEntryInput
    from ainovel.services.story_memory import StoryMemoryService
    from ainovel.services.projects import ProjectService
    workflow, step, card = paused_fixture(session, ready_project)
    memory = StoryMemoryService(session)
    pending = StoryMemoryEntry(project_id=ready_project.id, source_refs=deepcopy(card.entries[0]['source_refs']))
    if invalid == 'draft':
        card = memory.create_card(ready_project.id, [MemoryEntryInput.model_validate(e) for e in card.entries], 'author')
    elif invalid == 'missing':
        card = None
    else:
        ProjectService(session).add_constitution(ready_project.id, {'rule': 'changed'}, True)
    assert not memory.pending_confirmed(pending, card)

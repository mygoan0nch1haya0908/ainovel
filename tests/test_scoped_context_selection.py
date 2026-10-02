from copy import deepcopy
import pytest
from sqlalchemy import select

from test_orchestrator import ready_project
from test_plot_point_stages import proposed_plot
from test_story_memory import all_entries
from ainovel.models import WorkflowStep
from ainovel.services.story_memory import StoryMemoryService


def scoped_fixture(session, project, count=3):
    service, stage, version = proposed_plot(session, project, (2, 4, 2))
    service.approve_roadmap(stage.id, version.id, 'author')
    workflow = service.start_next_batch(stage.id, 'author', 'fake', 'demo', count).workflow
    memory = StoryMemoryService(session)
    entries = all_entries(memory, project.id)
    card = memory.create_card(project.id, entries, 'author')
    memory.approve_card(card.id, card.source_fingerprint, 'author')
    session.commit()
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    return workflow, step, card, version


def test_selection_uses_reserved_slots_not_future_points(session, ready_project):
    from ainovel.services.scoped_context import ScopedContextService
    workflow, step, card, version = scoped_fixture(session, ready_project)
    result = ScopedContextService(session).select(workflow.id, step.id, card_id=card.id)
    assert result.missing == []
    stage = next(i for i in result.required if i['kind'] == 'stage')['value']
    assert [p['point_id'] for p in stage['points']] == ['p1', 'p2']
    assert [s['node_id'] for s in stage['slots']] == ['p1:1', 'p1:2', 'p2:1']
    assert 'roadmap' not in stage and 'outline_constraints' not in stage
    assert all(i['audience'] == 'author_only' for i in result.required if i['kind'] == 'rule')


def test_invalid_dependency_is_not_silently_ignored(session, ready_project):
    from ainovel.services.scoped_context import ScopedContextService
    workflow, step, card, version = scoped_fixture(session, ready_project)
    payload = deepcopy(version.payload)
    payload['points'][0]['dependencies'] = ['p2']
    version.payload = payload
    session.commit()
    with pytest.raises(ValueError, match='dependencies'):
        ScopedContextService(session).select(workflow.id, step.id, card_id=card.id)


def test_cross_project_card_cannot_supply_constraints(session, ready_project):
    from ainovel.services.scoped_context import ScopedContextService
    from ainovel.services.projects import ProjectService
    workflow, step, card, version = scoped_fixture(session, ready_project)
    other = ProjectService(session).create('other', 2000000, 5000000)
    wrong = StoryMemoryService(session).create_card(other.id, [], 'author')
    with pytest.raises(ValueError, match='project'):
        ScopedContextService(session).select(workflow.id, step.id, card_id=wrong.id)


def test_latest_point_summary_prevents_history_growth(session, ready_project):
    from uuid import uuid4
    from ainovel.models import StoryMemoryEntry
    from ainovel.services.scoped_context import ScopedContextService
    workflow, step, card, version = scoped_fixture(session, ready_project)
    writer = WorkflowStep(id=str(uuid4()), workflow_id=workflow.id, kind='WRITING', ordinal=3, position=1, status='PENDING')
    session.add(writer)
    for number in (1, 2, 3):
        session.add(StoryMemoryEntry(id=str(uuid4()), project_id=ready_project.id, source_key=f'summary-{number}',
            kind='summary', text=f'point-summary-{number}', source_refs=card.entries[0]['source_refs'],
            point_ids=[f'{version.id}:p2'], entity_ids=[], effective_from=number,
            audience='narratable', state_scope='official'))
    session.commit()
    selected = ScopedContextService(session).select(workflow.id, writer.id, card_id=card.id)
    summaries = [i['text'] for i in selected.required if i['kind'] == 'summary']
    assert summaries == ['point-summary-3']


def test_explicit_alias_and_secret_remain_required_without_keyword_match(session, ready_project):
    from ainovel.agents.memory_contracts import MemoryEntryInput
    from ainovel.services.scoped_context import ScopedContextService
    workflow, step, card, version = scoped_fixture(session, ready_project, 1)
    memory = StoryMemoryService(session)
    entries = [MemoryEntryInput.model_validate(e) for e in card.entries]
    entries += [MemoryEntryInput(kind='fact', text='代号夜鸦的真名不得提前公开', source_refs=entries[0].source_refs,
        point_ids=['p1'], entity_ids=['从未出现在剧情文本中的人物ID'], reveal_from=20, audience='narratable')]
    new = memory.create_card(ready_project.id, entries, 'author')
    memory.approve_card(new.id, new.source_fingerprint, 'author')
    session.commit()
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=new.id)
    secret = next(i for i in selected.required if i['kind'] == 'fact')
    assert secret['audience'] == 'author_only' and secret['reveal_from'] == 20

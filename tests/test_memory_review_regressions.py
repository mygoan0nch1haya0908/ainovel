from copy import deepcopy
from uuid import uuid4

import pytest

from test_orchestrator import ready_project
from test_scoped_context_selection import scoped_fixture
from test_story_memory import all_entries
from ainovel.agents.memory_contracts import MemoryEntryInput
from ainovel.models import StoryMemoryEntry, WorkflowStep
from ainovel.services.scoped_context import ScopedContextService
from ainovel.services.story_memory import StoryMemoryService


def revise_card(session, project, card, additions):
    service = StoryMemoryService(session)
    new = service.create_card(project.id, [MemoryEntryInput.model_validate(e) for e in card.entries] + additions, 'author')
    service.approve_card(new.id, new.source_fingerprint, 'author')
    session.commit()
    return new


def test_entity_only_memory_never_silently_disappears(session, ready_project):
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    new = revise_card(session, ready_project, card, [MemoryEntryInput(kind='character', text='HERO_INJURED',
        source_refs=card.entries[0]['source_refs'], entity_ids=['hero'])])
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=new.id)
    assert any('关联' in reason for reason in selected.missing)


def test_dependency_constraints_use_same_range_and_disclosure_rules(session, ready_project):
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    # Ask for p2 without a same-batch p1 predecessor by narrowing the fixture's
    # prior mapping to p2. This is a selector test, not an executable workflow.
    from ainovel.models import StageWorkflowNode, StageWorkflow, StoryStage
    nodes = session.query(StageWorkflowNode).filter_by(workflow_id=workflow.id).all()
    nodes[0].node_id = 'p2:1'
    nodes[0].stage_ordinal = nodes[0].book_ordinal = 3
    mapping = session.get(StageWorkflow, workflow.id)
    mapping.confirmed_start = 2
    session.get(StoryStage, mapping.stage_id).confirmed_chapters = 2
    ref = card.entries[0]['source_refs']
    new = revise_card(session, ready_project, card, [
        MemoryEntryInput(kind='summary', text='SECRET_RESULT', source_refs=ref, point_ids=['p1'], reveal_from=20, audience='narratable'),
        MemoryEntryInput(kind='fact', text='DEPENDENCY_CONSTRAINT', source_refs=ref, point_ids=['p1']),
        MemoryEntryInput(kind='summary', text='EXPIRED_RESULT', source_refs=ref, point_ids=['p1'], effective_until=2),
    ])
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=new.id)
    values = {i['text']: i for i in selected.required}
    assert 'DEPENDENCY_CONSTRAINT' in values
    assert 'EXPIRED_RESULT' not in values
    assert values['SECRET_RESULT']['audience'] == 'author_only'


def test_ambiguous_bare_point_is_rejected(session, ready_project):
    from test_plot_point_stages import proposed_plot
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    # Source inventory is project-wide; simulate an independently approved second stage.
    ready_project.active_workflow_id = None
    session.commit()
    service, stage, second = proposed_plot(session, ready_project)
    service.approve_roadmap(stage.id, second.id, 'author')
    memory = StoryMemoryService(session)
    with pytest.raises(ValueError, match='ambiguous|歧义'):
        memory.create_card(ready_project.id, [MemoryEntryInput(kind='fact', text='one stage only',
            source_refs=card.entries[0]['source_refs'], point_ids=['p1'])], 'author')


def test_superseded_character_state_is_not_still_mandatory(session, ready_project):
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    old_id, new_id = str(uuid4()), str(uuid4())
    common = dict(project_id=ready_project.id, kind='character', source_refs=card.entries[0]['source_refs'],
                  point_ids=[f'{roadmap.id}:p1'], entity_ids=['hero'], effective_from=1,
                  audience='narratable', state_scope='official')
    session.add(StoryMemoryEntry(id=old_id, source_key='old', text='HERO_ALIVE', **common))
    session.flush()
    session.add(StoryMemoryEntry(id=new_id, source_key='new', text='HERO_DEAD', supersedes_id=old_id, **common))
    session.commit()
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=card.id)
    states = [i['text'] for i in selected.required if i['kind'] == 'character']
    assert states == ['HERO_DEAD']


def test_optional_history_retrieval_is_bounded_and_project_scoped(session, ready_project):
    from ainovel.models import Chapter, WritingBatch
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    # An earlier formal chapter, not candidate prose, is the only searchable history.
    batch = WritingBatch(id=str(uuid4()), project_id=ready_project.id, base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=999, planned_chapters=1, status='approved')
    session.add(batch)
    session.flush()
    session.add(Chapter(id=str(uuid4()), batch_id=batch.id, project_id=ready_project.id, ordinal=1,
        title='history', body='Alice found the bronze key.\n\n' + 'padding ' * 700,
        visible_char_count=4500, status='official', official_chapter_number=1, revision=1, state_delta={}))
    from ainovel.models import StageWorkflowNode
    node = session.query(StageWorkflowNode).filter_by(workflow_id=workflow.id).one()
    node.book_ordinal = 2
    session.commit()
    new = revise_card(session, ready_project, card, [MemoryEntryInput(kind='fact', text='Alice is investigating',
        source_refs=card.entries[0]['source_refs'], point_ids=['p1'], entity_ids=['Alice'])])
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=new.id)
    assert any('bronze key' in i['text'] for i in selected.optional)
    assert len(selected.optional) <= 8
    assert all(r['project_id'] == ready_project.id for i in selected.optional for r in i['source_refs'])


def test_ambiguous_character_versions_block_instead_of_guessing(session, ready_project):
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    new = revise_card(session, ready_project, card, [MemoryEntryInput(kind='character', text=text,
        source_refs=card.entries[0]['source_refs'], point_ids=['p1'], entity_ids=['hero'])
        for text in ('HERO_ALIVE', 'HERO_DEAD')])
    selected = ScopedContextService(session).select(workflow.id, step.id, card_id=new.id)
    assert any('状态重叠' in reason for reason in selected.missing)


def test_optional_keyword_history_has_exclusion_diagnostics(session, ready_project):
    from ainovel.models import Chapter, WritingBatch
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    batch = WritingBatch(id=str(uuid4()), project_id=ready_project.id, base_outline_version_id=workflow.base_outline_version_id,
        sequence_number=999, planned_chapters=1, status='approved')
    session.add(batch)
    session.flush()
    session.add(Chapter(id=str(uuid4()), batch_id=batch.id, project_id=ready_project.id, ordinal=1,
        title='history', body='bronze key found\n\nunrelated paragraph', visible_char_count=4500,
        status='official', official_chapter_number=1, revision=1, state_delta={}))
    session.commit()
    service = ScopedContextService(session)
    optional, excluded = service._history(ready_project.id, 2, set(), 'bronze key')
    assert optional[0]['text'] == 'bronze key found'
    assert excluded[0]['trim_reason'] == 'no_keyword_match'

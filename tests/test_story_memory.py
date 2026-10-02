import pytest
from sqlalchemy import select

from ainovel.agents.memory_contracts import MemoryEntryInput, SourceRef
from ainovel.models import ContextSource
from ainovel.services.projects import ProjectService


def setup_memory(session, project):
    from ainovel.services.story_memory import StoryMemoryService
    ProjectService(session).add_constitution(project.id, {'rules': '不能复活\n\n不能瞬移', 'ending': '回到故乡'}, True)
    return StoryMemoryService(session)


def all_entries(service, project_id):
    return [MemoryEntryInput(kind='rule', text=item['text'], source_refs=[SourceRef(**item['ref'])])
            for item in service.sources(project_id) if item['needs_classification']]


def test_cards_require_full_author_classification_and_do_not_commit(session, project):
    service = setup_memory(session, project)
    card = service.create_card(project.id, [], 'author')
    assert card.status == 'DRAFT'
    with pytest.raises(ValueError, match='uncovered'):
        service.approve_card(card.id, card.source_fingerprint, 'author')
    full = service.create_card(project.id, all_entries(service, project.id), 'author')
    service.approve_card(full.id, full.source_fingerprint, 'author')
    assert full.status == 'APPROVED'
    full_id = full.id
    session.rollback()
    from ainovel.models import MemoryCardVersion
    assert session.get(MemoryCardVersion, full_id) is None


@pytest.mark.parametrize('change', [{'project_id': 'other'}, {'content_hash': '0' * 64}, {'excerpt_start': 0, 'excerpt_end': 9999}])
def test_invalid_source_references_are_rejected(session, project, change):
    service = setup_memory(session, project)
    ref = service.sources(project.id)[0]['ref'] | change
    with pytest.raises(ValueError):
        service.validate_refs([SourceRef(**ref)])


def test_source_versions_invalidate_card_and_preserve_index_history(session, project):
    service = setup_memory(session, project)
    before = service.ensure_index(project.id)
    card = service.create_card(project.id, all_entries(service, project.id), 'author')
    session.commit()
    old_refs = card.entries[0]['source_refs']
    old_ids = set(session.scalars(select(ContextSource.id)).all())
    ProjectService(session).add_constitution(project.id, {'rules': '新世界规则'}, True)
    assert service.ensure_index(project.id) != before
    assert old_ids <= set(session.scalars(select(ContextSource.id)).all())
    with pytest.raises(ValueError, match='stale'):
        service.approve_card(card.id, before, 'author')
    service.validate_refs([SourceRef(**ref) for ref in old_refs])  # historical provenance remains resolvable


def test_paragraphs_keep_exact_offsets_and_large_paragraphs(session, project):
    from ainovel.services.story_memory import StoryMemoryService
    text = '甲\n\n' + '乙' * 20000
    ProjectService(session).add_constitution(project.id, {'rules': text}, True)
    sources = StoryMemoryService(session).sources(project.id)
    assert [s['text'] for s in sources] == ['甲', '乙' * 20000]
    assert sources[1]['ref']['excerpt_start'] == 3
    assert sources[1]['ref']['excerpt_end'] == 20003


def test_approved_card_cannot_be_rewritten(session, project):
    service = setup_memory(session, project)
    card = service.create_card(project.id, all_entries(service, project.id), 'author')
    service.approve_card(card.id, card.source_fingerprint, 'author')
    session.commit()
    card.entries = []
    with pytest.raises(ValueError, match='immutable'):
        session.flush()
    session.rollback()

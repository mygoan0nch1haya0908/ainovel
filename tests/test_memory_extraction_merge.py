from copy import deepcopy
from uuid import uuid4
import pytest
from test_orchestrator import ready_project
from test_memory_extraction_execution import extraction_fixture, ExtractionProvider
from ainovel.agents.memory_contracts import MemoryEntryInput
from ainovel.models import MemoryCardVersion
from ainovel.services.story_memory import StoryMemoryService


def complete(session, project):
    service, job, auth = extraction_fixture(session, project)
    for _ in range(8):
        service.run_next(job.id, auth, ExtractionProvider())
    if job.status == 'PAUSED':
        auth = str(uuid4())
        service.authorize(job.id, job.revision, auth, 'author')
        service.run_next(job.id, auth, ExtractionProvider())
    return service, job


def test_forged_cross_chunk_reference_rejected(session, ready_project):
    from ainovel.services.memory_extraction_merge import validate_chunk_result
    service, job, auth = extraction_fixture(session, ready_project)
    chunk = service.chunks(job.id)[0].snapshot
    with pytest.raises(ValueError):
        validate_chunk_result(chunk, dict(entries=[dict(kind='rule', text='fake', references=[
            dict(source_id='other-chunk', start=0, end=1)])], unresolved=[]))


def test_unresolved_blocks_complete_merge(session, ready_project):
    from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
    service, job = complete(session, ready_project)
    chunk = service.chunks(job.id)[0]
    chunk.result = {**chunk.result, 'unresolved': ['待作者确认']}
    session.commit()
    merge = MemoryExtractionMergeService(session)
    preview = merge.preview(job.id)
    assert preview['blockers']
    with pytest.raises(ValueError):
        merge.merge(job.id, job.revision, preview['fingerprint'], preview['entries'], 'author')


def test_manual_edit_creates_new_draft_not_approval(session, ready_project):
    from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
    service, job = complete(session, ready_project)
    merge = MemoryExtractionMergeService(session)
    preview = merge.preview(job.id)
    edited = deepcopy(preview['entries'])
    edited[0]['text'] = '作者修订的约束'
    card = merge.merge(job.id, job.revision, preview['fingerprint'], edited, 'author')
    assert card.status == 'DRAFT' and card.entries[0]['text'] == '作者修订的约束'
    again = merge.merge(job.id, 1, preview['fingerprint'], edited, 'author')
    assert again.id == card.id
    assert len(session.query(MemoryCardVersion).all()) == 1


def test_new_card_during_review_rejects_merge(session, ready_project):
    from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
    service, job = complete(session, ready_project)
    merge = MemoryExtractionMergeService(session)
    preview = merge.preview(job.id)
    StoryMemoryService(session).create_card(ready_project.id, [MemoryEntryInput.model_validate(e) for e in preview['entries']], 'author')
    session.commit()
    with pytest.raises(ValueError):
        merge.merge(job.id, job.revision, preview['fingerprint'], preview['entries'], 'author')


def test_locked_entries_unchanged(session, ready_project):
    from ainovel.services.memory_extraction_merge import validate_chunk_result
    service, job, auth = extraction_fixture(session, ready_project)
    chunk = deepcopy(service.chunks(job.id)[0].snapshot)
    chunk['request']['input_payload']['existing_entries'] = [dict(entry_id='locked', author_locked=True)]
    s = chunk['request']['input_payload']['source']
    value = validate_chunk_result(chunk, dict(entries=[dict(kind='rule', text='AI修改', replaces_entry_id='locked',
        references=[dict(source_id=s['source_id'], start=s['start'], end=s['end'])])], unresolved=[]))
    assert value['conflicts'] and value['entries'] == []


def test_invalid_reference_keeps_billable_usage(session, ready_project):
    from ainovel.models import MemoryExtractionAttempt
    from sqlalchemy import select
    service, job, auth = extraction_fixture(session, ready_project)
    class Invalid(ExtractionProvider):
        def generate(self, request):
            response = super().generate(request)
            response.structured['entries'][0]['references'][0]['source_id'] = 'forged'
            return response
    service.run_next(job.id, auth, Invalid())
    attempt = session.scalar(select(MemoryExtractionAttempt))
    assert attempt.input_tokens == 100 and attempt.output_tokens == 50
    assert service.chunks(job.id)[0].result is None


def test_multiple_updates_to_base_entry_are_not_last_wins(session, ready_project):
    from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
    service, job = complete(session, ready_project)
    chunks = service.chunks(job.id)
    base = MemoryEntryInput(kind='rule', text='原规则', entry_id='shared',
        source_refs=[chunks[0].snapshot['source']['ref']]).model_dump()
    job.snapshot = {**job.snapshot, 'base': [base]}
    for index, chunk in enumerate(chunks[:2]):
        snapshot = deepcopy(chunk.snapshot)
        snapshot['request']['input_payload']['existing_entries'] = [base]
        chunk.snapshot = snapshot
        result = deepcopy(chunk.result)
        result['entries'][0].update(replaces_entry_id='shared', text=f'互斥更新{index}')
        chunk.result = result
    session.commit()
    preview = MemoryExtractionMergeService(session).preview(job.id)
    assert preview['blockers']


def test_replaced_source_requires_explicit_disposition(session, ready_project):
    from ainovel.services.memory_extraction_merge import MemoryExtractionMergeService
    from ainovel.services.projects import ProjectService
    service, old = complete(session, ready_project)
    merge = MemoryExtractionMergeService(session)
    preview = merge.preview(old.id)
    merge.merge(old.id, old.revision, preview['fingerprint'], preview['entries'], 'author')
    ProjectService(session).add_constitution(ready_project.id, {'rule': '替换后的新规则'}, True)
    job = service.prepare(ready_project.id, old.model_profile_version_id, None, [], 'author')
    auth = str(uuid4())
    service.authorize(job.id, job.revision, auth, 'author')
    service.run_next(job.id, auth, ExtractionProvider())
    preview = merge.preview(job.id)
    assert preview['obsolete_ids']
    with pytest.raises(ValueError):
        merge.merge(job.id, job.revision, preview['fingerprint'], preview['entries'], 'author')
    # Explicit removal via the candidate editor; old card remains untouched.
    edited = [e for e in preview['entries'] if e['entry_id'] not in preview['obsolete_ids']]
    card = merge.merge(job.id, job.revision, preview['fingerprint'], edited, 'author')
    assert card.status == 'DRAFT' and len(card.entries) == 1

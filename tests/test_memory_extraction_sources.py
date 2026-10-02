from copy import deepcopy
from hashlib import sha256
import pytest
from ainovel.services.story_memory import canonical
from ainovel.context.budget import ConservativeEstimator
from test_orchestrator import ready_project
from test_scoped_context_selection import scoped_fixture


def snapshot(text='甲乙\n  丙丁' * 20000):
    return {'project_id': 'p', 'sources': [{'text': text, 'ref': dict(project_id='p',
        source_type='constitution', source_id='c', source_version='1', content_hash=sha256(text.encode()).hexdigest(),
        field_path='/rule', excerpt_start=0, excerpt_end=len(text)), 'needs_classification': True}],
        'metadata': [], 'points': []}


PROFILE = dict(model='test', version_id='v1', context_limit=140000, output_limit=64000, timeout_seconds=120)


def test_long_chinese_roundtrip_ranges():
    from ainovel.services.memory_extraction_sources import build_chunks
    source = snapshot()
    chunks = build_chunks(source, [], PROFILE)
    assert len(chunks) > 1
    assert ''.join(c['source']['text'] for c in chunks) == source['sources'][0]['text']
    assert chunks[0]['source']['ref']['excerpt_start'] == 0
    assert chunks[-1]['source']['ref']['excerpt_end'] == len(source['sources'][0]['text'])
    for left, right in zip(chunks, chunks[1:]):
        assert left['source']['ref']['excerpt_end'] == right['source']['ref']['excerpt_start']


def test_full_request_budget_includes_schema():
    from ainovel.services.memory_extraction_sources import build_chunks
    for chunk in build_chunks(snapshot(), [], PROFILE):
        assert chunk['request']['max_output_tokens'] == 64000
        assert chunk['request']['max_input_tokens'] == 64000
        assert ConservativeEstimator().estimate(canonical(chunk['request'])) <= 64000
    with pytest.raises(ValueError):
        build_chunks(snapshot('短'), [], {**PROFILE, 'context_limit': 64000})


def test_metadata_is_not_manual_prose(session, ready_project):
    from ainovel.services.memory_extraction_sources import freeze_sources
    workflow, step, card, roadmap = scoped_fixture(session, ready_project, 1)
    value = freeze_sources(session, ready_project.id, [])
    assert any(s['ref']['field_path'] == '/author_locked' for s in value['metadata'])
    assert all(s['ref']['field_path'] not in ('/title', '/kind', '/author_locked')
               for s in value['sources'] if s['ref']['source_type'] == 'outline_node')
    assert all(p['point_id'].startswith(roadmap.id + ':') for p in value['points'])


def test_changed_or_deleted_source_invalidates_cache():
    from ainovel.services.memory_extraction_sources import build_chunks
    original = snapshot('规则一')
    first = build_chunks(original, [], PROFILE)[0]['cache_key']
    for change in (snapshot('规则二'), {**original, 'metadata': [{'author_locked': True}]},
                   {**original, 'points': [{'point_id': 'different'}]}):
        assert build_chunks(change, [], PROFILE)[0]['cache_key'] != first
    assert build_chunks(original, [], {**PROFILE, 'version_id': 'v2'})[0]['cache_key'] != first
    assert build_chunks({**original, 'sources': []}, [], PROFILE) == []

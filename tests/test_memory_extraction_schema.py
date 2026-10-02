import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from ainovel.db import create_engine_for_url


def test_upgrade_preserves_cards_and_workflows(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'extraction.db'}"
    monkeypatch.setenv('AINOVEL_DATABASE_URL', url)
    config = Config('alembic.ini')
    command.upgrade(config, '0009_scoped_story_memory')
    engine = create_engine_for_url(url)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO novel_projects (id,title,target_chars_min,target_chars_max,next_batch_sequence,next_official_chapter_number,created_at,updated_at) VALUES ('p','preserved',2000000,5000000,1,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        conn.execute(text("INSERT INTO memory_card_versions (id,project_id,version_number,status,entries,source_fingerprint,created_at,updated_at) VALUES ('c','p',1,'DRAFT','[]','original',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
        before = conn.execute(text('SELECT * FROM memory_card_versions')).all()
    command.upgrade(config, 'head')
    assert 'memory_extraction_jobs' in inspect(engine).get_table_names()
    with engine.connect() as conn:
        assert conn.execute(text('SELECT * FROM memory_card_versions')).all() == before
        assert conn.execute(text('SELECT count(*) FROM workflow_context_policies')).scalar() == 0
        assert conn.execute(text('PRAGMA foreign_key_check')).all() == []
    command.check(config)
    command.downgrade(config, '0009_scoped_story_memory')
    assert 'memory_extraction_jobs' not in inspect(engine).get_table_names()
    engine.dispose()


def test_unknown_usage_nullable():
    from ainovel.models.memory_extraction import MemoryExtractionAttempt
    attempt = MemoryExtractionAttempt(id='a', chunk_id='c', authorization_id='u', status='UNKNOWN')
    assert attempt.input_tokens is None and attempt.output_tokens is None


def test_old_entries_still_parse():
    from ainovel.agents.memory_contracts import MemoryEntryInput
    from ainovel.agents.memory_extraction_contracts import identified_entries
    value = dict(kind='rule', text='rule', source_refs=[dict(project_id='p', source_type='constitution',
        source_id='c', source_version='1', content_hash='a'*64, field_path='/rule')])
    parsed = MemoryEntryInput.model_validate(value)
    assert parsed.author_locked is False
    first = identified_entries('card-1', [value])
    assert first == identified_entries('card-1', [value])
    assert first[0]['entry_id'] != identified_entries('card-2', [value])[0]['entry_id']
    assert 'entry_id' not in value


def test_64k_profile_and_adapter_are_configurable(session):
    from ainovel.services.model_profiles import ModelProfileService, ProfileInput
    from test_model_profiles import MemoryVault
    from ainovel.providers.compatible import CompatibleProvider
    from ainovel.providers.endpoint_policy import normalize_endpoint
    view = ModelProfileService(session, vault=MemoryVault()).create(
        ProfileInput('large', 'https://example.com/v1', 'remote', 'test', 140000, 64000), api_key='fake')
    provider = CompatibleProvider(normalize_endpoint(view.base_url, 'remote'), view.model_name,
        context_window_limit=view.context_limit, max_output_tokens_limit=view.output_limit)
    assert provider.capabilities('test').max_output_tokens == 64000

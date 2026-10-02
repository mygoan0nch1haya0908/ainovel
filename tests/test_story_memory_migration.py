import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ainovel.db import create_engine_for_url, database_readiness


def test_memory_migration_preserves_legacy_and_guards_downgrade(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'memory.db'}"
    monkeypatch.setenv('AINOVEL_DATABASE_URL', url)
    config = Config('alembic.ini')
    command.upgrade(config, '0008_model_output_capacity')
    engine = create_engine_for_url(url)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO novel_projects (id,title,target_chars_min,target_chars_max,next_batch_sequence,next_official_chapter_number,created_at,updated_at) VALUES ('p','preserved',2000000,5000000,1,1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
    command.upgrade(config, 'head')
    assert {'memory_card_versions', 'story_memory_entries', 'workflow_context_policies'} <= set(inspect(engine).get_table_names())
    from ainovel.models.story_memory import MemoryCardVersion
    with Session(engine) as session:
        card = MemoryCardVersion(id='c', project_id='p', version_number=1, entries=[], source_fingerprint='a' * 64)
        session.add(card)
        session.commit()
        assert card.status == 'DRAFT'
    with engine.connect() as conn:
        assert conn.execute(text("SELECT title FROM novel_projects WHERE id='p'")).scalar_one() == 'preserved'
        assert conn.execute(text('PRAGMA foreign_key_check')).all() == []
        assert conn.execute(text('SELECT COUNT(*) FROM workflow_context_policies')).scalar_one() == 0
    assert database_readiness(engine) == (True, None)
    command.check(config)
    with pytest.raises(ValueError, match='memory'):
        command.downgrade(config, '0008_model_output_capacity')
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM memory_card_versions'))
    command.downgrade(config, '0008_model_output_capacity')
    assert 'memory_card_versions' not in inspect(engine).get_table_names()
    engine.dispose()


def test_memory_contract_rejects_bad_ranges():
    from ainovel.agents.memory_contracts import MemoryEntryInput, SourceRef
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SourceRef(project_id='p', source_type='constitution', source_id='c', source_version='1', content_hash='a' * 64, field_path='/rules', excerpt_start=10, excerpt_end=2)
    with pytest.raises(ValidationError):
        MemoryEntryInput(kind='fact', text='secret', source_refs=[], effective_from=4, effective_until=2)


def test_upgrade_preserves_existing_stage_and_workflow_mapping(tmp_path, monkeypatch):
    from ainovel.db import create_session_factory
    from ainovel.services.projects import ProjectService
    from ainovel.services.outlines import OutlineService, OutlineNodeInput
    from ainovel.services.stages import StageService
    from ainovel.providers.fake import FakeProvider
    from test_orchestrator import response
    from test_plot_point_planning import plot_payload
    from ainovel.services.scoped_context import active_policy
    url = f"sqlite+pysqlite:///{tmp_path / 'legacy-stage.db'}"
    monkeypatch.setenv('AINOVEL_DATABASE_URL', url)
    config = Config('alembic.ini')
    command.upgrade(config, '0008_model_output_capacity')
    engine = create_engine_for_url(url)
    with create_session_factory(engine)() as session:
        project = ProjectService(session).create('legacy', 2000000, 5000000)
        ProjectService(session).add_constitution(project.id, {'rule': 'original'}, True)
        outlines = OutlineService(session)
        outline = outlines.create_candidate(project.id, [OutlineNodeInput('book', None, 'book', 'book', 0)], 'original')
        outlines.approve(outline.id)
        stages = StageService(session)
        stage = stages.create(project.id, 'original stage')
        roadmap = stages.propose_roadmap(stage.id, 'author', 'fake', 'demo')
        stages.generate_roadmap(roadmap.id, FakeProvider([response(plot_payload(), 1)]))
        stages.approve_roadmap(stage.id, roadmap.id, 'author')
        workflow = stages.start_next_batch(stage.id, 'author', 'fake', 'demo', 1).workflow
        workflow_id = workflow.id
    tables = ('novel_projects', 'story_stages', 'stage_roadmap_versions', 'stage_workflows', 'stage_workflow_nodes', 'generation_workflows', 'workflow_steps', 'workflow_prompt_snapshots')
    with engine.connect() as conn:
        before = {t: conn.execute(text(f'SELECT * FROM {t}')).all() for t in tables}
    command.upgrade(config, 'head')
    with engine.connect() as conn:
        assert {t: conn.execute(text(f'SELECT * FROM {t}')).all() for t in tables} == before
        assert conn.execute(text('PRAGMA foreign_key_check')).all() == []
    with create_session_factory(engine)() as session:
        assert active_policy(session, workflow_id) is None
    engine.dispose()

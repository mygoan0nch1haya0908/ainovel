"""Preview on a temporary SQLite backup. Never writes the original or calls an LLM."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from ainovel.db import create_engine_for_url, create_session_factory
from ainovel.models import GenerationWorkflow, WorkflowStep
from ainovel.services.scoped_context import ScopedContextService
from ainovel.services.story_memory import StoryMemoryService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--card')
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error('database does not exist')
    root = Path(__file__).resolve().parents[1]
    scratch = root / '.superpowers' / 'runtime'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='memory-preview-', dir=scratch) as temp:
        copied = Path(temp) / 'preview.db'
        original = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
        backup = sqlite3.connect(copied)
        try:
            original.backup(backup)
        finally:
            backup.close()
            original.close()
        url = 'sqlite+pysqlite:///' + copied.as_posix()
        previous_url = os.environ.get('AINOVEL_DATABASE_URL')
        try:
            os.environ['AINOVEL_DATABASE_URL'] = url
            config = Config(str(root / 'alembic.ini'))
            config.set_main_option('script_location', str(root / 'alembic'))
            command.upgrade(config, 'head')
        finally:
            if previous_url is None:
                os.environ.pop('AINOVEL_DATABASE_URL', None)
            else:
                os.environ['AINOVEL_DATABASE_URL'] = previous_url
        engine = create_engine_for_url(url)
        try:
            with create_session_factory(engine)() as session:
                workflow = session.get(GenerationWorkflow, args.workflow)
                if workflow is None:
                    parser.error('workflow does not exist')
                step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.position == workflow.current_position))
                if step is None:
                    parser.error('workflow has no current model step')
                memory = StoryMemoryService(session)
                memory.ensure_index(workflow.project_id)  # copied database only
                preview = ScopedContextService(session).preview(workflow.id, step.id, card_id=args.card)
                sources = memory.sources(workflow.project_id)
                report = dict(workflow_id=workflow.id, status=workflow.status,
                    recorded_model_calls=workflow.model_calls_used, diagnostic_model_calls=0,
                    original_modified=False, source_paragraphs=len(sources),
                    needs_author_classification=sum(s['needs_classification'] for s in sources),
                    legacy_fixed_input_estimate=preview.legacy_estimated_tokens,
                    scoped_input_estimate=preview.estimated_tokens, capacity=preview.capacity, deficit=preview.deficit,
                    blockers=preview.blockers, warning='Estimates are local, not provider usage. Missing-card estimates are incomplete and cannot authorize dispatch.')
                print(json.dumps(report, ensure_ascii=False, indent=2))
                session.rollback()
        finally:
            engine.dispose()


if __name__ == '__main__':
    main()

import json
import re
from sqlalchemy import select
from test_orchestrator import ready_project
from test_context_policy import paused_fixture
from ainovel.models import MemoryCardVersion, WorkflowContextPolicy


def token(response):
    return re.search(r'name="csrf_token" value="([^"]+)"', response.text).group(1)


def test_memory_get_is_read_only_and_create_requires_confirmation(client, session, ready_project):
    workflow, step, card = paused_fixture(session, ready_project)
    response = client.get(f'/projects/{ready_project.id}/memory')
    assert response.status_code == 200
    assert '尚未启用自动 AI 提取' in response.text
    assert len(session.scalars(select(MemoryCardVersion)).all()) == 1
    route = f'/projects/{ready_project.id}/memory/cards'
    assert client.post(route, data={'entries_json': '[]'}).status_code == 403
    saved = client.post(route, data={'csrf_token': token(response), 'entries_json': json.dumps(card.entries)})
    assert saved.status_code == 200
    session.expire_all()
    rows = session.scalars(select(MemoryCardVersion).order_by(MemoryCardVersion.version_number)).all()
    assert len(rows) == 2 and rows[-1].status == 'DRAFT'


def test_preview_requires_explicit_confirm_and_never_runs(client, session, ready_project):
    workflow, step, card = paused_fixture(session, ready_project)
    response = client.get(f'/workflows/{workflow.id}/context-preview?card_id={card.id}')
    assert response.status_code == 200
    assert '本地估算' in response.text and 'p1' in response.text
    csrf = token(response)
    fingerprint = re.search(r'name="fingerprint" value="([^"]+)"', response.text).group(1)
    route = f'/workflows/{workflow.id}/context-policy'
    data = {'csrf_token': csrf, 'card_id': card.id, 'fingerprint': fingerprint}
    assert client.post(route, data=data).status_code == 422
    assert client.post(route, data={**data, 'author_confirm': 'yes'}).status_code == 200
    session.expire_all()
    assert workflow.model_calls_used == 0 and workflow.status == 'PAUSED_CONTEXT_OVERFLOW'
    assert session.scalar(select(WorkflowContextPolicy)).strategy == 'scoped_story_v1'


def test_empty_card_form_is_rejected_without_creating_version(client, session, ready_project):
    workflow, step, card = paused_fixture(session, ready_project)
    page = client.get(f'/projects/{ready_project.id}/memory')
    response = client.post(f'/projects/{ready_project.id}/memory/cards', data={
        'csrf_token': token(page), 'entries_json': '[]'})
    assert response.status_code == 422
    assert '尚未填写任何精简内容' in response.text
    assert len(session.scalars(select(MemoryCardVersion)).all()) == 1

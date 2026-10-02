from uuid import uuid4
from sqlalchemy import select
from test_orchestrator import ready_project
from test_memory_web import token
from test_memory_extraction_execution import extraction_fixture
from test_memory_extraction_merge import complete
from ainovel.models import MemoryExtractionJob, MemoryExtractionAttempt, MemoryExtractionAuthorization


def test_get_and_prepare_never_call_model(client, session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    page = client.get(f'/projects/{ready_project.id}/memory')
    assert 'AI 整理记忆' in page.text and '高级编辑' in page.text
    response = client.post(f'/projects/{ready_project.id}/memory/extractions', data={
        'csrf_token': token(page), 'profile_version_id': job.model_profile_version_id})
    assert response.status_code == 200
    assert '64000' in response.text
    assert session.scalar(select(MemoryExtractionAttempt)) is None


def test_csrf_and_cross_project_rejected(client, session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    route = f'/memory-extractions/{job.id}/cancel'
    assert client.post(route, data={}).status_code == 403
    page = client.get(f'/memory-extractions/{job.id}')
    response = client.post(route, data={'csrf_token': token(page), 'project_id': 'other', 'revision': job.revision})
    assert response.status_code == 404


def test_authorize_double_submit_no_double_budget(client, session, ready_project):
    service, job, auth = extraction_fixture(session, ready_project)
    page = client.get(f'/memory-extractions/{job.id}')
    for _ in range(2):
        response = client.post(f'/memory-extractions/{job.id}/authorize', data={
            'csrf_token': token(page), 'project_id': ready_project.id, 'revision': 1,
            'authorization_id': auth, 'author_confirm': 'yes'})
        assert response.status_code == 200
    assert len(session.scalars(select(MemoryExtractionAuthorization)).all()) == 1


def test_preview_escapes_model_text(client, session, ready_project):
    service, job = complete(session, ready_project)
    chunk = service.chunks(job.id)[0]
    from copy import deepcopy
    result = deepcopy(chunk.result)
    result['entries'][0]['text'] = '<script>alert(1)</script>'
    chunk.result = result
    session.commit()
    page = client.get(f'/memory-extractions/{job.id}/diff')
    assert page.status_code == 200
    assert '<script>alert(1)</script>' not in page.text
    assert '&lt;script&gt;' in page.text
    assert 'credential_ref' not in page.text
    assert '确认合并' in page.text


def test_diff_shows_reference_and_scope_changes(client, session, ready_project):
    service, job = complete(session, ready_project)
    page = client.get(f'/memory-extractions/{job.id}/diff')
    assert '原文引用' in page.text
    assert '适用起始章' in page.text
    assert '最早揭示章' in page.text
    assert '修改前' in page.text and '修改后' in page.text


def test_claim_recovery_requires_explicit_confirmation(client, session, ready_project):
    from ainovel.services.project_llm_guard import ProjectLLMGuard
    from ainovel.models import ProjectLLMClaim
    ProjectLLMGuard.claim(session, ready_project.id, 'test-owner')
    session.commit()
    page = client.get(f'/projects/{ready_project.id}/memory')
    assert '确认中断并解除旧请求占用' in page.text
    response = client.post(f'/projects/{ready_project.id}/memory/reconcile-call', data={
        'csrf_token': token(page), 'owner_id': 'test-owner'})
    assert response.status_code == 422
    assert session.get(ProjectLLMClaim, ready_project.id) is not None

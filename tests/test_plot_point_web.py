from copy import deepcopy

from test_stage_web import ready_project, provider_registry, stage_provider, post
from test_plot_point_stages import proposed_plot
from test_orchestrator import response
from ainovel.providers.fake import FakeProvider


def test_plot_ui_ranges_progress_and_feedback_diff(client, session, ready_project, stage_provider):
    service, stage, version = proposed_plot(session, ready_project, (6, 54))
    service.approve_roadmap(stage.id, version.id, 'author')
    stage.confirmed_chapters = 3
    session.commit()
    path = f'/stages/{stage.id}'
    page = client.get(path)
    assert page.status_code == 200
    assert '剧情点' in page.text
    assert '第 1—6 章' in page.text and '第 7—60 章' in page.text
    assert '已确认 3 / 6 章' in page.text
    assert '预计 60 章' in page.text
    saved = post(client, f'{path}/roadmaps/{version.id}/feedback', path,
                 {'feedback': '后段增加两章 <script>alert(1)</script>', 'feedback_confirm':'yes', 'stage_revision':str(stage.revision)})
    assert saved.status_code == 303
    revision = service.list_roadmaps(stage.id)[-1]
    assert revision.attempts_used == 0 and stage_provider.roles == []
    assert 'points' in revision.prompt_snapshot['schema']['properties']
    changed = deepcopy(version.payload)
    changed['points'][1]['chapter_count'] = 56
    service.generate_roadmap(revision.id, FakeProvider([response(changed, 2)]))
    html = client.get(path).text
    assert '第 7—60 章' in html and '第 7—62 章' in html
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;' in html


def test_legacy_conversion_entry_and_confirmed_restriction(client, session, ready_project, stage_provider):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, '旧模式')
    old = service.propose_roadmap(stage.id, 'author', 'fake', 'demo', roadmap_format='legacy')
    service.generate_roadmap(old.id, stage_provider)
    service.approve_roadmap(stage.id, old.id, 'author')
    path = f'/stages/{stage.id}'
    assert '新建剧情点规划' in client.get(path).text
    stage.confirmed_chapters = 1
    session.commit()
    assert '已有确认正文，不能转换为剧情点格式' in client.get(path).text


def test_legacy_overflow_rebuild_keeps_format_after_validation_error(client, session, ready_project, stage_provider):
    from ainovel.services.stages import StageService
    service = StageService(session)
    stage = service.create(ready_project.id, '旧请求')
    old = service.propose_roadmap(stage.id, 'author', 'fake', 'demo', roadmap_format='legacy')
    old.status = 'PAUSED_CONTEXT_OVERFLOW'
    session.commit()
    path = f'/stages/{stage.id}'
    html = client.get(f'{path}?retry_roadmap={old.id}').text
    assert 'name="roadmap_format" value="legacy"' in html
    invalid = post(client, f'{path}/roadmaps', path,
                   {'provider_name':'fake','model_name':'demo','roadmap_format':'legacy'})
    assert invalid.status_code == 422
    assert 'name="roadmap_format" value="legacy"' in invalid.text
    rebuilt = post(client, f'{path}/roadmaps', path,
                   {'provider_name':'fake','model_name':'demo','roadmap_format':'legacy','author_confirm':'yes'})
    assert rebuilt.status_code == 303
    new = service.list_roadmaps(stage.id)[-1]
    assert 'nodes' in new.prompt_snapshot['schema']['properties']
    assert new.attempts_used == 0 and stage_provider.roles == []
    assert old.status == 'PAUSED_CONTEXT_OVERFLOW'
    assert 'name="roadmap_format" value="plot_points_v1"' in client.get(path).text

import pytest

from test_stage_web import ready_project, stage_provider, provider_registry, post
from ainovel.services.stages import StageService, stage_request


def test_custom_output_budget_is_frozen_and_sent(session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    old = service.propose_roadmap(stage.id, 'author', 'fake', 'demo')
    new = service.propose_roadmap(stage.id, 'author', 'fake', 'demo',
        context_window=64000, max_output_tokens=12000, requested_output_tokens=12000)
    assert old.output_token_limit == 8000
    assert new.output_token_limit == 12000
    assert new.total_output_token_limit == 24000
    assert new.input_token_limit == 50976
    assert stage_request(new, 64000, 12000).max_output_tokens == 12000
    assert 'points' in new.prompt_snapshot['schema']['properties']


@pytest.mark.parametrize('value', [0,-1,True,1.5,'12000',12001])
def test_custom_output_budget_rejects_invalid_or_over_capacity(session, ready_project, value):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    with pytest.raises(ValueError):
        service.propose_roadmap(stage.id, 'author', 'fake', 'demo',
            max_output_tokens=12000, requested_output_tokens=value)
    assert service.list_roadmaps(stage.id) == []


def test_output_budget_form_preserves_errors_and_saves_without_call(client, session, ready_project, stage_provider):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    path = f'/stages/{stage.id}'
    data = dict(provider_name='fake', model_name='demo', output_token_limit='12000')
    failed = post(client, path+'/roadmaps', path, data)
    assert failed.status_code == 422
    assert 'name="output_token_limit"' in failed.text and 'value="12000"' in failed.text
    result = post(client, path+'/roadmaps', path, {**data, 'author_confirm':'yes'})
    assert result.status_code == 303
    version = service.list_roadmaps(stage.id)[-1]
    assert version.output_token_limit == 12000
    assert version.attempts_used == 0 and stage_provider.roles == []
    bad = post(client, path+'/roadmaps', path, {**data,'output_token_limit':'16001','author_confirm':'yes'})
    assert bad.status_code == 422
    assert len(service.list_roadmaps(stage.id)) == 1

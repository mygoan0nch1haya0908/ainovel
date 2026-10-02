from test_orchestrator import ready_project
from test_stage_web import provider_registry,stage_provider
from ainovel.services.stages import StageService
from ainovel.providers.llm_response import LLMProviderError
from ainovel.providers.diagnostics import FailureReason
from ainovel.providers.fake import FakeProvider
from ainovel.models.audit import AuditEvent
from test_workflows import clock,workflow
import pytest

def test_stage_failure_diagnostic_is_saved_displayed_and_cooldown_blocks_dispatch(client,session,ready_project):
    svc=StageService(session)
    stage=svc.create(ready_project.id,'诊断测试')
    v=svc.propose_roadmap(stage.id,'author','fake','demo')
    class Limited(FakeProvider):
        calls=0
        def generate(self,request):
            self.calls+=1
            raise LLMProviderError(FailureReason.RATE_LIMIT,http_status=429)
    provider=Limited([])
    svc.generate_roadmap(v.id,provider)
    event=session.query(AuditEvent).filter_by(action='llm_call_failed').one()
    assert event.details['diagnostic']['error_category']=='rate_limit'
    page=client.get('/stages/'+stage.id)
    assert page.status_code==200 and '无法确定' in page.text and '技术详情（debug）' in page.text
    assert 'Retry-After' in page.text and 'unknown_rate_limit' in page.text
    with pytest.raises(ValueError,match='cooldown'):
        svc.generate_roadmap(v.id,provider)
    assert provider.calls==1

def test_http_action_groups_multiple_model_calls(client):
    from ainovel.providers.compatible import CompatibleProvider
    from ainovel.providers.endpoint_policy import normalize_endpoint
    from test_compatible_provider import request,Transport,reply
    provider=CompatibleProvider(normalize_endpoint('https://api.example/v1','remote'),'model-a',api_key='SYNTHETIC',transport=Transport(reply()),allow_real_calls=True)
    @client.app.get('/diagnostic-test-action')
    def action():
        return [provider.generate(request()).diagnostic['operation_id'] for _ in range(2)]
    first=client.get('/diagnostic-test-action').json()
    second=client.get('/diagnostic-test-action').json()
    assert first[0] and first[0]==first[1] and second[0]==second[1] and first[0]!=second[0]

def test_workflow_rate_limit_pauses_and_resumes_after_cooldown(session,workflow,clock):
    from ainovel.services.workflows import WorkflowService
    svc=WorkflowService(session,clock=clock)
    step=svc.claim_step(workflow.id,{'PLANNING'},'test')
    attempt=svc.record_attempt_start(step.id,'a'*64,claim_revision=step.revision)
    error=LLMProviderError(FailureReason.RATE_LIMIT,http_status=429)
    error.diagnostic={'retry_after':12,'http_status':429}
    failed=svc.fail_attempt(attempt.id,error)
    assert failed.status=='PAUSED_PROVIDER'
    with pytest.raises(ValueError,match='cooldown'): svc.resume(workflow.id)
    clock.advance(seconds=13)
    assert svc.resume(workflow.id).status=='PLANNING'

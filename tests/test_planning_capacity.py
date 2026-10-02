from dataclasses import replace

from test_orchestrator import ready_project
from test_model_profiles import MemoryVault
from test_compatible_provider import Transport, reply, request
from test_safe_transport import ScriptedSocket, Connector, response
from ainovel.services.stages import StageService, stage_request
from ainovel.services.model_profiles import ModelProfileService, ProfileInput
from ainovel.providers.compatible import CompatibleProvider
from ainovel.providers.endpoint_policy import normalize_endpoint
from ainovel.providers import safe_transport


def test_new_planning_reserves_32k_and_preserves_old_snapshot(session, ready_project):
    service = StageService(session)
    stage = service.create(ready_project.id, '调查')
    old = service.propose_roadmap(stage.id, 'author', 'fake', 'demo', requested_output_tokens=8000)
    old_prompt = dict(old.prompt_snapshot)
    new = service.propose_roadmap(stage.id, 'author', 'fake', 'demo', context_window=64000, max_output_tokens=32000)
    req = stage_request(new, 64000, 32000)
    assert req.max_output_tokens == 32000
    assert req.max_input_tokens == 30976
    assert req.timeout_seconds == 600
    assert new.total_output_token_limit == 64000
    assert old.output_token_limit == 8000 and old.prompt_snapshot == old_prompt


def test_profile_persists_32k_output(session):
    service = ModelProfileService(session, vault=MemoryVault())
    view = service.create(ProfileInput('Test', 'https://example.com/v1', 'remote', 'model-a',
        context_limit=64000, output_limit=32000), api_key='TEST_SECRET')
    assert service.get_public(view.version_id).output_limit == 32000


def test_compatible_sends_full_planning_budget_and_timeout_once():
    transport = Transport(reply())
    provider = CompatibleProvider(normalize_endpoint('https://example.com/v1', 'remote'), 'model-a',
        api_key='TEST_SECRET', allow_real_calls=True, transport=transport,
        context_window_limit=64000, max_output_tokens_limit=32000)
    provider.generate(replace(request(), max_output_tokens=32000, timeout_seconds=600))
    assert len(transport.calls) == 1
    assert transport.calls[0][1]['payload']['max_tokens'] == 32000
    assert transport.calls[0][1]['timeout_seconds'] == 600


def test_transport_waits_past_old_180_cap_without_extending_connection_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(safe_transport.time, 'monotonic', lambda: now[0])
    class SlowSocket(ScriptedSocket):
        def recv(self, size):
            if self.response:
                assert self.timeouts[-1] > 300
                now[0] += 314
            return super().recv(size)
    sock = SlowSocket(response())
    connector = Connector(sock)
    result = safe_transport.SafeTransport(resolver=lambda *args: ['93.184.216.34'], connector=connector).request_json(
        normalize_endpoint('https://example.com/v1', 'remote'), 'GET', 'models', api_key='TEST_SECRET',
        timeout_seconds=600, max_response_bytes=65536)
    assert result == {'data': []} and sock.closed
    assert connector.calls[0][2]['timeout_seconds'] <= 10

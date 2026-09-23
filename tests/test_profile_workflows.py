from dataclasses import replace
import json
from threading import Event, Thread, current_thread
from uuid import uuid4

import pytest
from sqlalchemy import event, select

from ainovel.models.audit import AuditEvent
from ainovel.models.workflow import GenerationWorkflow
from ainovel.providers.contracts import ModelRequest, ProviderAuthenticationError, ProviderProtocolError
from ainovel.providers.fake import FakeProvider
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.model_profiles import ModelProfileService, ProfileInput
from ainovel.services.projects import ProjectService
from ainovel.services.stages import StageService
from ainovel.services.workflows import WorkflowBudgets, WorkflowService
from ainovel.workflows.orchestrator import WorkflowOrchestrator
from ainovel.agents.runner import AgentRunner
from ainovel.models.workflow import WorkflowStep


SECRET = "SENTINEL_PROFILE_42"


class Vault:
    def __init__(self):
        self.values = {}

    def put(self, secret):
        ref = str(uuid4())
        self.values[ref] = secret
        return ref

    def get(self, ref):
        return self.values[ref]

    def delete(self, ref):
        self.values.pop(ref, None)


class Transport:
    def __init__(self):
        self.calls = []

    def request_json(self, endpoint, method, path, *, api_key, payload=None, timeout_seconds, max_response_bytes):
        self.calls.append((endpoint.base_url, api_key, payload))
        return {"id": "safe-1", "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


@pytest.fixture
def profile_setup(session, client):
    vault, transport = Vault(), Transport()
    service = ModelProfileService(session, vault=vault)
    values = ProfileInput("Profile", "https://example.com/v1", "remote", "model-a", 8000, 1000)
    version = service.create(values, api_key=SECRET)
    return service, client.app.state.session_factory, vault, version, values, transport


def _resolver(session_factory, vault, transport):
    try:
        from ainovel.services.provider_resolution import ProviderResolver
    except ImportError:
        pytest.fail("profile-bound provider resolver is missing")
    return ProviderResolver(session_factory, ProviderRegistry({"fake": lambda: object()}),
                            vault=vault, transport=transport)


def _request(model="model-a"):
    return ModelRequest(model, "system", {"prompt": "hello"}, {"type": "object"}, 100, 10, 10.0, {})


def test_cached_proxy_rechecks_version_and_never_retains_secret(profile_setup):
    service, factory, vault, version, _, transport = profile_setup
    resolver = _resolver(factory, vault, transport)
    provider = resolver.resolve("compatible", "model-a", model_profile_version_id=version.version_id)
    assert SECRET not in repr(provider)
    assert provider.generate(_request()).structured == {"ok": True}
    assert len(transport.calls) == 1
    service.set_enabled(version.version_id, False)
    with pytest.raises(ProviderAuthenticationError):
        provider.generate(_request())
    assert len(transport.calls) == 1


def test_revision_does_not_move_existing_proxy_and_revoke_blocks_dispatch(profile_setup):
    service, factory, vault, version, values, transport = profile_setup
    resolver = _resolver(factory, vault, transport)
    provider = resolver.resolve("compatible", "model-a", model_profile_version_id=version.version_id)
    replacement = service.revise(version.profile_id, replace(values, model_name="model-b"), keep_existing_key=True)
    assert provider.generate(_request()).structured == {"ok": True}
    newer = resolver.resolve("compatible", "model-b", model_profile_version_id=replacement.version_id)
    assert newer.generate(_request("model-b")).structured == {"ok": True}
    service.revoke(version.profile_id)
    with pytest.raises(ProviderAuthenticationError):
        provider.generate(_request())
    with pytest.raises(ProviderAuthenticationError):
        newer.generate(_request("model-b"))
    assert len(transport.calls) == 2


def test_resolver_rejects_mismatches_without_legacy_fallback(profile_setup):
    _, factory, vault, version, _, transport = profile_setup
    resolver = _resolver(factory, vault, transport)
    with pytest.raises(ProviderProtocolError):
        resolver.resolve("compatible", "model-a")
    with pytest.raises(ProviderProtocolError):
        resolver.resolve("compatible", "model-b", model_profile_version_id=version.version_id)
    with pytest.raises(ProviderProtocolError):
        resolver.resolve("fake", "model-a", model_profile_version_id=version.version_id)
    assert transport.calls == []


def test_workflow_binds_version_and_clamps_budgets_in_persisted_snapshot(profile_setup, session, project, official_outline):
    service, _, _, version, _, _ = profile_setup
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    budgets = WorkflowBudgets()
    workflow = WorkflowService(session).start(project.id, "compatible", "model-a", 1, budgets,
                                              generation_version=2, model_profile_version_id=version.version_id)
    assert workflow.model_profile_version_id == version.version_id
    assert workflow.writer_output_tokens == 1000
    assert workflow.writer_input_tokens == 7000
    audit = session.scalar(select(AuditEvent).where(AuditEvent.entity_id == workflow.id,
                                                   AuditEvent.action == "workflow_started"))
    assert audit.details["model_profile_version_id"] == version.version_id
    assert SECRET not in json.dumps(audit.details)
    updated = service.revise(version.profile_id, replace(ProfileInput("Profile", "https://example.com/v1", "remote", "model-a", 8000, 1000), model_name="model-b"), keep_existing_key=True)
    assert updated.version_id != session.get(GenerationWorkflow, workflow.id).model_profile_version_id


def test_workflow_rejects_unbound_compatible_and_bound_legacy(profile_setup, session, project, official_outline):
    _, _, _, version, _, _ = profile_setup
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    with pytest.raises(ValueError):
        WorkflowService(session).start(project.id, "compatible", "model-a", 1, WorkflowBudgets())
    with pytest.raises(ValueError):
        WorkflowService(session).start(project.id, "fake", "demo", 1, WorkflowBudgets(),
                                       model_profile_version_id=version.version_id)
    assert session.scalar(select(GenerationWorkflow).where(GenerationWorkflow.project_id == project.id)) is None


def test_revocation_counts_include_historical_bindings_and_nonterminal_workflows(profile_setup, session, project, official_outline):
    service, _, _, version, _, _ = profile_setup
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    workflow = WorkflowService(session).start(project.id, "compatible", "model-a", 1, WorkflowBudgets(),
                                              model_profile_version_id=version.version_id)
    counts = service.impacted_task_counts(version.profile_id)
    assert counts == {"workflows": 1, "active_workflows": 1, "roadmaps": 0}
    workflow.status = "COMPLETED"
    session.commit()
    assert service.impacted_task_counts(version.profile_id)["active_workflows"] == 0


def _ready_stage(session, project):
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    return StageService(session).create(project.id, "A careful stage outline")


def test_stage_proposal_pins_version_and_batch_inherits_it(profile_setup, session, project, official_outline):
    service, _, _, version, _, _ = profile_setup
    stage = _ready_stage(session, project)
    roadmap = StageService(session).propose_roadmap(
        stage.id, "author", "compatible", "model-a",
        model_profile_version_id=version.version_id,
    )
    assert roadmap.model_profile_version_id == version.version_id
    assert roadmap.output_token_limit == 1000
    assert roadmap.input_token_limit == 7000
    roadmap.status = "PROPOSED"
    roadmap.payload = {
        "goal": "goal", "start_state": "start", "end_state": "end",
        "key_events": ["event"], "foreshadowing": [],
        "nodes": [{"node_id": "one", "ordinal": 1, "title": "title",
                   "goal": "goal", "dependencies": []}],
    }
    session.commit()
    StageService(session).approve_roadmap(stage.id, roadmap.id, "author")
    started = StageService(session).start_next_batch(
        stage.id, "author", "compatible", "model-a", 1,
    )
    assert started.workflow.model_profile_version_id == version.version_id
    assert service.impacted_task_counts(version.profile_id) == {
        "workflows": 1, "active_workflows": 1, "roadmaps": 1,
    }


def test_stage_dispatch_requires_resolver_for_bound_profile(profile_setup, session, project, official_outline):
    _, _, _, version, _, _ = profile_setup
    stage = _ready_stage(session, project)
    roadmap = StageService(session).propose_roadmap(
        stage.id, "author", "compatible", "model-a",
        model_profile_version_id=version.version_id,
    )
    fake = FakeProvider([])
    with pytest.raises(ValueError):
        StageService(session).generate_roadmap(roadmap.id, fake)
    assert fake.requests == []


def test_stage_retry_after_revoke_pauses_without_transport(profile_setup, session, project, official_outline):
    service, factory, vault, version, _, transport = profile_setup
    stage = _ready_stage(session, project)
    roadmap = StageService(session).propose_roadmap(
        stage.id, "author", "compatible", "model-a",
        model_profile_version_id=version.version_id,
    )
    resolver = _resolver(factory, vault, transport)
    service.revoke(version.profile_id)
    retried = StageService(session, provider_resolver=resolver).generate_roadmap(roadmap.id, None)
    assert retried.status == "PAUSED_PROVIDER"
    assert transport.calls == []


def test_orchestrator_cached_provider_pauses_after_revoke(profile_setup, session, client, project, official_outline):
    service, factory, vault, version, _, transport = profile_setup
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    workflow = WorkflowService(session).start(
        project.id, "compatible", "model-a", 1, WorkflowBudgets(),
        model_profile_version_id=version.version_id,
    )
    resolver = _resolver(factory, vault, transport)
    orchestrator = WorkflowOrchestrator(factory, ProviderRegistry({}), AgentRunner(),
                                         provider_resolver=resolver)
    step = session.scalar(select(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    cached = orchestrator._provider(step)
    assert SECRET not in repr(cached)
    service.revoke(version.profile_id)
    result = orchestrator.advance(workflow.id)
    assert result.status == "PAUSED_PROVIDER"
    assert transport.calls == []


def test_ownership_claim_commits_binding_before_waiting_revoke(profile_setup, session, client, project, official_outline):
    service, factory, vault, version, _, transport = profile_setup
    ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
    revoke_reached_write = Event()
    revoker_errors = []

    def observe_write(_connection, _cursor, statement, _parameters, _context, _executemany):
        if current_thread().name == "profile-revoker" and statement.lstrip().upper().startswith("UPDATE MODEL_PROFILES"):
            revoke_reached_write.set()

    def revoke():
        try:
            with factory() as other_session:
                ModelProfileService(other_session, vault=vault).revoke(version.profile_id)
        except Exception as error:
            revoker_errors.append(error)

    thread = Thread(target=revoke, name="profile-revoker")

    def before_commit(_workflow):
        thread.start()
        assert revoke_reached_write.wait(5), "revoker did not reach the profile write"

    event.listen(client.app.state.engine, "before_cursor_execute", observe_write)
    try:
        workflow = WorkflowService(session).start(
            project.id, "compatible", "model-a", 1, WorkflowBudgets(),
            model_profile_version_id=version.version_id, _before_commit=before_commit,
        )
        thread.join(10)
        assert not thread.is_alive()
        assert revoker_errors == []
        assert workflow.model_profile_version_id == version.version_id
        provider = _resolver(factory, vault, transport).resolve
        with pytest.raises(ProviderAuthenticationError):
            provider("compatible", "model-a", model_profile_version_id=version.version_id)
        assert transport.calls == []
    finally:
        event.remove(client.app.state.engine, "before_cursor_execute", observe_write)
        if thread.is_alive():
            thread.join(10)


def test_cached_inflight_call_finishes_but_next_call_after_revoke_is_blocked(profile_setup):
    service, factory, vault, version, _, _ = profile_setup
    entered, release = Event(), Event()
    results, errors = [], []

    class BlockingTransport(Transport):
        def request_json(self, *args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("in-flight test did not release transport")
            return super().request_json(*args, **kwargs)

    transport = BlockingTransport()
    proxy = _resolver(factory, vault, transport).resolve(
        "compatible", "model-a", model_profile_version_id=version.version_id
    )

    def dispatch():
        try:
            results.append(proxy.generate(_request()).structured)
        except Exception as error:
            errors.append(error)

    thread = Thread(target=dispatch, name="profile-inflight")
    thread.start()
    try:
        assert entered.wait(5), "admitted request did not reach transport"
        service.revoke(version.profile_id)
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert errors == []
        assert results == [{"ok": True}]
        assert len(transport.calls) == 1
        with pytest.raises(ProviderAuthenticationError):
            proxy.generate(_request())
        assert len(transport.calls) == 1
    finally:
        release.set()
        if thread.is_alive():
            thread.join(10)

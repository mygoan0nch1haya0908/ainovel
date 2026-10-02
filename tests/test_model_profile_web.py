from html import unescape
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, wait
import re
from threading import Event
from uuid import uuid4
import json
from pathlib import Path
import shutil
import subprocess
from sqlalchemy import select, text
from ainovel.models.workflow import GenerationWorkflow
from ainovel.models.stage import StageRoadmapVersion
from ainovel.services.projects import ProjectService
from ainovel.services.outlines import OutlineService, OutlineNodeInput

import pytest
from fastapi.testclient import TestClient

from ainovel.app import create_app
from ainovel.models import Base

SECRET = "SENTINEL_WEB_KEY_719"
FORM = dict(name="Author model", connection_kind="remote", base_url="https://example.com/v1",
            model_name="model-a", context_limit="32000", output_limit="12000", api_key=SECRET)


class Vault:
    def __init__(self):
        self.values = {}
    def put(self, value):
        ref = str(uuid4())
        self.values[ref] = value
        return ref
    def get(self, ref):
        return self.values[ref]
    def delete(self, ref):
        self.values.pop(ref, None)


class Transport:
    def __init__(self):
        self.calls = []
        self.models = ["model-a", '<img src=x onerror="boom()">']
        self.error = False
    def request_json(self, endpoint, method, path, **kwargs):
        self.calls.append((endpoint, method, path, kwargs))
        if self.error:
            raise RuntimeError(SECRET)
        if path == "models":
            return {"data": [{"id": model} for model in self.models]}
        return {"id": "test-1", "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


@pytest.fixture
def profile_client(database_url):
    vault, transport = Vault(), Transport()
    app = create_app(database_url, profile_vault=vault, profile_transport=transport)
    Base.metadata.create_all(app.state.engine)
    with app.state.engine.begin() as connection:
        connection.execute(text("CREATE VIRTUAL TABLE context_source_fts USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"))
    with TestClient(app) as client:
        yield client, vault, transport


def csrf(client, path="/model-profiles"):
    page = client.get(path)
    assert page.status_code == 200
    return unescape(re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1))


def save(client):
    response = client.post("/model-profiles", data={**FORM, "csrf_token": csrf(client)}, follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"]


def test_save_check_get_are_offline_and_secret_never_rendered(profile_client, caplog):
    client, vault, network = profile_client
    token = csrf(client)
    invalid = client.post("/model-profiles", data={**FORM, "name": "", "csrf_token": token})
    assert invalid.status_code == 422
    assert SECRET not in invalid.text + caplog.text
    check = client.post("/model-profiles/check", data={**FORM, "csrf_token": token})
    assert check.status_code == 200
    assert not vault.values
    page = client.get(save(client))
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert 'type="password"' in page.text and 'autocomplete="new-password"' in page.text
    assert SECRET not in page.text + client.get("/model-profiles").text + caplog.text
    assert network.calls == []


@pytest.mark.parametrize("key", ["white space", "nonascii-密钥", "newline\nkey", "x" * 8193])
def test_invalid_bearer_is_rejected_locally(profile_client, key):
    client, vault, network = profile_client
    response = client.post("/model-profiles", data={**FORM, "api_key": key, "csrf_token": csrf(client)})
    assert response.status_code == 422
    assert key not in response.text
    assert not vault.values and not network.calls


def test_secret_posts_reject_missing_csrf_duplicates_files_malformed_and_oversize(profile_client, caplog):
    client, vault, network = profile_client
    token = csrf(client)
    bodies = [
        ({"data": FORM}, 403),
        ({"content": f"csrf_token={token}&api_key={SECRET}&api_key={SECRET}", "headers": {"content-type": "application/x-www-form-urlencoded"}}, 422),
        ({"files": {"api_key": ("key.txt", SECRET)}, "data": {"csrf_token": token}}, 422),
        ({"content": SECRET, "headers": {"content-type": "multipart/form-data"}}, 422),
        ({"content": SECRET * 10000, "headers": {"content-type": "application/x-www-form-urlencoded"}}, 413),
    ]
    for payload, code in bodies:
        response = client.post("/model-profiles", **payload)
        assert response.status_code == code
        assert response.headers["cache-control"] == "no-store"
        assert SECRET not in response.text + caplog.text
    assert not vault.values and not network.calls


def test_network_and_destructive_actions_require_confirmation_and_escape_models(profile_client, caplog):
    client, vault, network = profile_client
    location = save(client)
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    token = csrf(client)
    base = f"/model-profiles/versions/{view.version_id}"
    for path in [base + "/models", base + "/test", base + "/enabled", location + "/revoke"]:
        assert client.post(path, data={"csrf_token": token}).status_code == 422
    assert not network.calls
    listed = client.post(base + "/models", data={"csrf_token": token, "confirm": "yes"})
    assert listed.status_code == 200
    assert '<img src=x' not in listed.text and '&lt;img' in listed.text
    network.error = True
    failed = client.post(base + "/test", data={"csrf_token": token, "confirm": "yes"})
    assert failed.status_code == 422
    assert SECRET not in failed.text + caplog.text
    assert 'name="model_name"' in failed.text
    revoked = client.post(location + "/revoke", data={"csrf_token": token, "confirm": "yes"})
    assert revoked.status_code == 200 and not vault.values


def test_exact_model_id_survives_save_binding_dispatch_and_audit(profile_client):
    from ainovel.models.audit import AuditEvent
    client, _, network = profile_client
    model = ' <img src=x onerror="boom()"> '
    created = client.post("/model-profiles", data={**FORM, "model_name": model, "csrf_token": csrf(client)})
    assert created.status_code == 200
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
        assert view.model_name == model
    project_id = ready_project(client)
    start = client.post(f"/projects/{project_id}/workflows", data={"csrf_token": csrf(client),
        "model_profile_version_id": view.version_id, "requested_chapters": "1", "provider_consent": "yes"}, follow_redirects=False)
    assert start.status_code == 303
    path = start.headers["location"]
    with client.app.state.session_factory() as session:
        workflow = session.get(GenerationWorkflow, path.rsplit("/", 1)[-1])
        assert workflow.model_name == model
        audit = session.scalar(select(AuditEvent).where(AuditEvent.action == "workflow_started"))
        assert audit.details["model_name"] == model
    client.post(path + "/run", data={"csrf_token": csrf(client), "provider_consent": "yes"})
    assert network.calls[0][3]["payload"]["model"] == model
    assert '<img src=x' not in client.get(path).text


def test_selector_javascript_preserves_raw_model_and_resets_consent():
    node = shutil.which("node")
    assert node, "Node is required for actual JS behavior acceptance"
    script = Path(__file__).resolve().parents[1] / "src/ainovel/static/profile-selection.js"
    assert script.exists(), "profile selector behavior script is missing"
    harness = r'''
const fs = require('fs'); const vm = require('vm'); const assert = require('assert');
const raw = ' <img src=x onerror="boom()"> ';
const listeners = {};
const select = { value: '', selectedOptions: [{dataset: {}}], addEventListener: (name, fn) => listeners[name] = fn };
const consent = { checked: true, required: false };
const destination = { textContent: '', set innerHTML(value) { throw Error('HTML sink'); } };
const model = {value: 'demo', readOnly: false}; const provider = {disabled: false};
const form = {elements: {model_name: model, provider_name: provider}};
const selector = {closest: () => form, querySelector: (key) => ({'[data-profile-select]': select, '[data-profile-consent]': consent, '[data-profile-destination]': destination})[key]};
const document = {querySelectorAll: () => [selector]};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {document});
assert.equal(model.value, 'demo');
select.value = 'version'; select.selectedOptions = [{dataset: {model: raw, target: 'https://example.com/v1'}}]; listeners.change();
assert.equal(model.value, raw); assert.equal(model.readOnly, true); assert.equal(provider.disabled, true);
assert.equal(consent.checked, false); assert.equal(consent.required, true); assert(destination.textContent.includes(raw));
consent.checked = true; select.value = ''; select.selectedOptions = [{dataset:{}}]; listeners.change();
assert.equal(consent.checked, false); assert.equal(consent.required, false); assert.equal(model.value, 'demo');
assert.equal(model.readOnly, false); assert.equal(provider.disabled, false);
select.value = 'missing-version'; select.selectedOptions = [{dataset: {model: 'chosen-model', unavailable: 'yes'}}];
listeners.change();
assert.equal(select.value, 'missing-version'); assert.equal(model.value, 'chosen-model');
assert.equal(consent.checked, false); assert.equal(consent.required, true);
assert.equal(provider.disabled, true); assert(destination.textContent.includes('不可用'));
const retrySelect = {value: 'version', selectedOptions: [{dataset: {model: 'saved-model', target: 'https://example.com/v1'}}], addEventListener: () => {}};
const retryModel = {value: 'submitted-model', readOnly: false};
const retryProvider = {disabled: false};
const retryConsent = {checked: true, required: false};
const retryForm = {elements: {model_name: retryModel, provider_name: retryProvider}};
const retrySelector = {closest: () => retryForm, querySelector: (key) => ({'[data-profile-select]': retrySelect, '[data-profile-consent]': retryConsent, '[data-profile-destination]': destination})[key]};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {document: {querySelectorAll: () => [retrySelector]}});
assert.equal(retryModel.value, 'submitted-model'); assert.equal(retryModel.readOnly, true);
assert.equal(retryProvider.disabled, true); assert.equal(retryConsent.checked, false);
'''
    result = subprocess.run([node, "-e", harness, str(script)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def ready_project(client):
    with client.app.state.session_factory() as session:
        project = ProjectService(session).create("Novel", 2000000, 5000000)
        ProjectService(session).add_constitution(project.id, {"genre": "fantasy"}, author_approved=True)
        outline = OutlineService(session).create_candidate(project.id, [OutlineNodeInput(
            key="book", parent_key=None, kind="book", title="Book", order=0)], reason="author")
        OutlineService(session).approve(outline.id)
        return project.id


def test_project_selection_and_run_require_destination_consent(profile_client):
    client, _, network = profile_client
    save(client)
    project_id = ready_project(client)
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    path = f"/projects/{project_id}"
    page = client.get(path)
    assert view.version_id in page.text and view.base_url in page.text
    data = {"csrf_token": csrf(client, path), "model_profile_version_id": view.version_id, "requested_chapters": "1"}
    assert client.post(path + "/workflows", data=data).status_code == 422
    created = client.post(path + "/workflows", data={**data, "provider_consent": "yes"}, follow_redirects=False)
    assert created.status_code == 303
    workflow_path = created.headers["location"]
    assert view.base_url in client.get(workflow_path).text
    assert client.post(workflow_path + "/run", data={"csrf_token": csrf(client)}).status_code == 422
    assert not network.calls
    with client.app.state.session_factory() as session:
        workflow = session.get(GenerationWorkflow, workflow_path.rsplit("/", 1)[-1])
        assert workflow.model_profile_version_id == view.version_id
        assert workflow.model_name == "model-a" and workflow.provider_name == "compatible"


@pytest.mark.parametrize("mode", ["single_chapter", "hierarchical"])
def test_chapter_setup_uses_selected_version_and_injected_vault(database_url, mode):
    from ainovel.chapter_test import create_chapter_test_app
    try:
        app = create_chapter_test_app(database_url, profile_vault=Vault(), profile_transport=Transport())
    except TypeError:
        pytest.fail("chapter test app does not forward profile dependency injection")
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        save(client)
        with app.state.session_factory() as session:
            view = app.state.model_profile_service_factory(session).list_public()[0]
        page = client.get("/chapter-test")
        assert view.version_id in page.text
        submission = unescape(re.search(r'name="submission_token" value="([^"]+)"', page.text).group(1))
        data = dict(project_title="Novel", setting_style="Fantasy", provisional_ending="End", book_outline="Book",
                    chapter_outline="Chapter", stage_architecture="Stage", setup_mode=mode, author_confirm="yes",
                    csrf_token=csrf(client), submission_token=submission, model_profile_version_id=view.version_id)
        assert client.post("/chapter-test", data=data).status_code == 422
        created = client.post("/chapter-test", data={**data, "provider_consent": "yes"}, follow_redirects=False)
        assert created.status_code == 303
        with app.state.session_factory() as session:
            bound = session.scalar(select(GenerationWorkflow if mode == "single_chapter" else StageRoadmapVersion))
            assert bound.model_profile_version_id == view.version_id
            assert bound.model_name == "model-a"
        assert not app.state.provider_resolver.transport.calls


def test_stage_proposal_resolves_selection_and_rejects_missing_consent(profile_client):
    from ainovel.services.stages import StageService
    client, _, network = profile_client
    save(client)
    project_id = ready_project(client)
    with client.app.state.session_factory() as session:
        stage = StageService(session).create(project_id, "Architecture", "author")
        stage_id = stage.id
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    path = f"/stages/{stage_id}"
    assert view.version_id in client.get(path).text
    data = {"csrf_token": csrf(client), "model_profile_version_id": view.version_id, "author_confirm": "yes"}
    assert client.post(path + "/roadmaps", data=data).status_code == 422
    assert client.post(path + "/roadmaps", data={**data, "provider_consent": "yes"}, follow_redirects=False).status_code == 303
    with client.app.state.session_factory() as session:
        roadmap = session.scalar(select(StageRoadmapVersion))
        assert roadmap.model_profile_version_id == view.version_id
        roadmap_id = roadmap.id
    assert view.base_url in client.get(path).text
    assert client.post(path + f"/roadmaps/{roadmap_id}/generate", data={**data, "model_call_confirm": "yes"}).status_code == 422
    assert not network.calls


def test_offline_complete_author_flow_revision_isolation_revoke_and_secret_scans(profile_client, caplog):
    from test_chapter_test import provider_script
    from ainovel.models.batch import Chapter
    from ainovel.models.audit import AuditEvent
    from ainovel.models.workflow import WorkflowArtifact
    from ainovel.models.prompt import WorkflowPromptSnapshot

    client, _, network = profile_client
    responses = iter(provider_script())
    original_request = network.request_json

    def scripted(endpoint, method, path, **kwargs):
        if path == "models":
            return original_request(endpoint, method, path, **kwargs)
        network.calls.append((endpoint, method, path, kwargs))
        response = next(responses)
        return {"id": response.provider_response_id, "usage": {"prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens}, "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(response.structured, ensure_ascii=False)}}]}

    network.request_json = scripted
    location = save(client)
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    token = csrf(client)
    assert client.post(f"/model-profiles/versions/{view.version_id}/models", data={"csrf_token": token, "confirm": "yes"}).status_code == 200
    project_id = ready_project(client)
    data = {"csrf_token": token, "model_profile_version_id": view.version_id, "requested_chapters": "1", "provider_consent": "yes"}
    start = client.post(f"/projects/{project_id}/workflows", data=data, follow_redirects=False)
    assert start.status_code == 303
    path = start.headers["location"]
    html = [client.get(path).text]
    assert client.post(path + "/run", data={"csrf_token": token, "provider_consent": "yes"}).status_code == 200
    with client.app.state.session_factory() as session:
        workflow = session.get(GenerationWorkflow, path.rsplit("/", 1)[-1])
        assert workflow.status == "AWAITING_PLAN_APPROVAL"
    assert len(network.calls) == 2
    # Revision must not silently redirect the already-created workflow.
    revised = client.post(location + "/revise", data={**FORM, "api_key": "", "keep_existing_key": "yes",
        "model_name": "model-b", "csrf_token": token})
    assert revised.status_code == 200
    assert client.post(path + "/plan/approve", data={"csrf_token": token}).status_code == 200
    assert len(network.calls) == 2
    html.append(client.post(path + "/run", data={"csrf_token": token, "provider_consent": "yes"}).text)
    with client.app.state.session_factory() as session:
        workflow = session.get(GenerationWorkflow, path.rsplit("/", 1)[-1])
        assert workflow.status == "AWAITING_CONTENT_APPROVAL"
        assert workflow.model_profile_version_id == view.version_id
        batch_id = workflow.candidate_batch_id
    assert len(network.calls) == 5
    assert all(call[3]["payload"]["model"] == "model-a" for call in network.calls if call[2] == "chat/completions")
    html.append(client.get(f"/batches/{batch_id}").text)
    assert client.post(f"/batches/{batch_id}/approve", data={"csrf_token": token}).status_code == 200
    assert client.post(path + "/reconcile", data={"csrf_token": token}).status_code == 200
    with client.app.state.session_factory() as session:
        assert session.get(GenerationWorkflow, path.rsplit("/", 1)[-1]).status == "COMPLETED"
        approved = session.scalars(select(Chapter)).all()
        assert len(approved) == 1
        # No export endpoint exists: serialize the actual official chapter fields,
        # and scan the complete database dump as the broader persistence boundary.
        exported = json.dumps([{"title": c.title, "body": c.body} for c in approved], ensure_ascii=False)
        for model in [AuditEvent, WorkflowArtifact, WorkflowPromptSnapshot]:
            assert session.scalars(select(model)).first() is not None
    second = client.post(f"/projects/{project_id}/workflows", data=data, follow_redirects=False)
    assert second.status_code == 303
    assert client.post(location + "/revoke", data={"csrf_token": token, "confirm": "yes"}).status_code == 200
    paused = client.post(second.headers["location"] + "/run", data={"csrf_token": token, "provider_consent": "yes"})
    html.append(paused.text)
    with client.app.state.session_factory() as session:
        assert session.get(GenerationWorkflow, second.headers["location"].rsplit("/", 1)[-1]).status == "PAUSED_PROVIDER"
    assert len(network.calls) == 5
    with client.app.state.engine.connect() as connection:
        dump = "\n".join(connection.connection.driver_connection.iterdump())
    database_bytes = Path(client.app.state.engine.url.database).read_bytes()
    assert SECRET not in dump + exported + caplog.text + "".join(html)
    assert SECRET.encode() not in database_bytes
    assert all(SECRET not in json.dumps(call[3].get("payload")) for call in network.calls)


def test_local_key_optional_changed_target_key_isolation_and_disabled_selection(profile_client):
    client, vault, network = profile_client
    location = save(client)
    token = csrf(client)
    assert client.post(location + "/revise", data={**FORM, "api_key": "", "keep_existing_key": "yes",
        "base_url": "https://other.example/v1", "csrf_token": token}).status_code == 422
    assert client.post(location + "/revise", data={**FORM, "api_key": "", "keep_existing_key": "yes",
        "base_url": "http://127.0.0.1:8000/v1", "connection_kind": "loopback", "csrf_token": token}).status_code == 422
    local = {**FORM, "api_key": "", "connection_kind": "loopback", "base_url": "http://127.0.0.1:8000/v1", "csrf_token": token}
    saved = client.post("/model-profiles", data=local, follow_redirects=False)
    assert saved.status_code == 303
    assert client.post(saved.headers["location"] + "/revise", data={**local, "base_url": "http://127.0.0.1:8001/v1"}).status_code == 200
    with client.app.state.session_factory() as session:
        remote = client.app.state.model_profile_service_factory(session).list_public()[0]
    assert client.post(f"/model-profiles/versions/{remote.version_id}/enabled", data={"csrf_token": token,
        "confirm": "yes", "enabled": "no"}).status_code == 200
    project_id = ready_project(client)
    assert remote.version_id not in client.get(f"/projects/{project_id}").text
    assert client.post(f"/projects/{project_id}/workflows", data={"csrf_token": token, "provider_consent": "yes",
        "model_profile_version_id": remote.version_id, "requested_chapters": "1"}).status_code == 422
    assert not network.calls


def test_previous_versions_remain_visible_and_can_be_disabled(profile_client):
    client, _, network = profile_client
    location = save(client)
    with client.app.state.session_factory() as session:
        old = client.app.state.model_profile_service_factory(session).list_public()[0]
    revised = client.post(location + "/revise", data={**FORM, "api_key": "", "keep_existing_key": "yes",
        "model_name": "model-b", "csrf_token": csrf(client)})
    assert revised.status_code == 200
    assert f'/versions/{old.version_id}/enabled' in revised.text
    assert client.post(f"/model-profiles/versions/{old.version_id}/enabled", data={"csrf_token": csrf(client),
        "enabled": "no", "confirm": "yes"}).status_code == 200
    with client.app.state.session_factory() as session:
        service = client.app.state.model_profile_service_factory(session)
        assert not service.get_public(old.version_id).enabled
        assert service.list_public()[0].enabled
    assert not network.calls


@pytest.mark.parametrize("operation", ["models", "test"])
def test_provider_key_echo_is_never_rendered_or_logged(profile_client, caplog, operation):
    client, _, network = profile_client
    save(client)
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    network.models = [SECRET]
    if operation == "test":
        network.error = True
    response = client.post(f"/model-profiles/versions/{view.version_id}/{operation}",
        data={"csrf_token": csrf(client), "confirm": "yes"})
    assert response.status_code == 422
    assert SECRET not in response.text + caplog.text
    assert 'name="model_name"' in response.text


@pytest.mark.parametrize("operation", ["models", "test"])
def test_blocked_diagnostic_allows_health_and_revoke(profile_client, operation):
    client, _, network = profile_client
    save(client)
    with client.app.state.session_factory() as session:
        view = client.app.state.model_profile_service_factory(session).list_public()[0]
    token = csrf(client)
    entered, release = Event(), Event()
    original_request = network.request_json

    def blocked_request(endpoint, method, path, **kwargs):
        entered.set()
        if not release.wait(5):
            raise AssertionError("diagnostic was not released")
        return original_request(endpoint, method, path, **kwargs)

    network.request_json = blocked_request
    with ThreadPoolExecutor(max_workers=3) as pool:
        diagnostic = pool.submit(client.post, f"/model-profiles/versions/{view.version_id}/{operation}",
                                 data={"csrf_token": token, "confirm": "yes"})
        try:
            assert entered.wait(2)
            health = pool.submit(client.get, "/health")
            revoke = pool.submit(client.post, f"/model-profiles/{view.profile_id}/revoke",
                                 data={"csrf_token": token, "confirm": "yes"})
            completed, _ = wait((health, revoke), timeout=1)
            assert len(completed) == 2, "diagnostic blocked an independent request"
        finally:
            release.set()
        assert health.result().json() == {"status": "ready"}
        assert revoke.result().status_code == 200
        assert diagnostic.result().status_code in (200, 422)


def test_local_check_and_save_never_resolve_dns(profile_client, monkeypatch):
    client, _, network = profile_client
    import socket
    dns_calls = []
    def forbidden(*args, **kwargs):
        dns_calls.append(True)
        raise AssertionError("unexpected DNS")
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    data = {**FORM, "csrf_token": csrf(client)}
    assert client.post("/model-profiles/check", data=data).status_code == 200
    assert client.post("/model-profiles", data=data).status_code == 200
    assert not dns_calls and not network.calls


class SetupSelects(HTMLParser):
    """Read the selected option as a browser would, defaulting to the first."""
    def __init__(self, html):
        super().__init__()
        self.name = None
        self.values = {}
        self.options = {}
        self.inputs = {}
        self.textareas = {}
        self.textarea_name = None
        self.consent_checked = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.name = attrs.get("name")
        elif tag == "option" and self.name:
            self.options.setdefault(self.name, []).append(attrs)
            if self.name not in self.values or "selected" in attrs:
                self.values[self.name] = attrs.get("value", "")
        elif tag == "input" and attrs.get("name") == "provider_consent":
            self.consent_checked = "checked" in attrs
        elif tag == "input" and attrs.get("name"):
            self.inputs.setdefault(attrs["name"], attrs.get("value", ""))
        elif tag == "textarea":
            self.textarea_name = attrs.get("name")

    def handle_data(self, data):
        if self.textarea_name:
            self.textareas[self.textarea_name] = self.textareas.get(self.textarea_name, "") + data

    def handle_endtag(self, tag):
        if tag == "select":
            self.name = None
        elif tag == "textarea":
            self.textarea_name = None


@pytest.fixture
def setup_client(database_url):
    from ainovel.chapter_test import create_chapter_test_app
    app = create_chapter_test_app(database_url, profile_vault=Vault(), profile_transport=Transport())
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        yield client


def setup_form(client, **changes):
    page = client.get("/chapter-test")
    submission = unescape(re.search(r'name="submission_token" value="([^"]+)"', page.text).group(1))
    return dict(project_title="Novel", setting_style="Fantasy", provisional_ending="End", book_outline="Book",
                chapter_outline="Chapter", stage_architecture="Stage", setup_mode="single_chapter", author_confirm="yes",
                csrf_token=csrf(client), submission_token=submission, **changes)


@pytest.mark.parametrize("provider", ["compatible", "compatible_revised", "fake", "qwen"])
def test_setup_validation_error_roundtrips_explicit_model_selection(setup_client, provider):
    client = setup_client
    version_id = ""
    revised = provider == "compatible_revised"
    if revised:
        provider = "compatible"
    if provider == "compatible":
        location = save(client)
        with client.app.state.session_factory() as session:
            version_id = client.app.state.model_profile_service_factory(session).list_public()[0].version_id
        if revised:
            assert client.post(location + "/revise", data={**FORM, "model_name": "model-b", "api_key": "",
                "keep_existing_key": "yes", "csrf_token": csrf(client)}).status_code == 200
    data = setup_form(client, provider_name="qwen" if provider == "compatible" else provider,
                      model_name="model-a" if version_id else "custom-model", model_profile_version_id=version_id,
                      provider_consent="yes")
    data["project_title"] = ""
    invalid = client.post("/chapter-test", data=data)
    assert invalid.status_code == 422
    rendered = SetupSelects(invalid.text)
    assert rendered.values["model_profile_version_id"] == version_id
    if version_id:
        selected = next(option for option in rendered.options["model_profile_version_id"] if option["value"] == version_id)
        assert selected.get("data-unavailable") != "yes"
    if not version_id:
        assert rendered.values["provider_name"] == provider
    assert not rendered.consent_checked
    corrected = {**data, **rendered.values, "project_title": "Corrected", "provider_consent": ""}
    if version_id:
        assert client.post("/chapter-test", data=corrected).status_code == 422
        corrected["provider_consent"] = "yes"
    result = client.post("/chapter-test", data=corrected, follow_redirects=False)
    assert result.status_code == 303
    with client.app.state.session_factory() as session:
        workflow = session.scalar(select(GenerationWorkflow))
        assert workflow.provider_name == provider
        assert workflow.model_profile_version_id == (version_id or None)
        assert workflow.model_name == data["model_name"]
    assert not client.app.state.provider_resolver.transport.calls


@pytest.mark.parametrize("state", ["disabled", "missing"])
def test_unavailable_setup_selection_survives_errors_without_fallback(setup_client, state):
    client = setup_client
    version_id = "missing-profile-version"
    if state == "disabled":
        save(client)
        with client.app.state.session_factory() as session:
            service = client.app.state.model_profile_service_factory(session)
            version_id = service.list_public()[0].version_id
            service.set_enabled(version_id, False)
    data = setup_form(client, model_profile_version_id=version_id, model_name="chosen-model", provider_consent="yes")
    response = client.post("/chapter-test", data=data)
    assert response.status_code == 422
    rendered = SetupSelects(response.text)
    assert rendered.values["model_profile_version_id"] == version_id
    selected = next(option for option in rendered.options["model_profile_version_id"] if option["value"] == version_id)
    assert selected.get("data-unavailable") == "yes" and "disabled" not in selected
    assert not rendered.consent_checked
    assert client.post("/chapter-test", data={**data, **rendered.values}).status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalar(select(GenerationWorkflow)) is None
    assert not client.app.state.provider_resolver.transport.calls


@pytest.mark.parametrize("destination", ["project", "stage"])
@pytest.mark.parametrize("state", ["current", "historical", "disabled", "revoked", "missing"])
def test_creation_error_retains_saved_destination(profile_client, destination, state):
    from ainovel.services.stages import StageService

    client, _, network = profile_client
    location = save(client)
    project_id = ready_project(client)
    with client.app.state.session_factory() as session:
        service = client.app.state.model_profile_service_factory(session)
        view = service.list_public()[0]
        version_id = view.version_id
        if state == "disabled":
            service.set_enabled(version_id, False)
        elif state == "revoked":
            service.revoke(view.profile_id)
        if destination == "stage":
            stage_id = StageService(session).create(project_id, "Architecture", "author").id
    if state == "historical":
        revised = client.post(location + "/revise", data={**FORM, "model_name": "model-b", "api_key": "",
            "keep_existing_key": "yes", "csrf_token": csrf(client)})
        assert revised.status_code == 200
    if state == "missing":
        version_id = "missing-profile-version"

    path = f"/projects/{project_id}/workflows" if destination == "project" else f"/stages/{stage_id}/roadmaps"
    data = {"csrf_token": csrf(client), "model_profile_version_id": version_id,
            "provider_name": "qwen", "model_name": "chosen-model", "provider_consent": "yes",
            "requested_chapters": "bad" if state in {"current", "historical"} else "1",
            "architecture": "Revised architecture", "author_confirm": "" if state in {"current", "historical"} else "yes"}
    response = client.post(path, data=data)
    assert response.status_code == 422
    rendered = SetupSelects(response.text)
    assert rendered.values["model_profile_version_id"] == version_id
    assert rendered.values["provider_name"] == "qwen"
    assert rendered.inputs["model_name"] == "chosen-model"
    assert not rendered.consent_checked
    selected = next(option for option in rendered.options["model_profile_version_id"] if option["value"] == version_id)
    assert (selected.get("data-unavailable") == "yes") == (state in {"disabled", "revoked", "missing"})
    if destination == "project":
        assert rendered.inputs["requested_chapters"] == data["requested_chapters"]
    else:
        assert rendered.textareas["architecture"] == "Revised architecture"
    if state in {"disabled", "revoked", "missing"}:
        assert client.post(path, data=data).status_code == 422
        with client.app.state.session_factory() as session:
            assert session.scalar(select(GenerationWorkflow if destination == "project" else StageRoadmapVersion)) is None
    else:
        corrected = {**data, "provider_consent": "", "requested_chapters": "1", "author_confirm": "yes"}
        assert client.post(path, data=corrected).status_code == 422
        assert client.post(path, data={**corrected, "provider_consent": "yes"}, follow_redirects=False).status_code == 303
        with client.app.state.session_factory() as session:
            created = session.scalar(select(GenerationWorkflow if destination == "project" else StageRoadmapVersion))
            assert created.model_profile_version_id == version_id
            assert (created.provider_name, created.model_name) == ("compatible", "model-a")
    assert not network.calls


@pytest.mark.parametrize("destination", ["project", "stage"])
@pytest.mark.parametrize("provider", ["fake", "qwen"])
def test_creation_error_retains_explicit_legacy_destination(profile_client, destination, provider):
    from ainovel.services.stages import StageService

    client, _, network = profile_client
    project_id = ready_project(client)
    if destination == "stage":
        with client.app.state.session_factory() as session:
            stage_id = StageService(session).create(project_id, "Architecture", "author").id
    path = f"/projects/{project_id}/workflows" if destination == "project" else f"/stages/{stage_id}/roadmaps"
    data = {"csrf_token": csrf(client), "model_profile_version_id": "", "provider_name": provider,
            "model_name": "custom-model", "requested_chapters": "bad", "author_confirm": "",
            "architecture": "Revised architecture"}
    response = client.post(path, data=data)
    assert response.status_code == 422
    rendered = SetupSelects(response.text)
    assert rendered.values["model_profile_version_id"] == ""
    assert rendered.values["provider_name"] == provider
    assert rendered.inputs["model_name"] == "custom-model"
    assert not rendered.consent_checked
    if destination == "project":
        assert rendered.inputs["requested_chapters"] == "bad"
        data["requested_chapters"] = "1"
    else:
        assert rendered.textareas["architecture"] == "Revised architecture"
        data["author_confirm"] = "yes"
    assert client.post(path, data=data, follow_redirects=False).status_code == 303
    with client.app.state.session_factory() as session:
        created = session.scalar(select(GenerationWorkflow if destination == "project" else StageRoadmapVersion))
        assert (created.provider_name, created.model_name, created.model_profile_version_id) == (provider, "custom-model", None)
    assert not network.calls


@pytest.mark.parametrize("mode", ["hierarchical", "stage"])
@pytest.mark.parametrize("provider", ["fake", "qwen"])
def test_explicit_legacy_stage_setup_freezes_selected_provider_model(setup_client, mode, provider):
    client = setup_client
    data = setup_form(client, provider_name=provider, model_name="custom-stage-model")
    data["setup_mode"] = mode
    result = client.post("/chapter-test", data=data, follow_redirects=False)
    assert result.status_code == 303
    with client.app.state.session_factory() as session:
        roadmap = session.scalar(select(StageRoadmapVersion))
        assert roadmap is not None
        assert (roadmap.provider_name, roadmap.model_name) == (provider, "custom-stage-model")
        assert roadmap.model_profile_version_id is None
    stage_page = client.get(result.headers["location"])
    options = SetupSelects(stage_page.text).options["provider_name"]
    assert {option["value"] for option in options} >= {"fake", "qwen"}
    assert not client.app.state.provider_resolver.transport.calls

from dataclasses import asdict, replace
import json
import socket
from uuid import uuid4

import pytest
from sqlalchemy import text

from ainovel.services import model_profiles as profile_api

SENTINEL = "SENTINEL_KEY_42"


class MemoryVault:
    """Test-only credential boundary; never shipped with application code."""
    def __init__(self):
        self.secrets = {}
        self.fail_put = False
        self.fail_delete = False

    def put(self, secret):
        if self.fail_put:
            raise RuntimeError(SENTINEL)
        reference = str(uuid4())
        self.secrets[reference] = secret
        return reference

    def get(self, reference):
        return self.secrets[reference]

    def delete(self, reference):
        if self.fail_delete:
            raise RuntimeError(SENTINEL)
        self.secrets.pop(reference, None)


@pytest.fixture
def api():
    return profile_api


@pytest.fixture
def setup(session, api):
    vault = MemoryVault()
    service = api.ModelProfileService(session, vault=vault)
    values = api.ProfileInput("Test", "https://EXAMPLE.com:443/v1/", "remote", "model-a")
    return service, values, vault


def test_public_and_sql_are_secret_free_and_save_never_resolves(setup, session, monkeypatch, caplog):
    service, values, vault = setup
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("save must not resolve DNS"))
    created = service.create(values, api_key=SENTINEL)
    public = json.dumps(asdict(created))
    assert created.has_key and created.base_url == "https://example.com/v1"
    assert SENTINEL not in public + repr(created) + caplog.text
    assert not any(ref in public for ref in vault.secrets)
    assert service.list_public() == [created]
    dump = "\n".join(session.connection().connection.driver_connection.iterdump())
    assert SENTINEL not in dump
    resolved = service.resolve_for_call(created.version_id)
    assert resolved.api_key == SENTINEL
    assert SENTINEL not in repr(resolved)
    with pytest.raises(TypeError):
        json.dumps(resolved)


def test_revisions_keep_metadata_and_credentials_bound_to_old_version(setup):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    second = service.revise(first.profile_id, replace(values, model_name="model-b"), keep_existing_key=True)
    assert second.version_id != first.version_id
    assert service.get_public(first.version_id).model_name == "model-a"
    assert service.get_public(second.version_id).model_name == "model-b"
    assert len(vault.secrets) == 2
    third = service.revise(first.profile_id, values, api_key="OTHER_SENTINEL")
    assert service.resolve_for_call(first.version_id).api_key == SENTINEL
    assert service.resolve_for_call(third.version_id).api_key == "OTHER_SENTINEL"
    assert service.list_public() == [third]


@pytest.mark.parametrize("changes,key", [
    ({"name": ""}, SENTINEL), ({"name": "x" * 121}, SENTINEL),
    ({"name": SENTINEL}, SENTINEL), ({"model_name": SENTINEL}, SENTINEL),
    ({"model_name": "x" * 256}, SENTINEL), ({"model_name": ""}, SENTINEL),
    ({"context_limit": 32001}, SENTINEL), ({"context_limit": 0}, SENTINEL),
    ({"output_limit": 12001}, SENTINEL), ({"output_limit": 0}, SENTINEL),
    ({"context_limit": 100, "output_limit": 101}, SENTINEL),
    ({"output_limit": True}, SENTINEL), ({}, None), ({}, ""), ({}, "x" * 8193),
    ({"base_url": "http://example.com/v1"}, SENTINEL),
    ({"base_url": "https://user:pass@example.com"}, SENTINEL),
    ({"base_url": "https://example.com?token=" + SENTINEL}, SENTINEL),
    ({"base_url": "https://example.com/#fragment"}, SENTINEL),
    ({"base_url": "https://example.com:65536"}, SENTINEL),
    ({"base_url": "https://example.com/" + "a" * 2048}, SENTINEL),
    ({"base_url": "https://example.com\\@evil.com"}, SENTINEL),
    ({"base_url": "https://example.com\n"}, SENTINEL),
    ({"connection_kind": "other"}, SENTINEL),
    ({"connection_kind": "loopback", "base_url": "http://192.168.1.1"}, None),
], ids=[f"invalid-{n}" for n in range(25)])
def test_invalid_input_is_rejected_without_secret_persistence(setup, changes, key):
    service, values, vault = setup
    with pytest.raises(ValueError) as error:
        service.create(replace(values, **changes), api_key=key)
    assert SENTINEL not in str(error.value)
    assert not vault.secrets and service.list_public() == []


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434/v1", "http://[::1]:11434/v1", "http://localhost/v1"])
def test_explicit_loopback_can_omit_key(setup, url):
    service, values, vault = setup
    created = service.create(replace(values, base_url=url, connection_kind="loopback"), api_key=None)
    assert not created.has_key
    assert service.resolve_for_call(created.version_id).api_key is None
    assert not vault.secrets


@pytest.mark.parametrize("changes", [{"base_url": "https://other.example/v1"}, {"base_url": "https://example.com/v2"}, {"base_url": "http://localhost/v1", "connection_kind": "loopback"}])
def test_changed_target_requires_new_key_even_with_reuse_opt_in(setup, changes):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    with pytest.raises(ValueError):
        service.revise(first.profile_id, replace(values, **changes), keep_existing_key=True)
    assert len(vault.secrets) == 1


def test_reuse_requires_opt_in_and_metadata_cannot_equal_reused_secret(setup):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    with pytest.raises(ValueError):
        service.revise(first.profile_id, values)
    with pytest.raises(ValueError):
        service.revise(first.profile_id, replace(values, name=SENTINEL), keep_existing_key=True)
    assert len(vault.secrets) == 1


def test_disable_reenable_and_irreversible_revoke(setup):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    service.set_enabled(first.version_id, False)
    with pytest.raises(ValueError):
        service.resolve_for_call(first.version_id)
    service.set_enabled(first.version_id, True)
    assert service.resolve_for_call(first.version_id).api_key == SENTINEL
    service.revise(first.profile_id, values, keep_existing_key=True)
    service.revoke(first.profile_id)
    assert not vault.secrets
    assert service.get_public(first.version_id).revoked
    assert not service.get_public(first.version_id).has_key
    with pytest.raises(ValueError):
        service.set_enabled(first.version_id, True)
    with pytest.raises(ValueError):
        service.resolve_for_call(first.version_id)
    with pytest.raises(ValueError):
        service.revise(first.profile_id, values, api_key=SENTINEL)


def test_missing_credential_cannot_be_reenabled(setup):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    service.set_enabled(first.version_id, False)
    vault.secrets.clear()
    with pytest.raises(ValueError):
        service.set_enabled(first.version_id, True)
    assert not service.get_public(first.version_id).enabled


def test_failed_commit_compensates_vault_and_hides_error(setup, session, monkeypatch):
    service, values, vault = setup
    def fail():
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(session, "commit", fail)
    with pytest.raises(ValueError) as error:
        service.create(values, api_key=SENTINEL)
    assert SENTINEL not in str(error.value)
    assert vault.secrets == {}
    assert service.list_public() == []


def test_encryption_failure_leaves_no_metadata(setup):
    service, values, vault = setup
    vault.fail_put = True
    with pytest.raises(ValueError) as error:
        service.create(values, api_key=SENTINEL)
    assert SENTINEL not in str(error.value)
    assert service.list_public() == []


def test_revocation_commits_before_cleanup_and_cleanup_failure_stays_revoked(setup, session):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    def failing_delete(reference):
        from sqlalchemy.orm import Session
        with Session(session.bind) as fresh:
            assert fresh.execute(text("SELECT revoked FROM model_profiles")).scalar_one()
        raise RuntimeError(SENTINEL)
    vault.delete = failing_delete
    with pytest.raises(ValueError) as error:
        service.revoke(first.profile_id)
    assert SENTINEL not in str(error.value)
    assert service.get_public(first.version_id).revoked
    with pytest.raises(ValueError):
        service.resolve_for_call(first.version_id)


def test_failed_revocation_commit_does_not_destroy_credentials(setup, session, monkeypatch):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    def fail():
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(session, "commit", fail)
    with pytest.raises(ValueError):
        service.revoke(first.profile_id)
    assert service.resolve_for_call(first.version_id).api_key == SENTINEL


def test_resolve_rechecks_authorization_across_sessions(setup, session, api):
    from sqlalchemy.orm import Session
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    service.resolve_for_call(first.version_id)
    with Session(session.bind) as other:
        api.ModelProfileService(other, vault=vault).revoke(first.profile_id)
    with pytest.raises(ValueError):
        service.resolve_for_call(first.version_id)


def test_changed_keyless_loopback_target_can_remain_keyless(setup):
    service, values, vault = setup
    local = replace(values, connection_kind="loopback", base_url="http://localhost:11434/v1")
    first = service.create(local, api_key=None)
    second = service.revise(first.profile_id, replace(local, base_url="http://localhost:11435/v1"))
    assert second.base_url == "http://localhost:11435/v1" and not second.has_key
    assert not vault.secrets


def test_upper_bounds_and_version_metadata_cannot_be_changed_in_place(setup, session):
    from ainovel.models.model_profile import ModelProfileVersion
    service, values, vault = setup
    first = service.create(replace(values, name="n" * 120, model_name="m" * 255), api_key="k" * 8192)
    assert first.context_limit == 32000 and first.output_limit == 12000
    row = session.get(ModelProfileVersion, first.version_id)
    row.model_name = "changed"
    with pytest.raises(ValueError):
        session.commit()
    session.rollback()
    assert service.get_public(first.version_id).model_name == "m" * 255


def test_failed_revision_preserves_old_version_and_removes_new_credential(setup, session, monkeypatch):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    def fail():
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(session, "commit", fail)
    with pytest.raises(ValueError):
        service.revise(first.profile_id, replace(values, model_name="new"), keep_existing_key=True)
    assert len(vault.secrets) == 1
    assert service.list_public() == [first]


def test_cleanup_failure_reports_safe_error_and_revoke_can_retry(setup, session, monkeypatch):
    service, values, vault = setup
    first = service.create(values, api_key=SENTINEL)
    vault.fail_delete = True
    with pytest.raises(ValueError) as error:
        service.revoke(first.profile_id)
    assert SENTINEL not in str(error.value)
    assert service.get_public(first.version_id).revoked
    vault.fail_delete = False
    service.revoke(first.profile_id)
    assert not vault.secrets


def test_credential_cannot_be_embedded_in_base_url(setup):
    service, values, vault = setup
    with pytest.raises(ValueError):
        service.create(replace(values, base_url="https://example.com/" + SENTINEL), api_key=SENTINEL)
    assert not vault.secrets

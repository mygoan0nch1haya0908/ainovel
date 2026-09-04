import pytest
from sqlalchemy.exc import IntegrityError

from ainovel.models.project import ConstitutionVersion
from ainovel.services.projects import ProjectService


def test_create_project_uses_confirmed_length_range(session) -> None:
    project = ProjectService(session).create("星海问道", 2_000_000, 5_000_000)

    assert project.title == "星海问道"
    assert project.target_chars_min == 2_000_000
    assert project.target_chars_max == 5_000_000
    assert project.official_outline_version_id is None
    assert project.active_batch_id is None
    assert project.next_batch_sequence == 1


def test_project_rejects_reversed_length_range(session) -> None:
    service = ProjectService(session)

    with pytest.raises(ValueError, match="minimum target must not exceed maximum"):
        service.create("错误项目", 5_000_000, 2_000_000)


def test_get_returns_project_by_id(session) -> None:
    created = ProjectService(session).create("星海问道", 2_000_000, 5_000_000)

    project = ProjectService(session).get(created.id)

    assert project.id == created.id


def test_add_constitution_increments_version_numbers(session, project) -> None:
    service = ProjectService(session)

    first = service.add_constitution(project.id, {"tone": "warm"}, False)
    second = service.add_constitution(project.id, {"tone": "dark"}, False)

    assert first.version_number == 1
    assert second.version_number == 2


def test_only_approved_constitution_updates_project_pointer(session, project) -> None:
    service = ProjectService(session)

    draft = service.add_constitution(project.id, {"tone": "warm"}, False)
    assert service.get(project.id).current_constitution_version_id is None

    approved = service.add_constitution(project.id, {"tone": "dark"}, True)

    assert service.get(project.id).current_constitution_version_id == approved.id
    assert draft.id != approved.id


def test_constitution_persists_after_prior_read_and_session_reopen(client, project) -> None:
    with client.app.state.session_factory() as first_session:
        service = ProjectService(first_session)
        service.get(project.id)
        created = service.add_constitution(project.id, {"tone": "dark"}, True)

    with client.app.state.session_factory() as reopened_session:
        persisted = ProjectService(reopened_session).get(project.id)
        version = reopened_session.get(ConstitutionVersion, created.id)

    assert version is not None
    assert version.version_number == 1
    assert persisted.current_constitution_version_id == created.id


def test_constitution_retries_after_allocation_collision(session, project, monkeypatch) -> None:
    service = ProjectService(session)
    service.add_constitution(project.id, {"tone": "warm"}, False)
    original_flush = session.flush
    original_rollback = session.rollback
    flush_calls = 0
    rollback_calls = 0

    def collide_once(*args, **kwargs):
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 1:
            raise IntegrityError("insert", {}, Exception("duplicate version"))
        return original_flush(*args, **kwargs)

    def record_rollback(*args, **kwargs):
        nonlocal rollback_calls
        rollback_calls += 1
        return original_rollback(*args, **kwargs)

    monkeypatch.setattr(session, "flush", collide_once)
    monkeypatch.setattr(session, "rollback", record_rollback)

    version = service.add_constitution(project.id, {"tone": "dark"}, True)

    assert rollback_calls == 1
    assert flush_calls >= 2
    assert version.version_number == 2
    assert service.get(project.id).current_constitution_version_id == version.id

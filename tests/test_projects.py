import pytest

from ainovel.services.projects import ProjectService


def test_create_project_uses_confirmed_length_range(session) -> None:
    project = ProjectService(session).create("星海问道", 2_000_000, 5_000_000)

    assert project.title == "星海问道"
    assert project.target_chars_min == 2_000_000
    assert project.target_chars_max == 5_000_000
    assert project.official_outline_version_id is None


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

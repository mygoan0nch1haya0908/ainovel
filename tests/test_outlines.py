import pytest

from ainovel.models.outline import OutlineVersion
from ainovel.services.outlines import OutlineNodeInput, OutlineService


def test_candidate_outline_does_not_replace_official_project_version(session, project) -> None:
    candidate = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="initial outline",
    )

    session.refresh(project)
    assert candidate.status == "candidate"
    assert candidate.version_number == 1
    assert project.official_outline_version_id is None


def test_approved_outline_becomes_official_without_deleting_candidate_history(
    session, project
) -> None:
    service = OutlineService(session)
    first = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
        reason="initial outline",
    )

    approved = service.approve(first.id)

    session.refresh(project)
    assert approved.status == "official"
    assert project.official_outline_version_id == approved.id
    assert service.get_tree(approved.id)[0].title == "全书总纲"
    assert session.get(OutlineVersion, first.id) is not None


def test_approving_replacement_supersedes_previous_official_version(session, project) -> None:
    service = OutlineService(session)
    first = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="first", order=0)],
        reason="first",
    )
    service.approve(first.id)
    replacement = service.create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="second", order=0)],
        reason="replacement",
    )

    approved = service.approve(replacement.id)

    assert approved.status == "official"
    assert session.get(OutlineVersion, first.id).status == "superseded"
    assert session.get(OutlineVersion, replacement.id).base_version_id == first.id


def test_approved_outline_persists_after_session_reopen(client, project) -> None:
    with client.app.state.session_factory() as first_session:
        service = OutlineService(first_session)
        candidate = service.create_candidate(
            project.id,
            [OutlineNodeInput(key="book", parent_key=None, kind="book", title="全书总纲", order=0)],
            reason="initial outline",
        )
        approved = service.approve(candidate.id)

    with client.app.state.session_factory() as reopened_session:
        persisted = reopened_session.get(OutlineVersion, approved.id)
        current_project = reopened_session.get(type(project), project.id)

    assert persisted is not None
    assert persisted.status == "official"
    assert current_project.official_outline_version_id == approved.id


@pytest.mark.parametrize(
    "nodes",
    [
        [
            OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0),
            OutlineNodeInput(key="book", parent_key="book", kind="volume", title="重复", order=1),
        ],
        [OutlineNodeInput(key="stage", parent_key="missing", kind="stage", title="阶段", order=0)],
        [
            OutlineNodeInput(key="root-a", parent_key=None, kind="book", title="根A", order=0),
            OutlineNodeInput(key="root-b", parent_key=None, kind="book", title="根B", order=1),
        ],
        [
            OutlineNodeInput(key="book", parent_key="volume", kind="book", title="总纲", order=0),
            OutlineNodeInput(key="volume", parent_key="book", kind="volume", title="卷一", order=1),
        ],
    ],
)
def test_invalid_outline_tree_is_not_persisted(session, project, nodes) -> None:
    service = OutlineService(session)
    before = service.count_versions(project.id)

    with pytest.raises(ValueError):
        service.create_candidate(project.id, nodes, reason="invalid tree")

    assert service.count_versions(project.id) == before


def test_get_tree_returns_children_in_order_with_stable_key_tiebreaker(session, project) -> None:
    version = OutlineService(session).create_candidate(
        project.id,
        [
            OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0),
            OutlineNodeInput(key="volume-b", parent_key="book", kind="volume", title="乙", order=1),
            OutlineNodeInput(key="volume-a", parent_key="book", kind="volume", title="甲", order=1),
        ],
        reason="ordered tree",
    )

    tree = OutlineService(session).get_tree(version.id)

    assert [node.key for node in tree] == ["book"]
    assert [node.key for node in tree[0].children] == ["volume-a", "volume-b"]


def test_compare_reports_stable_keys_without_database_fields(session, project) -> None:
    service = OutlineService(session)
    first = service.create_candidate(
        project.id,
        [
            OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0),
            OutlineNodeInput(key="removed", parent_key="book", kind="volume", title="旧卷", order=1),
            OutlineNodeInput(key="changed", parent_key="book", kind="volume", title="旧名", order=2),
        ],
        reason="first",
    )
    second = service.create_candidate(
        project.id,
        [
            OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0),
            OutlineNodeInput(key="changed", parent_key="book", kind="volume", title="新名", order=2),
            OutlineNodeInput(key="added", parent_key="book", kind="volume", title="新卷", order=1),
        ],
        reason="second",
    )

    diff = service.compare(first.id, second.id)

    assert diff.added == ["added"]
    assert diff.removed == ["removed"]
    assert diff.changed == ["changed"]
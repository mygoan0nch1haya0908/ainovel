import pytest
from sqlalchemy import select

from ainovel.models.outline import OutlineVersion
from ainovel.services.outlines import OutlineApprovalConflict, OutlineNodeInput, OutlineService


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

def test_stale_candidate_approval_conflicts_without_creating_two_official_versions(
    client, project
) -> None:
    with client.app.state.session_factory() as setup_session:
        setup_service = OutlineService(setup_session)
        initial = setup_service.create_candidate(
            project.id,
            [OutlineNodeInput(key="book", parent_key=None, kind="book", title="initial", order=0)],
            reason="initial",
        )
        setup_service.approve(initial.id)

    with client.app.state.session_factory() as first_candidate_session:
        first_candidate = OutlineService(first_candidate_session).create_candidate(
            project.id,
            [OutlineNodeInput(key="book", parent_key=None, kind="book", title="first", order=0)],
            reason="first",
        )
        first_candidate_id = first_candidate.id

    with client.app.state.session_factory() as second_candidate_session:
        second_candidate = OutlineService(second_candidate_session).create_candidate(
            project.id,
            [OutlineNodeInput(key="book", parent_key=None, kind="book", title="second", order=0)],
            reason="second",
        )
        second_candidate_id = second_candidate.id

    with client.app.state.session_factory() as stale_session:
        stale_session.get(type(project), project.id)
        with client.app.state.session_factory() as winning_session:
            winner = OutlineService(winning_session).approve(first_candidate_id)
            winner_id = winner.id

        with pytest.raises(OutlineApprovalConflict, match="outline approval conflict"):
            OutlineService(stale_session).approve(second_candidate_id)

    with client.app.state.session_factory() as verify_session:
        persisted_project = verify_session.get(type(project), project.id)
        official_versions = verify_session.scalars(
            select(OutlineVersion).where(
                OutlineVersion.project_id == project.id,
                OutlineVersion.status == "official",
            )
        ).all()

    assert persisted_project.official_outline_version_id == winner_id
    assert [version.id for version in official_versions] == [winner_id]

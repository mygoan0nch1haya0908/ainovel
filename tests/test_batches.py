import pytest
from sqlalchemy import event, select

from ainovel.models.audit import AuditEvent
from ainovel.services.batches import BatchService
from ainovel.services.counting import count_visible_characters
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService


def test_visible_count_excludes_whitespace_but_includes_punctuation_and_ascii() -> None:
    assert count_visible_characters(" 甲，A1\n乙。 ") == 6


@pytest.mark.parametrize("planned", [0, 6])
def test_batch_size_must_be_between_one_and_five(session, project, official_outline, planned) -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        BatchService(session).create(project.id, official_outline.id, planned)


def test_batch_requires_the_projects_official_outline(session, project, official_outline) -> None:
    candidate = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="候选", order=0)],
        reason="unapproved",
    )

    with pytest.raises(ValueError, match="official"):
        BatchService(session).create(project.id, candidate.id, 1)


def test_batch_rejects_an_official_outline_from_another_project(session, official_outline) -> None:
    other_project = ProjectService(session).create("另一部小说", 10, 20)

    with pytest.raises(ValueError, match="another project"):
        BatchService(session).create(other_project.id, official_outline.id, 1)


def test_saved_candidate_chapter_does_not_become_official(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, "初临", "甲" * 4500, {"time": "+1 day"})

    assert chapter.status == "candidate"
    assert chapter.visible_char_count == 4500
    assert chapter.official_chapter_number is None
    assert batch.status == "draft"


@pytest.mark.parametrize("ordinal", [0, 2])
def test_candidate_ordinal_must_be_within_the_batches_plan(
    session, project, official_outline, ordinal
) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)

    with pytest.raises(ValueError, match="ordinal"):
        service.save_candidate_chapter(batch.id, ordinal, "越界", "正文", {})


def test_candidate_state_delta_is_copied_before_persistence(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    state_delta = {"timeline": {"day": 1}}
    chapter = service.save_candidate_chapter(batch.id, 1, "初临", "正文", state_delta)
    state_delta["timeline"]["day"] = 2
    session.expire(chapter)

    assert service.get_chapter(chapter.id).state_delta == {"timeline": {"day": 1}}


def test_duplicate_candidate_ordinal_is_not_persisted(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "初稿", "正文", {})

    with pytest.raises(ValueError, match="ordinal"):
        service.save_candidate_chapter(batch.id, 1, "重复", "另一正文", {})

    assert [chapter.title for chapter in service.list_chapters(batch.id)] == ["初稿"]


@pytest.mark.parametrize("length", [4500, 6000])
def test_boundary_length_chapter_can_be_marked_ready(session, project, official_outline, length) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "边界", "甲" * length, {})

    ready = service.mark_ready(batch.id)

    assert ready.status == "ready_for_review"


@pytest.mark.parametrize("length", [4499, 6001])
def test_out_of_range_chapter_remains_saved_but_cannot_be_ready(
    session, project, official_outline, length
) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, "越界", "甲" * length, {})

    with pytest.raises(ValueError, match="4500.*6000"):
        service.mark_ready(batch.id)

    assert service.get_chapter(chapter.id).status == "candidate"


def test_ready_requires_every_planned_ordinal(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    service.save_candidate_chapter(batch.id, 1, "第一章", "甲" * 4500, {})

    with pytest.raises(ValueError, match="all planned"):
        service.mark_ready(batch.id)

    assert service.get(batch.id).status == "draft"


def test_reject_keeps_candidate_chapter_rows(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, "保留", "正文", {})

    rejected = service.reject(batch.id, "needs rewrite")

    assert rejected.status == "rejected"
    assert service.get_chapter(chapter.id).status == "candidate"
    assert [saved.id for saved in service.list_chapters(batch.id)] == [chapter.id]


def test_list_chapters_orders_candidates_by_ordinal(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    service.save_candidate_chapter(batch.id, 2, "第二章", "正文", {})
    service.save_candidate_chapter(batch.id, 1, "第一章", "正文", {})

    assert [chapter.ordinal for chapter in service.list_chapters(batch.id)] == [1, 2]


def test_failed_approval_keeps_every_chapter_candidate(session, project, official_outline) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    service.save_candidate_chapter(batch.id, 1, "一", "甲" * 4500, {"seq": 1})
    service.save_candidate_chapter(batch.id, 2, "二", "乙" * 4500, {"seq": 2})
    service.mark_ready(batch.id)

    @event.listens_for(session, "before_commit", once=True)
    def fail_commit(_session) -> None:
        raise RuntimeError("simulated commit failure")

    with pytest.raises(RuntimeError, match="simulated commit failure"):
        service.approve(batch.id, official_outline.id)

    session.rollback()
    session.expire_all()
    chapters = service.list_chapters(batch.id)
    assert [chapter.status for chapter in chapters] == ["candidate", "candidate"]
    assert service.get(batch.id).status != "approved"
    assert session.scalars(
        select(AuditEvent).where(
            AuditEvent.entity_id == batch.id, AuditEvent.action == "batch_approved"
        )
    ).all() == []


def test_approval_persists_official_chapters_and_one_event_in_new_session(
    session, client, project, official_outline
) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    first = service.save_candidate_chapter(batch.id, 1, "一", "甲" * 4500, {"seq": 1})
    second = service.save_candidate_chapter(batch.id, 2, "二", "乙" * 4500, {"seq": 2})
    service.mark_ready(batch.id)

    service.approve(batch.id, official_outline.id, actor="editor")

    with client.app.state.session_factory() as second_session:
        persisted = BatchService(second_session).list_chapters(batch.id)
        event_record = second_session.scalar(
            select(AuditEvent).where(
                AuditEvent.entity_type == "writing_batch",
                AuditEvent.entity_id == batch.id,
                AuditEvent.action == "batch_approved",
            )
        )
        assert [chapter.status for chapter in persisted] == ["official", "official"]
        assert [chapter.official_chapter_number for chapter in persisted] == [1, 2]
        assert event_record is not None
        assert event_record.actor == "editor"
        assert event_record.details == {
            "chapter_ids": [first.id, second.id],
            "state_deltas": [{"seq": 1}, {"seq": 2}],
        }


def test_approval_requires_the_batches_existing_official_outline(
    session, project, official_outline
) -> None:
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    service.save_candidate_chapter(batch.id, 1, "一", "甲" * 4500, {})
    service.mark_ready(batch.id)
    replacement = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="新版", order=0)],
        reason="revision",
    )
    replacement = OutlineService(session).approve(replacement.id)

    with pytest.raises(ValueError, match="batch.*official outline"):
        service.approve(batch.id, replacement.id)

    assert service.get(batch.id).status == "ready_for_review"


def test_published_chapter_cannot_be_replaced(session, approved_chapter) -> None:
    service = BatchService(session)
    service.publish_chapter(approved_chapter.id)

    with pytest.raises(PermissionError, match="published chapters are frozen"):
        service.replace_candidate_body(approved_chapter.id, "新正文")


def test_official_chapter_cannot_be_replaced(session, approved_chapter) -> None:
    with pytest.raises(PermissionError, match="candidate chapters"):
        BatchService(session).replace_candidate_body(approved_chapter.id, "新正文")
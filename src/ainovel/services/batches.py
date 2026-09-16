from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import exists, func, select, update
from sqlalchemy.orm import Session

from ainovel.models.audit import AuditEvent
from ainovel.models.batch import Chapter, WritingBatch
from ainovel.models.outline import OutlineVersion
from ainovel.models.project import NovelProject
from ainovel.models.workflow import GenerationWorkflow
from ainovel.services.counting import count_visible_characters

MIN_BATCH_CHAPTERS = 1
MAX_BATCH_CHAPTERS = 5
MIN_VISIBLE_CHARACTERS = 4500
MAX_VISIBLE_CHARACTERS = 6000
CHAPTER_NUMBER_ALLOCATION_ATTEMPTS = 3


@dataclass(frozen=True)
class OfficialChapterStatistics:
    chapter_count: int
    visible_character_count: int


class BatchService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        project_id: str,
        outline_version_id: str,
        planned_chapters: int,
        *,
        source_workflow_id: str | None = None,
    ) -> WritingBatch:
        if not MIN_BATCH_CHAPTERS <= planned_chapters <= MAX_BATCH_CHAPTERS:
            raise ValueError("batch size must be between 1 and 5")
        if source_workflow_id is not None and (
            not isinstance(source_workflow_id, str) or not source_workflow_id.strip()
        ):
            raise ValueError("source workflow id must be nonblank")
        project = self.session.get(NovelProject, project_id)
        if project is None:
            self.session.rollback()
            raise ValueError("project not found")
        outline = self.session.get(OutlineVersion, outline_version_id)
        if outline is None:
            self.session.rollback()
            raise ValueError("outline version not found")
        if outline.project_id != project.id:
            self.session.rollback()
            raise ValueError("official outline belongs to another project")
        if outline.status != "official":
            self.session.rollback()
            raise ValueError("batch requires an official outline")
        batch_id = str(uuid4())
        sequence_number = project.next_batch_sequence
        workflow_guard = (
            NovelProject.active_workflow_id.is_(None)
            if source_workflow_id is None
            else (
                (NovelProject.active_workflow_id == source_workflow_id)
                & exists().where(
                    GenerationWorkflow.id == source_workflow_id,
                    GenerationWorkflow.project_id == project.id,
                    GenerationWorkflow.status == "CREATING_CANDIDATE_BATCH",
                )
            )
        )
        ownership = self.session.execute(
            update(NovelProject)
            .where(
                NovelProject.id == project.id,
                NovelProject.active_batch_id.is_(None),
                NovelProject.next_batch_sequence == sequence_number,
                NovelProject.official_outline_version_id == outline.id,
                workflow_guard,
            )
            .values(
                active_batch_id=batch_id,
                next_batch_sequence=sequence_number + 1,
            )
        )
        if ownership.rowcount != 1:
            self.session.rollback()
            raise ValueError("project already has an active batch or its state changed")
        batch = WritingBatch(
            id=batch_id,
            project_id=project.id,
            base_outline_version_id=outline.id,
            sequence_number=sequence_number,
            planned_chapters=planned_chapters,
            status="draft",
            source_workflow_id=source_workflow_id,
        )
        self.session.add(batch)
        audit_details: dict[str, object] = {
            "base_outline_version_id": outline.id,
            "planned_chapters": planned_chapters,
            "sequence_number": sequence_number,
        }
        if source_workflow_id is not None:
            audit_details["source_workflow_id"] = source_workflow_id
        self._add_audit(
            project.id,
            "writing_batch",
            batch.id,
            "batch_created",
            "author",
            audit_details,
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return batch

    def get(self, batch_id: str) -> WritingBatch:
        batch = self.session.get(WritingBatch, batch_id)
        if batch is None:
            raise ValueError("writing batch not found")
        return batch

    def get_chapter(self, chapter_id: str) -> Chapter:
        chapter = self.session.get(Chapter, chapter_id)
        if chapter is None:
            raise ValueError("chapter not found")
        return chapter

    def list_chapters(self, batch_id: str) -> list[Chapter]:
        self.get(batch_id)
        return self.session.scalars(
            select(Chapter).where(Chapter.batch_id == batch_id).order_by(Chapter.ordinal)
        ).all()

    def list_for_project(self, project_id: str) -> list[WritingBatch]:
        return self.session.scalars(
            select(WritingBatch)
            .where(WritingBatch.project_id == project_id)
            .order_by(WritingBatch.created_at.desc())
        ).all()

    def official_chapter_statistics(self, project_id: str) -> OfficialChapterStatistics:
        chapter_count, visible_character_count = self.session.execute(
            select(
                func.count(Chapter.id),
                func.coalesce(func.sum(Chapter.visible_char_count), 0),
            ).where(
                Chapter.project_id == project_id,
                Chapter.official_chapter_number.is_not(None),
            )
        ).one()
        return OfficialChapterStatistics(chapter_count, visible_character_count)

    def list_audit_events(self, project_id: str) -> list[AuditEvent]:
        return self.session.scalars(
            select(AuditEvent)
            .where(AuditEvent.project_id == project_id)
            .order_by(AuditEvent.created_at.desc())
        ).all()

    def save_candidate_chapter(
        self,
        batch_id: str,
        ordinal: int,
        title: str,
        body: str,
        state_delta: dict[str, object],
    ) -> Chapter:
        batch = self.get(batch_id)
        if batch.status != "draft":
            self.session.rollback()
            raise ValueError("candidate chapters can only be saved to draft batches")
        if not 1 <= ordinal <= batch.planned_chapters:
            self.session.rollback()
            raise ValueError("chapter ordinal must be within the batch plan")
        existing = self.session.scalar(
            select(Chapter.id).where(Chapter.batch_id == batch.id, Chapter.ordinal == ordinal)
        )
        if existing is not None:
            self.session.rollback()
            raise ValueError("chapter ordinal already exists in batch")
        chapter = Chapter(
            id=str(uuid4()),
            batch_id=batch.id,
            project_id=batch.project_id,
            ordinal=ordinal,
            title=title,
            body=body,
            visible_char_count=count_visible_characters(body),
            status="candidate",
            state_delta=deepcopy(state_delta),
            official_chapter_number=None,
            revision=1,
        )
        self.session.add(chapter)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return chapter

    def replace_candidate_body(self, chapter_id: str, body: str) -> Chapter:
        self.session.expire_all()
        chapter = self.get_chapter(chapter_id)
        if chapter.status == "published":
            self.session.rollback()
            raise PermissionError("published chapters are frozen")
        if chapter.status != "candidate":
            self.session.rollback()
            raise PermissionError("only candidate chapters can be edited")
        visible_char_count = count_visible_characters(body)
        result = self.session.execute(
            update(Chapter)
            .where(
                Chapter.id == chapter.id,
                Chapter.status == "candidate",
                Chapter.revision == chapter.revision,
            )
            .values(
                body=body,
                visible_char_count=visible_char_count,
                revision=chapter.revision + 1,
            )
        )
        if result.rowcount != 1:
            self.session.rollback()
            raise PermissionError("candidate body replacement conflict")
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        self.session.expire_all()
        return self.get_chapter(chapter.id)

    def mark_ready(self, batch_id: str) -> WritingBatch:
        self.session.expire_all()
        batch = self.get(batch_id)
        if batch.status != "draft":
            self.session.rollback()
            raise ValueError("only draft batches can be marked ready")
        chapters = self.list_chapters(batch.id)
        self._validate_planned_chapters(batch, chapters)
        self._validate_visible_counts(chapters)
        claim = self.session.execute(
            update(WritingBatch)
            .where(
                WritingBatch.id == batch.id,
                WritingBatch.status == "draft",
                exists().where(
                    NovelProject.id == batch.project_id,
                    NovelProject.active_batch_id == batch.id,
                ),
            )
            .values(status="ready_for_review")
        )
        if claim.rowcount != 1:
            self.session.rollback()
            raise ValueError("batch ready conflict")
        self._add_audit(
            batch.project_id,
            "writing_batch",
            batch.id,
            "batch_ready",
            "author",
            {"chapter_ids": [chapter.id for chapter in chapters]},
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        self.session.expire_all()
        return self.get(batch.id)

    def reject(self, batch_id: str, reason: str) -> WritingBatch:
        normalized_reason = reason.strip()
        if not normalized_reason:
            self.session.rollback()
            raise ValueError("rejection reason is required")
        self.session.expire_all()
        batch = self.get(batch_id)
        if batch.status not in {"draft", "ready_for_review"}:
            self.session.rollback()
            raise ValueError("only draft or ready batches can be rejected")
        claim = self.session.execute(
            update(WritingBatch)
            .where(WritingBatch.id == batch.id, WritingBatch.status == batch.status)
            .values(status="rejected")
        )
        if claim.rowcount != 1:
            self.session.rollback()
            raise ValueError("batch rejection conflict")
        ownership = self.session.execute(
            update(NovelProject)
            .where(
                NovelProject.id == batch.project_id,
                NovelProject.active_batch_id == batch.id,
                NovelProject.next_batch_sequence == batch.sequence_number + 1,
            )
            .values(active_batch_id=None)
        )
        if ownership.rowcount != 1:
            self.session.rollback()
            raise ValueError("batch rejection conflict")
        self._add_audit(
            batch.project_id,
            "writing_batch",
            batch.id,
            "batch_rejected",
            "author",
            {"reason": normalized_reason},
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        self.session.expire_all()
        return self.get(batch.id)

    def approve(
        self, batch_id: str, approved_outline_version_id: str, actor: str = "author"
    ) -> WritingBatch:
        for attempt in range(CHAPTER_NUMBER_ALLOCATION_ATTEMPTS):
            self.session.expire_all()
            try:
                batch = self.get(batch_id)
                if batch.status != "ready_for_review":
                    raise ValueError("approval conflict")
                chapters = self.list_chapters(batch.id)
                self._validate_planned_chapters(batch, chapters)
                self._validate_visible_counts(chapters)
                if any(chapter.status != "candidate" for chapter in chapters):
                    raise ValueError("approval conflict")
                project = self.session.get(NovelProject, batch.project_id)
                if project is None:
                    raise ValueError("project not found")
                outline = self.session.get(OutlineVersion, approved_outline_version_id)
                if (
                    outline is None
                    or approved_outline_version_id != batch.base_outline_version_id
                    or outline.project_id != batch.project_id
                ):
                    raise ValueError("approval requires the batch's existing official outline")
                if outline.status != "official":
                    raise ValueError("approval conflict")
                if project.official_outline_version_id != outline.id:
                    raise ValueError("approval conflict")

                batch_claim = self.session.execute(
                    update(WritingBatch)
                    .where(
                        WritingBatch.id == batch.id,
                        WritingBatch.status == "ready_for_review",
                    )
                    .values(status="approved")
                )
                if batch_claim.rowcount != 1:
                    self.session.rollback()
                    raise ValueError("approval conflict")

                first_number = project.next_official_chapter_number
                reservation = self.session.execute(
                    update(NovelProject)
                    .where(
                        NovelProject.id == project.id,
                        NovelProject.active_batch_id == batch.id,
                        NovelProject.next_batch_sequence == batch.sequence_number + 1,
                        NovelProject.next_official_chapter_number == first_number,
                        NovelProject.official_outline_version_id == approved_outline_version_id,
                    )
                    .values(
                        active_batch_id=None,
                        next_official_chapter_number=first_number + len(chapters),
                    )
                )
                if reservation.rowcount != 1:
                    self.session.rollback()
                    if attempt == CHAPTER_NUMBER_ALLOCATION_ATTEMPTS - 1:
                        raise ValueError("approval conflict")
                    continue

                for number, chapter in enumerate(chapters, start=first_number):
                    promotion = self.session.execute(
                        update(Chapter)
                        .where(
                            Chapter.id == chapter.id,
                            Chapter.status == "candidate",
                            Chapter.revision == chapter.revision,
                            Chapter.body == chapter.body,
                            Chapter.visible_char_count == chapter.visible_char_count,
                        )
                        .values(
                            status="official",
                            official_chapter_number=number,
                            revision=chapter.revision + 1,
                        )
                    )
                    if promotion.rowcount != 1:
                        self.session.rollback()
                        if attempt == CHAPTER_NUMBER_ALLOCATION_ATTEMPTS - 1:
                            raise ValueError("approval conflict")
                        break
                else:
                    from ainovel.services.stages import StageService

                    StageService(self.session).commit_batch_progress(batch, chapters, first_number, actor)
                    self._add_audit(
                        batch.project_id,
                        "writing_batch",
                        batch.id,
                        "batch_approved",
                        actor,
                        {
                            "chapter_ids": [chapter.id for chapter in chapters],
                            "state_deltas": [deepcopy(chapter.state_delta) for chapter in chapters],
                        },
                    )
                    self.session.commit()
                    self.session.expire_all()
                    return self.get(batch.id)
            except Exception:
                self.session.rollback()
                raise
        raise ValueError("approval conflict")

    def publish_chapter(self, chapter_id: str) -> Chapter:
        self.session.expire_all()
        chapter = self.get_chapter(chapter_id)
        if chapter.status != "official":
            self.session.rollback()
            if chapter.status == "published":
                raise ValueError("publication conflict")
            raise ValueError("only official chapters can be published")
        claim = self.session.execute(
            update(Chapter)
            .where(
                Chapter.id == chapter.id,
                Chapter.status == "official",
                Chapter.revision == chapter.revision,
            )
            .values(status="published", revision=chapter.revision + 1)
        )
        if claim.rowcount != 1:
            self.session.rollback()
            raise ValueError("publication conflict")
        self._add_audit(
            chapter.project_id,
            "chapter",
            chapter.id,
            "chapter_published",
            "author",
            {"official_chapter_number": chapter.official_chapter_number},
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        self.session.expire_all()
        return self.get_chapter(chapter.id)

    @staticmethod
    def _validate_planned_chapters(batch: WritingBatch, chapters: list[Chapter]) -> None:
        expected_ordinals = set(range(1, batch.planned_chapters + 1))
        if len(chapters) != batch.planned_chapters or {chapter.ordinal for chapter in chapters} != expected_ordinals:
            raise ValueError("all planned chapter ordinals are required before review")

    @staticmethod
    def _validate_visible_counts(chapters: list[Chapter]) -> None:
        if any(
            not MIN_VISIBLE_CHARACTERS <= chapter.visible_char_count <= MAX_VISIBLE_CHARACTERS
            for chapter in chapters
        ):
            raise ValueError("chapter visible character counts must be between 4500 and 6000")

    def _add_audit(
        self,
        project_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str,
        details: dict[str, object],
    ) -> None:
        self.session.add(
            AuditEvent(
                id=str(uuid4()),
                project_id=project_id,
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                actor=actor,
                details=deepcopy(details),
            )
        )

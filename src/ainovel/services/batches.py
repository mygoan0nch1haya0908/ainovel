from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ainovel.models.audit import AuditEvent
from ainovel.models.batch import Chapter, WritingBatch
from ainovel.models.outline import OutlineVersion
from ainovel.models.project import NovelProject
from ainovel.services.counting import count_visible_characters

MIN_BATCH_CHAPTERS = 1
MAX_BATCH_CHAPTERS = 5
MIN_VISIBLE_CHARACTERS = 4500
MAX_VISIBLE_CHARACTERS = 6000
CHAPTER_NUMBER_ALLOCATION_ATTEMPTS = 3


class BatchService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self, project_id: str, outline_version_id: str, planned_chapters: int
    ) -> WritingBatch:
        if not MIN_BATCH_CHAPTERS <= planned_chapters <= MAX_BATCH_CHAPTERS:
            raise ValueError("batch size must be between 1 and 5")
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
        batch = WritingBatch(
            id=str(uuid4()),
            project_id=project.id,
            base_outline_version_id=outline.id,
            planned_chapters=planned_chapters,
            status="draft",
        )
        self.session.add(batch)
        self._add_audit(
            project.id,
            "writing_batch",
            batch.id,
            "batch_created",
            "author",
            {"base_outline_version_id": outline.id, "planned_chapters": planned_chapters},
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
        )
        self.session.add(chapter)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return chapter

    def replace_candidate_body(self, chapter_id: str, body: str) -> Chapter:
        chapter = self.get_chapter(chapter_id)
        if chapter.status == "published":
            self.session.rollback()
            raise PermissionError("published chapters are frozen")
        if chapter.status != "candidate":
            self.session.rollback()
            raise PermissionError("only candidate chapters can be edited")
        chapter.body = body
        chapter.visible_char_count = count_visible_characters(body)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return chapter

    def mark_ready(self, batch_id: str) -> WritingBatch:
        batch = self.get(batch_id)
        if batch.status != "draft":
            self.session.rollback()
            raise ValueError("only draft batches can be marked ready")
        chapters = self.list_chapters(batch.id)
        self._validate_planned_chapters(batch, chapters)
        self._validate_visible_counts(chapters)
        batch.status = "ready_for_review"
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
        return batch

    def reject(self, batch_id: str, reason: str) -> WritingBatch:
        batch = self.get(batch_id)
        if batch.status not in {"draft", "ready_for_review"}:
            self.session.rollback()
            raise ValueError("only draft or ready batches can be rejected")
        batch.status = "rejected"
        self._add_audit(
            batch.project_id,
            "writing_batch",
            batch.id,
            "batch_rejected",
            "author",
            {"reason": reason},
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return batch

    def approve(
        self, batch_id: str, approved_outline_version_id: str, actor: str = "author"
    ) -> WritingBatch:
        for attempt in range(CHAPTER_NUMBER_ALLOCATION_ATTEMPTS):
            try:
                batch = self.get(batch_id)
                if batch.status != "ready_for_review":
                    raise ValueError("only ready batches can be approved")
                chapters = self.list_chapters(batch.id)
                self._validate_planned_chapters(batch, chapters)
                self._validate_visible_counts(chapters)
                if any(chapter.status != "candidate" for chapter in chapters):
                    raise ValueError("only candidate chapters can be approved")
                project = self.session.get(NovelProject, batch.project_id)
                if project is None:
                    raise ValueError("project not found")
                outline = self.session.get(OutlineVersion, approved_outline_version_id)
                if (
                    outline is None
                    or approved_outline_version_id != batch.base_outline_version_id
                    or outline.project_id != batch.project_id
                    or outline.status != "official"
                    or project.official_outline_version_id != outline.id
                ):
                    raise ValueError("approval requires the batch's existing official outline")

                self.session.refresh(project)
                first_number = project.next_official_chapter_number
                reservation = self.session.execute(
                    update(NovelProject)
                    .where(
                        NovelProject.id == project.id,
                        NovelProject.next_official_chapter_number == first_number,
                    )
                    .values(
                        next_official_chapter_number=first_number + len(chapters),
                        official_outline_version_id=outline.id,
                    )
                )
                if reservation.rowcount != 1:
                    self.session.rollback()
                    if attempt == CHAPTER_NUMBER_ALLOCATION_ATTEMPTS - 1:
                        raise RuntimeError("official chapter number allocation exhausted")
                    continue

                for number, chapter in enumerate(chapters, start=first_number):
                    chapter.status = "official"
                    chapter.official_chapter_number = number
                batch.status = "approved"
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
                return batch
            except Exception:
                self.session.rollback()
                raise
        raise RuntimeError("official chapter number allocation exhausted")

    def publish_chapter(self, chapter_id: str) -> Chapter:
        chapter = self.get_chapter(chapter_id)
        if chapter.status != "official":
            self.session.rollback()
            raise ValueError("only official chapters can be published")
        chapter.status = "published"
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
        return chapter

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
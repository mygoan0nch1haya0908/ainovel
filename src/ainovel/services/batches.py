from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ainovel.models.batch import Chapter, WritingBatch
from ainovel.models.outline import OutlineVersion
from ainovel.models.project import NovelProject
from ainovel.services.counting import count_visible_characters

MIN_BATCH_CHAPTERS = 1
MAX_BATCH_CHAPTERS = 5
MIN_VISIBLE_CHARACTERS = 4500
MAX_VISIBLE_CHARACTERS = 6000


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

    def mark_ready(self, batch_id: str) -> WritingBatch:
        batch = self.get(batch_id)
        if batch.status != "draft":
            self.session.rollback()
            raise ValueError("only draft batches can be marked ready")
        chapters = self.list_chapters(batch.id)
        expected_ordinals = set(range(1, batch.planned_chapters + 1))
        if {chapter.ordinal for chapter in chapters} != expected_ordinals:
            self.session.rollback()
            raise ValueError("all planned chapter ordinals are required before review")
        if any(
            not MIN_VISIBLE_CHARACTERS
            <= chapter.visible_char_count
            <= MAX_VISIBLE_CHARACTERS
            for chapter in chapters
        ):
            self.session.rollback()
            raise ValueError("chapter visible character counts must be between 4500 and 6000")
        batch.status = "ready_for_review"
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return batch

    def reject(self, batch_id: str, reason: str) -> WritingBatch:
        del reason
        batch = self.get(batch_id)
        batch.status = "rejected"
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return batch

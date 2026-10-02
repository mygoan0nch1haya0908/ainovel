import pytest
from sqlalchemy import select
from ainovel.models import Chapter, StoryMemoryEntry
from ainovel.services.batches import BatchService
from ainovel.services.story_memory import StoryMemoryService


def test_memory_failure_rolls_back_chapter_approval(session, project, official_outline, monkeypatch):
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    chapter = service.save_candidate_chapter(batch.id, 1, '正文', '甲' * 4500, {})
    service.mark_ready(batch.id)
    def fail(self, batch_id, actor):
        raise ValueError('memory promotion failed')
    monkeypatch.setattr(StoryMemoryService, 'promote_batch', fail, raising=False)
    with pytest.raises(ValueError, match='memory promotion'):
        service.approve(batch.id, official_outline.id)
    session.expire_all()
    assert service.get(batch.id).status == 'ready_for_review'
    assert session.get(Chapter, chapter.id).status == 'candidate'
    assert project.next_official_chapter_number == 1
    assert session.scalar(select(StoryMemoryEntry)) is None


def test_candidate_cannot_be_promoted_directly(session, project, official_outline):
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    with pytest.raises(ValueError, match='approved'):
        StoryMemoryService(session).promote_batch(batch.id, 'author')

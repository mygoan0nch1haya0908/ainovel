"""Author-controlled memory. All writes flush; the caller owns the transaction."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ainovel.agents.memory_contracts import MemoryEntryInput, SourceRef
from ainovel.models import (AuditEvent, Chapter, ConstitutionVersion, ContextSource,
                            MemoryCardVersion, NovelProject, OutlineNode, OutlineVersion,
                            StageRoadmapVersion, StoryStage)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return sha256(canonical(value).encode('utf-8')).hexdigest()


def _leaves(value, path=''):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], path + '/' + str(key).replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, path + '/' + str(index))
    elif value is not None:
        text = value if isinstance(value, str) else canonical(value)
        if text.strip():
            yield path, text


class StoryMemoryService:
    def __init__(self, session: Session):
        self.session = session

    def _documents(self, project_id):
        project = self.session.get(NovelProject, project_id)
        if project is None:
            raise ValueError('project not found')
        docs = []
        if project.current_constitution_version_id:
            c = self.session.get(ConstitutionVersion, project.current_constitution_version_id)
            if c is None or c.project_id != project_id or not c.author_approved:
                raise ValueError('invalid constitution')
            docs.append(('constitution', c.id, str(c.version_number), c.content, True))
        if project.official_outline_version_id:
            outline = self.session.get(OutlineVersion, project.official_outline_version_id)
            if outline is None or outline.project_id != project_id or outline.status != 'official':
                raise ValueError('invalid official outline')
            for n in self.session.scalars(select(OutlineNode).where(OutlineNode.outline_version_id == outline.id).order_by(OutlineNode.order, OutlineNode.stable_key)):
                docs.append(('outline_node', n.id, str(outline.version_number),
                             {'title': n.title, 'kind': n.kind, 'author_locked': n.author_locked, 'payload': n.payload}, True))
        for stage in self.session.scalars(select(StoryStage).where(StoryStage.project_id == project_id).order_by(StoryStage.id)):
            if not stage.approved_roadmap_id:
                continue
            roadmap = self.session.get(StageRoadmapVersion, stage.approved_roadmap_id)
            if roadmap is None or roadmap.stage_id != stage.id or roadmap.status != 'APPROVED':
                raise ValueError('invalid approved roadmap')
            # Pin the immutable approved architecture, not an editable stage draft.
            docs.append(('stage_architecture', roadmap.id, str(roadmap.version_number), {'architecture': roadmap.architecture}, True))
            docs.append(('stage_roadmap', roadmap.id, str(roadmap.version_number), roadmap.payload, False))
        return docs

    def sources(self, project_id):
        """Read-only canonical paragraphs; offsets count Python Unicode characters."""
        result = []
        for kind, ident, version, data, needs_classification in self._documents(project_id):
            for path, text in _leaves(data):
                for match in re.finditer(r'\S.*?(?=\n\s*\n|\Z)', text, re.S):
                    ref = SourceRef(project_id=project_id, source_type=kind, source_id=ident,
                                    source_version=version, content_hash=sha256(text.encode()).hexdigest(),
                                    field_path=path, excerpt_start=match.start(), excerpt_end=match.end())
                    structural = kind == 'outline_node' and path in ('/title', '/kind', '/author_locked')
                    result.append({'ref': ref.model_dump(), 'text': match.group(), 'needs_classification': needs_classification and not structural})
        return result

    def fingerprint(self, project_id):
        return digest(self.sources(project_id))

    def _resolve(self, ref):
        if ref.source_type == 'constitution':
            row = self.session.get(ConstitutionVersion, ref.source_id)
            if row is None or row.project_id != ref.project_id or not row.author_approved:
                raise ValueError('invalid source project or approval')
            version, data = str(row.version_number), row.content
        elif ref.source_type == 'outline_node':
            row = self.session.get(OutlineNode, ref.source_id)
            outline = self.session.get(OutlineVersion, row.outline_version_id) if row else None
            if outline is None or outline.project_id != ref.project_id or outline.status not in ('official', 'superseded'):
                raise ValueError('invalid source project or approval')
            version, data = str(outline.version_number), {'title': row.title, 'kind': row.kind, 'author_locked': row.author_locked, 'payload': row.payload}
        elif ref.source_type in ('stage_architecture', 'stage_roadmap'):
            row = self.session.get(StageRoadmapVersion, ref.source_id)
            stage = self.session.get(StoryStage, row.stage_id) if row else None
            if stage is None or stage.project_id != ref.project_id or row.status != 'APPROVED':
                raise ValueError('invalid source project or approval')
            version = str(row.version_number)
            data = {'architecture': row.architecture} if ref.source_type == 'stage_architecture' else row.payload
        elif ref.source_type == 'official_chapter':
            row = self.session.get(Chapter, ref.source_id)
            if row is None or row.project_id != ref.project_id or row.status not in ('official', 'published'):
                raise ValueError('invalid source project or approval')
            version, data = str(row.revision), {'body': row.body, 'state_delta': row.state_delta}
        else:
            raise ValueError('unsupported memory source')
        leaves = dict(_leaves(data))
        text = leaves.get(ref.field_path)
        publication_only = (ref.source_type == 'official_chapter' and row.status == 'published'
                            and ref.source_version.isdigit() and row.revision == int(ref.source_version) + 1)
        if (version != ref.source_version and not publication_only) or text is None or sha256(text.encode()).hexdigest() != ref.content_hash:
            raise ValueError('source version/hash/path mismatch')
        if ref.excerpt_end is not None and ref.excerpt_end > len(text):
            raise ValueError('source excerpt exceeds original')
        return text

    def validate_refs(self, refs: list[SourceRef]):
        for ref in refs:
            self._resolve(ref)

    def pending_confirmed(self, pending, card):
        """Derive acknowledgement from the selected valid approved card, never delete history."""
        if (card is None or card.status != 'APPROVED' or card.project_id != pending.project_id
                or self.card_blockers(card)):
            return False
        refs = [SourceRef.model_validate(r) for entry in card.entries for r in entry['source_refs']]
        if not pending.source_refs:
            return False
        for raw in pending.source_refs:
            target = SourceRef.model_validate(raw)
            text = self._resolve(target)
            start, end = target.excerpt_start or 0, target.excerpt_end if target.excerpt_end is not None else len(text)
            spans = []
            for ref in refs:
                if all(getattr(ref, k) == getattr(target, k) for k in
                       ('project_id', 'source_type', 'source_id', 'source_version', 'content_hash', 'field_path')):
                    spans.append((ref.excerpt_start or 0, ref.excerpt_end if ref.excerpt_end is not None else len(text)))
            for left, right in sorted(spans):
                if left <= start:
                    start = max(start, right)
            if start < end:
                return False
        return True

    def ensure_index(self, project_id):
        """Append-only raw index, distinct from legacy's rebuilt official index."""
        sources = self.sources(project_id)
        for source in sources:
            ref = source['ref']
            source_id = digest(ref)
            existing = self.session.scalar(select(ContextSource).where(
                ContextSource.project_id == project_id, ContextSource.source_type == 'story_memory_raw',
                ContextSource.source_id == source_id, ContextSource.state_scope == 'memory:source'))
            if existing is None:
                self.session.add(ContextSource(id=str(uuid4()), project_id=project_id, source_type='story_memory_raw',
                    source_id=source_id, source_version=ref['source_version'], state_scope='memory:source',
                    layer=2, text=source['text'], content_hash=sha256(source['text'].encode()).hexdigest(),
                    canonical_source_type=ref['source_type'], canonical_source_id=ref['source_id'],
                    excerpt_start=ref['excerpt_start'], excerpt_end=ref['excerpt_end']))
        self.session.flush()
        return digest(sources)

    def create_card(self, project_id: str, entries: list[MemoryEntryInput], actor: str):
        entries = [MemoryEntryInput.model_validate(e).model_copy(deep=True) for e in entries]
        for entry in entries:
            entry.point_ids = self.qualify_points(project_id, entry.point_ids)
            if any(ref.project_id != project_id for ref in entry.source_refs):
                raise ValueError('cross-project memory source')
            self.validate_refs(entry.source_refs)
        previous = self.session.scalar(select(MemoryCardVersion).where(MemoryCardVersion.project_id == project_id).order_by(MemoryCardVersion.version_number.desc()))
        card = MemoryCardVersion(id=str(uuid4()), project_id=project_id,
            version_number=previous.version_number + 1 if previous else 1, parent_id=previous.id if previous else None,
            entries=[e.model_dump() for e in entries], source_fingerprint=self.fingerprint(project_id))
        self.session.add(card)
        self.session.flush()
        self._audit(card, 'memory_card_created', actor)
        return card

    def qualify_points(self, project_id, point_ids):
        available = {f'{ident}:{p["point_id"]}' for kind, ident, _, data, _ in self._documents(project_id)
                     if kind == 'stage_roadmap' for p in data.get('points', [])}
        result = []
        for point in point_ids:
            matches = [p for p in available if p == point or (':' not in point and p.rsplit(':', 1)[1] == point)]
            if len(matches) != 1:
                raise ValueError(f'ambiguous or unknown plot association / 剧情点关联有歧义或不存在: {point}')
            if matches[0] not in result:
                result.append(matches[0])
        return result

    def card_blockers(self, card):
        if card.source_fingerprint != self.fingerprint(card.project_id):
            return ['stale memory card / 来源已更新，请重新确认约束卡']
        entries = [MemoryEntryInput.model_validate(e) for e in card.entries]
        for entry in entries:
            self.qualify_points(card.project_id, entry.point_ids)
        refs = [r for e in entries for r in e.source_refs]
        if any(r.project_id != card.project_id for r in refs):
            return ['cross-project memory source']
        self.validate_refs(refs)
        missing = []
        for source in self.sources(card.project_id):
            if not source['needs_classification']:
                continue
            target = source['ref']
            spans = []
            for ref in refs:
                value = ref.model_dump()
                if all(value[k] == target[k] for k in ('project_id', 'source_type', 'source_id', 'source_version', 'content_hash', 'field_path')):
                    spans.append((ref.excerpt_start or 0, ref.excerpt_end if ref.excerpt_end is not None else len(self._resolve(ref))))
            covered = target['excerpt_start']
            for start, end in sorted(spans):
                if start <= covered:
                    covered = max(covered, end)
            if covered < target['excerpt_end']:
                missing.append(f"uncovered source / 未确认分类: {target['source_type']}:{target['source_id']}{target['field_path']}:{target['excerpt_start']}-{target['excerpt_end']}")
        if not entries:
            missing.append('uncovered rules / 约束卡不能为空')
        for entry in entries:
            if entry.kind != 'rule' and not entry.point_ids and not entry.entity_ids and entry.effective_until is None:
                missing.append('unclassified local memory / 局部记忆需要剧情点、人物或有限章节范围')
        return missing

    def approve_card(self, card_id: str, expected_fingerprint: str, actor: str):
        card = self.session.get(MemoryCardVersion, card_id)
        if card is None:
            raise ValueError('memory card not found')
        if expected_fingerprint != card.source_fingerprint:
            raise ValueError('stale memory card preview')
        blockers = self.card_blockers(card)
        if blockers:
            raise ValueError('; '.join(blockers))
        if card.status == 'APPROVED':
            return card
        card.status, card.approved_by, card.approved_at = 'APPROVED', actor, datetime.now(timezone.utc)
        self._audit(card, 'memory_card_approved', actor)
        self.session.flush()
        return card

    def _audit(self, card, action, actor):
        self.session.add(AuditEvent(id=str(uuid4()), project_id=card.project_id, entity_type='memory_card',
            entity_id=card.id, action=action, actor=actor, details={'source_fingerprint': card.source_fingerprint}))
        self.session.flush()

    def promote_batch(self, batch_id, actor):
        from ainovel.models import WritingBatch, StoryMemoryEntry, StageWorkflow, StageWorkflowNode
        from ainovel.services.scoped_context import active_policy
        from ainovel.services.context import ContextIndexService
        batch = self.session.get(WritingBatch, batch_id)
        if batch is None or batch.status != 'approved':
            raise ValueError('memory promotion requires approved batch')
        policy = active_policy(self.session, batch.source_workflow_id) if batch.source_workflow_id else None
        if policy is None or policy.strategy != 'scoped_story_v1':
            return
        mapping = self.session.get(StageWorkflow, batch.source_workflow_id)
        if mapping is None or mapping.committed_batch_id != batch_id:
            raise ValueError('approved memory missing stage progress')
        records = ContextIndexService(self.session)._official_memory_records(batch.project_id)
        chapters = self.session.scalars(select(Chapter).where(Chapter.batch_id == batch_id).order_by(Chapter.ordinal)).all()
        for chapter in chapters:
            if chapter.status not in ('official', 'published') or not chapter.official_chapter_number:
                raise ValueError('memory promotion requires approved chapters')
            node = self.session.get(StageWorkflowNode, (batch.source_workflow_id, chapter.ordinal))
            point_id = node.node_id.rsplit(':', 1)[0]
            ref = SourceRef(project_id=batch.project_id, source_type='official_chapter', source_id=chapter.id,
                source_version=str(chapter.revision), content_hash=sha256(chapter.body.encode()).hexdigest(), field_path='/body')
            # Only summaries which still match the approved body and state delta qualify.
            for record in records:
                if record['source_type'] != 'chapter_summary' or record.get('canonical_source_id') != chapter.id:
                    continue
                key = f'summary:{chapter.id}:{chapter.revision}'
                if self.session.scalar(select(StoryMemoryEntry.id).where(StoryMemoryEntry.project_id == batch.project_id, StoryMemoryEntry.source_key == key, StoryMemoryEntry.state_scope == 'official')):
                    continue
                self.session.add(StoryMemoryEntry(id=str(uuid4()), project_id=batch.project_id, source_key=key,
                    kind='summary', text=record['text'], source_refs=[ref.model_dump()], entity_ids=[],
                    point_ids=[f'{mapping.roadmap_id}:{point_id}'], effective_from=chapter.official_chapter_number + 1,
                    audience='narratable', state_scope='official'))
            # Free state_delta objects have no trustworthy entity/range mapping. Keep
            # them visible for author classification, never promote guessed facts.
            if chapter.state_delta:
                key = f'unclassified:{chapter.id}:{chapter.revision}'
                if not self.session.scalar(select(StoryMemoryEntry.id).where(StoryMemoryEntry.project_id == batch.project_id, StoryMemoryEntry.source_key == key, StoryMemoryEntry.state_scope == 'pending:' + batch_id)):
                    self.session.add(StoryMemoryEntry(id=str(uuid4()), project_id=batch.project_id, source_key=key,
                        kind='fact', text=canonical(chapter.state_delta), source_refs=[ref.model_dump()], entity_ids=[],
                        point_ids=[f'{mapping.roadmap_id}:{point_id}'], effective_from=chapter.official_chapter_number + 1,
                        audience='author_only', state_scope='pending:' + batch_id))
        self.session.flush()

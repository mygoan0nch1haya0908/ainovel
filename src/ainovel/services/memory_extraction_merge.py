"""Untrusted model suggestions become editable drafts, never approved memory."""
from copy import deepcopy
from uuid import uuid5, NAMESPACE_URL
from sqlalchemy import update
from ainovel.agents.memory_contracts import MemoryEntryInput
from ainovel.agents.memory_extraction_contracts import ExtractionResult
from ainovel.models import MemoryCardVersion, NovelProject
from ainovel.services.story_memory import StoryMemoryService, digest
from ainovel.services.memory_extraction import MemoryExtractionService


def validate_chunk_result(chunk: dict, result: dict) -> dict:
    parsed = ExtractionResult.model_validate(result)
    ref = chunk['source']['ref']
    start, end = ref['excerpt_start'], ref['excerpt_end']
    existing = {e['entry_id']: e for e in chunk['request']['input_payload']['existing_entries']}
    entries, conflicts, spans = [], [], []
    for index, item in enumerate(parsed.entries):
        refs = []
        for citation in item.references:
            if citation.source_id != chunk['source_id'] or not start <= citation.start < citation.end <= end:
                raise ValueError('模型引用超出本块来源范围')
            refs.append({**ref, 'excerpt_start': citation.start, 'excerpt_end': citation.end})
            spans.append((citation.start, citation.end))
        ident = item.replaces_entry_id
        if ident and ident not in existing:
            raise ValueError('模型修改了未提供的记忆身份')
        if ident and existing[ident]['author_locked']:
            conflicts.append(f'作者锁定条目保留，AI修改未采用：{ident}')
            continue
        entry = item.model_dump(exclude={'references', 'replaces_entry_id'})
        entry.update(source_refs=refs, entry_id=ident or str(uuid5(NAMESPACE_URL,
            f"ainovel:extraction:{chunk['cache_key']}:{index}")), origin='ai', author_locked=False)
        value = MemoryEntryInput.model_validate(entry).model_dump()
        allowed = {p['point_id'] for p in chunk['request']['input_payload']['plot_points']}
        if any(p not in allowed for p in value['point_ids']):
            raise ValueError('未知剧情点关联')
        if value['kind'] != 'rule' and not value['point_ids'] and not value['entity_ids'] and value['effective_until'] is None:
            raise ValueError('局部记忆缺少适用范围')
        entries.append(value)
    covered = start
    for left, right in sorted(spans):
        if left <= covered:
            covered = max(covered, right)
    unresolved = list(parsed.unresolved)
    if covered < end:
        unresolved.append('本块原文仍有未覆盖范围')
    return dict(entries=entries, unresolved=unresolved, conflicts=conflicts)


class MemoryExtractionMergeService:
    def __init__(self, session):
        self.session = session
        self.tasks = MemoryExtractionService(session)

    def preview(self, job_id):
        job = self.tasks.get(job_id)
        entries = {e['entry_id']: deepcopy(e) for e in job.snapshot['base']}
        proposed = {}
        blockers, conflicts = [], []
        if not self.tasks.current(job):
            blockers.append('来源或基础卡已更新，请重新准备')
        for chunk in self.tasks.chunks(job_id):
            if chunk.status not in ('SUCCEEDED', 'REUSED'):
                blockers.append(f'第{chunk.ordinal + 1}块尚未成功')
                continue
            result = validate_chunk_result(chunk.snapshot, chunk.result)
            blockers.extend(result['unresolved'])
            conflicts.extend(result['conflicts'])
            for entry in result['entries']:
                previous = entries.get(entry['entry_id'])
                if previous and previous.get('author_locked'):
                    conflicts.append('锁定条目已保留')
                elif entry['entry_id'] in proposed and proposed[entry['entry_id']] != entry:
                    blockers.append('多个来源对同一记忆提出不同更新')
                else:
                    proposed[entry['entry_id']] = entry
                    entries[entry['entry_id']] = entry
        base = {e['entry_id']: e for e in job.snapshot['base']}
        keys = ('source_type', 'source_id', 'source_version', 'content_hash', 'field_path')
        active = {tuple(s['ref'][k] for k in keys) for s in StoryMemoryService(self.session).sources(job.project_id)}
        managed = {'constitution', 'outline_node', 'stage_architecture', 'stage_roadmap'}
        obsolete_ids = [ident for ident, entry in entries.items() if any(
            ref['source_type'] in managed and tuple(ref[k] for k in keys) not in active
            for ref in entry['source_refs'])]
        changes = [dict(kind='修改' if k in base else '新增', entry_id=k, before=base.get(k), after=e)
                   for k, e in entries.items() if base.get(k) != e]
        value = dict(entries=list(entries.values()), blockers=blockers, conflicts=conflicts, changes=changes,
                     obsolete_ids=obsolete_ids)
        value['fingerprint'] = digest(dict(job=job.id, revision=job.revision, result=value))
        return value

    def merge(self, job_id, expected_revision, expected_fingerprint, edited_entries, actor, *, retained_obsolete_ids=()):
        job = self.tasks.lock(job_id)
        self.session.execute(update(NovelProject).where(NovelProject.id == job.project_id).values(title=NovelProject.title))
        self.session.expire_all()
        job = self.tasks.get(job_id)
        if job.status == 'MERGED':
            return self.session.get(MemoryCardVersion, job.merged_card_id)
        if job.status != 'NEEDS_REVIEW' or job.revision != expected_revision:
            raise ValueError('任务尚未完成或版本已变化')
        preview = self.preview(job_id)
        if preview['blockers'] or preview['fingerprint'] != expected_fingerprint:
            raise ValueError('候选存在未确认来源或差异预览已过期')
        values = [MemoryEntryInput.model_validate(e) for e in edited_entries]
        ids = [e.entry_id for e in values]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError('记忆身份缺失或重复')
        retained = set(retained_obsolete_ids)
        if not retained <= set(preview['obsolete_ids']):
            raise ValueError('保留确认不属于当前过期来源条目')
        if (set(ids) & set(preview['obsolete_ids'])) - retained:
            raise ValueError('旧来源已替换：请逐项明确保留，或从候选中删除')
        for previous in job.snapshot['base']:
            if previous['author_locked'] and not any(e.model_dump() == previous for e in values):
                raise ValueError('不能覆盖或删除锁定条目；请通过人工编辑另建版本')
        memory = StoryMemoryService(self.session)
        card = memory.create_card(job.project_id, values, actor)
        blockers = memory.card_blockers(card)
        if blockers:
            self.session.rollback()
            raise ValueError('人工修订仍存在未覆盖来源或关联问题')
        job.status, job.merged_card_id, job.revision = 'MERGED', card.id, job.revision + 1
        self.tasks.audit(job, 'memory_extraction_merged_as_draft', actor)
        self.session.commit()
        return card

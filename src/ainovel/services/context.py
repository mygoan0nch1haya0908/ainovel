from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Mapping, Protocol
from uuid import uuid4

from sqlalchemy import bindparam, select, text, update
from sqlalchemy.orm import Session

from ainovel.context import (
    ITEM_FRAMING_TOKENS,
    ContextBudgeter,
    ContextCandidate,
)
from ainovel.models.batch import Chapter
from ainovel.models.context import ContextPacket, ContextPacketItem, ContextSource
from ainovel.models.outline import OutlineNode, OutlineVersion
from ainovel.models.project import ConstitutionVersion, NovelProject
from ainovel.models.workflow import GenerationWorkflow, WorkflowArtifact, WorkflowStep


WORKFLOW_SCOPE_PREFIX = "workflow:"
EVENT_CHAIN_LIMIT = 30
RECENT_SUMMARY_LIMIT = 5

SOURCE_LAYERS: dict[str, int] = {
    "constitution": 0,
    "core_prohibition": 0,
    "provisional_ending": 0,
    "author_locked_fact": 0,
    "world_rule": 1,
    "power_system": 1,
    "core_character": 1,
    "current_volume": 2,
    "volume": 2,
    "plot_stage": 2,
    "current_stage_goal": 2,
    "outline_node": 2,
    "event_chain": 3,
    "chapter_summary": 4,
    "scene_summary": 4,
    "character": 5,
    "location": 5,
    "item": 5,
    "relationship": 5,
    "foreshadowing": 5,
    "critical_character_state": 5,
    "approved_batch_plan": 6,
    "batch_plan": 6,
    "chapter_summary_delta": 6,
    "summary_delta": 6,
    "state_delta": 6,
    "historical_excerpt": 7,
    "official_chapter": 7,
}

REQUIRED_SOURCE_TYPES = {
    "constitution",
    "core_prohibition",
    "provisional_ending",
    "author_locked_fact",
    "current_stage_goal",
    "approved_batch_plan",
    "critical_character_state",
}

AUTO_OFFICIAL_SOURCE_TYPES = set(SOURCE_LAYERS) - {
    "batch_plan",
    "chapter_summary_delta",
    "summary_delta",
    "state_delta",
    "official_chapter",
}
WORKFLOW_SOURCE_TYPES = {
    "event_chain",
    "chapter_summary",
    "scene_summary",
    "approved_batch_plan",
    "chapter_summary_delta",
    "summary_delta",
    "state_delta",
    "historical_excerpt",
}

ARTIFACT_SOURCE_TYPES = {
    "event_chain": "event_chain",
    "chapter_summary": "chapter_summary",
    "scene_summary": "scene_summary",
    "chapter_summary_delta": "chapter_summary_delta",
    "summary_delta": "summary_delta",
    "state_delta": "state_delta",
    "requested_excerpt": "historical_excerpt",
    "historical_excerpt": "historical_excerpt",
}

FTS_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
MAX_FTS_QUERY_CHARACTERS = 4_096
MAX_FTS_TERMS = 32
MAX_FTS_TERM_CHARACTERS = 128


class SemanticRetriever(Protocol):
    """Future semantic retrieval seam; Stage 2 intentionally has no implementation."""

    def retrieve(
        self, project_id: str, query: str, source_types: set[str], limit: int
    ) -> list[ContextSource]: ...


@dataclass(frozen=True)
class ContextLimits:
    input_capacity_tokens: int
    reserved_output_tokens: int
    fixed_overhead_tokens: int = 0


class ContextIndexService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def rebuild_official(self, project_id: str) -> int:
        project = self.session.get(NovelProject, project_id)
        if project is None:
            raise ValueError("project not found")
        desired = self._official_records(project)
        desired_keys = {self._record_key(record) for record in desired}
        try:
            existing = self.session.scalars(
                select(ContextSource).where(
                    ContextSource.project_id == project.id,
                    ContextSource.state_scope == "official",
                )
            ).all()
            by_key = {self._source_key(row): row for row in existing}
            obsolete = [row for row in existing if self._source_key(row) not in desired_keys]
            if obsolete:
                obsolete_ids = [row.id for row in obsolete]
                self.session.execute(
                    update(ContextPacketItem)
                    .where(ContextPacketItem.source_id.in_(obsolete_ids))
                    .values(source_id=None)
                )
                for row in obsolete:
                    self._delete_fts(row.id)
                    self.session.delete(row)

            for record in desired:
                key = self._record_key(record)
                row = by_key.get(key)
                if row is None:
                    row = ContextSource(id=str(uuid4()), **record)
                    self.session.add(row)
                    self.session.flush()
                else:
                    row.layer = record["layer"]
                    row.text = record["text"]
                    row.content_hash = record["content_hash"]
                self._mirror(row)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return len(desired)

    def index_workflow_artifact(self, artifact_id: str) -> ContextSource:
        artifact = self.session.get(WorkflowArtifact, artifact_id)
        if artifact is None:
            raise ValueError("workflow artifact not found")
        workflow = self.session.get(GenerationWorkflow, artifact.workflow_id)
        step = self.session.get(WorkflowStep, artifact.step_id)
        if workflow is None or step is None or step.workflow_id != workflow.id:
            raise ValueError("workflow artifact ownership is invalid")
        source_type = ARTIFACT_SOURCE_TYPES.get(artifact.kind)
        if source_type is None:
            raise ValueError("workflow artifact kind is not indexable")
        body = self._artifact_text(artifact)
        scope = f"{WORKFLOW_SCOPE_PREFIX}{workflow.id}"
        source_version = int(artifact.content_hash[:15], 16)
        try:
            prior = self.session.scalars(
                select(ContextSource).where(
                    ContextSource.project_id == workflow.project_id,
                    ContextSource.source_type == source_type,
                    ContextSource.source_id == artifact.id,
                    ContextSource.state_scope == scope,
                )
            ).all()
            for row in prior:
                if (
                    row.source_version == source_version
                    and row.content_hash == artifact.content_hash
                    and row.text == body
                ):
                    self._mirror(row)
                    self.session.commit()
                    return row
                self._delete_fts(row.id)
                self.session.delete(row)
            row = ContextSource(
                id=str(uuid4()),
                project_id=workflow.project_id,
                source_type=source_type,
                source_id=artifact.id,
                source_version=source_version,
                state_scope=scope,
                layer=SOURCE_LAYERS[source_type],
                text=body,
                content_hash=artifact.content_hash,
            )
            self.session.add(row)
            self.session.flush()
            self._mirror(row)
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def search(
        self, project_id: str, query: str, source_types: set[str], limit: int
    ) -> list[ContextSource]:
        if limit <= 0:
            raise ValueError("search limit must be positive")
        if not source_types:
            return []
        match_query = self._safe_match_query(query)
        if match_query is None:
            return []
        statement = text(
            "SELECT cs.id "
            "FROM context_source_fts "
            "JOIN context_sources AS cs ON cs.id = context_source_fts.source_id "
            "WHERE context_source_fts MATCH :match_query "
            "AND context_source_fts.project_id = :project_id "
            "AND cs.project_id = :project_id "
            "AND cs.state_scope = 'official' "
            "AND cs.source_type IN :source_types "
            "ORDER BY bm25(context_source_fts), cs.source_type, cs.source_id, "
            "cs.source_version DESC, cs.id LIMIT :limit"
        ).bindparams(bindparam("source_types", expanding=True))
        ids = self.session.execute(
            statement,
            {
                "match_query": match_query,
                "project_id": project_id,
                "source_types": sorted(source_types),
                "limit": limit,
            },
        ).scalars().all()
        if not ids:
            return []
        rows = self.session.scalars(select(ContextSource).where(ContextSource.id.in_(ids))).all()
        row_map = {
            row.id: row
            for row in rows
            if row.project_id == project_id and row.state_scope == "official"
        }
        return [row_map[row_id] for row_id in ids if row_id in row_map]

    def _official_records(self, project: NovelProject) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        if project.current_constitution_version_id is not None:
            constitution = self.session.get(
                ConstitutionVersion, project.current_constitution_version_id
            )
            if (
                constitution is None
                or constitution.project_id != project.id
                or not constitution.author_approved
            ):
                raise ValueError("current constitution is not approved for this project")
            body = self._json_text(constitution.content)
            records.append(
                self._record(
                    project.id,
                    "constitution",
                    constitution.id,
                    constitution.version_number,
                    0,
                    body,
                )
            )
        if project.official_outline_version_id is not None:
            outline = self.session.get(OutlineVersion, project.official_outline_version_id)
            if outline is None or outline.project_id != project.id or outline.status != "official":
                raise ValueError("official outline is invalid for this project")
            nodes = self.session.scalars(
                select(OutlineNode)
                .where(OutlineNode.outline_version_id == outline.id)
                .order_by(OutlineNode.order, OutlineNode.stable_key)
            ).all()
            for node in nodes:
                source_type = self._outline_source_type(node)
                body = "\n".join(
                    part
                    for part in (node.title.strip(), self._json_text(node.payload))
                    if part and part != "{}"
                )
                records.append(
                    self._record(
                        project.id,
                        source_type,
                        f"{outline.id}:{node.stable_key}",
                        outline.version_number,
                        SOURCE_LAYERS[source_type],
                        body,
                    )
                )
        chapters = self.session.scalars(
            select(Chapter)
            .where(
                Chapter.project_id == project.id,
                Chapter.status.in_(("official", "published")),
                Chapter.official_chapter_number.is_not(None),
            )
            .order_by(Chapter.official_chapter_number)
        ).all()
        for chapter in chapters:
            records.append(
                self._record(
                    project.id,
                    "official_chapter",
                    chapter.id,
                    chapter.revision,
                    7,
                    f"{chapter.title}\n{chapter.body}",
                )
            )
        return records

    @staticmethod
    def _outline_source_type(node: OutlineNode) -> str:
        if node.author_locked:
            return "author_locked_fact"
        if node.kind in SOURCE_LAYERS and node.kind != "official_chapter":
            return node.kind
        return "outline_node"

    @staticmethod
    def _record(
        project_id: str,
        source_type: str,
        source_id: str,
        source_version: int,
        layer: int,
        body: str,
    ) -> dict[str, object]:
        return {
            "project_id": project_id,
            "source_type": source_type,
            "source_id": source_id,
            "source_version": source_version,
            "state_scope": "official",
            "layer": layer,
            "text": body,
            "content_hash": sha256(body.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _record_key(record: Mapping[str, object]) -> tuple[object, ...]:
        return (
            record["project_id"],
            record["source_type"],
            record["source_id"],
            record["source_version"],
            record["state_scope"],
        )

    @staticmethod
    def _source_key(row: ContextSource) -> tuple[object, ...]:
        return (
            row.project_id,
            row.source_type,
            row.source_id,
            row.source_version,
            row.state_scope,
        )

    @staticmethod
    def _json_text(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _artifact_text(cls, artifact: WorkflowArtifact) -> str:
        if artifact.text_content and artifact.text_content.strip():
            return artifact.text_content
        if artifact.kind in {"chapter_summary", "scene_summary", "chapter_summary_delta"}:
            summary = artifact.payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                return cls._json_text(artifact.payload)
        if artifact.kind in {"event_chain", "summary_delta", "state_delta"} and artifact.payload:
            return cls._json_text(artifact.payload)
        raise ValueError("indexable workflow artifact has no text")

    @staticmethod
    def _safe_match_query(query: str) -> str | None:
        if len(query) > MAX_FTS_QUERY_CHARACTERS:
            return None
        terms = FTS_WORD.findall(query)
        if (
            not terms
            or len(terms) > MAX_FTS_TERMS
            or any(len(term) > MAX_FTS_TERM_CHARACTERS for term in terms)
        ):
            return None
        return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

    def _delete_fts(self, source_id: str) -> None:
        self.session.execute(
            text("DELETE FROM context_source_fts WHERE source_id = :source_id"),
            {"source_id": source_id},
        )

    def _mirror(self, row: ContextSource) -> None:
        self._delete_fts(row.id)
        self.session.execute(
            text(
                "INSERT INTO context_source_fts(source_id, project_id, text) "
                "VALUES (:source_id, :project_id, :body)"
            ),
            {"source_id": row.id, "project_id": row.project_id, "body": row.text},
        )


class ContextBuilder:
    def __init__(self, session: Session) -> None:
        self.session = session

    def candidates_for_step(
        self, workflow_id: str, step_id: str
    ) -> list[ContextCandidate]:
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        if workflow is None:
            raise ValueError("workflow not found")
        step = self.session.get(WorkflowStep, step_id)
        if step is None:
            raise ValueError("workflow step not found")
        if step.workflow_id != workflow.id:
            raise ValueError("step does not belong to workflow")
        workflow_scope = f"{WORKFLOW_SCOPE_PREFIX}{workflow.id}"
        rows = self.session.scalars(
            select(ContextSource).where(
                ContextSource.project_id == workflow.project_id,
                ContextSource.state_scope.in_(("official", workflow_scope)),
            )
        ).all()
        rows = [
            row
            for row in rows
            if (
                row.state_scope == "official" and row.source_type in AUTO_OFFICIAL_SOURCE_TYPES
            )
            or (
                row.state_scope == workflow_scope and row.source_type in WORKFLOW_SOURCE_TYPES
            )
        ]
        artifact_rows = self.session.scalars(
            select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == workflow.id)
        ).all()
        artifacts = {artifact.id: artifact for artifact in artifact_rows}
        max_versions: dict[int, int] = {}
        for row in rows:
            layer = SOURCE_LAYERS[row.source_type]
            max_versions[layer] = max(max_versions.get(layer, row.source_version), row.source_version)
        candidates: dict[str, ContextCandidate] = {}
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                row.state_scope != workflow_scope,
                -row.source_version,
                row.source_type,
                row.source_id,
                row.id,
            ),
        )
        for row in sorted_rows:
            stable_key = f"{row.source_type}:{row.source_id}"
            if stable_key in candidates:
                continue
            layer = SOURCE_LAYERS[row.source_type]
            artifact = artifacts.get(row.source_id)
            temporal_distance = self._temporal_distance(
                row, artifact, step, max_versions[layer]
            )
            excerpt_start, excerpt_end = self._excerpt_offsets(artifact)
            required = row.source_type in REQUIRED_SOURCE_TYPES
            candidates[stable_key] = ContextCandidate(
                stable_key=stable_key,
                layer=layer,
                text=row.text,
                required=required,
                relevance=100 if required else 50,
                temporal_distance=temporal_distance,
                source_id=row.id,
                source_type=row.source_type,
                source_version=row.content_hash,
                state_scope=row.state_scope,
                excerpt_start=excerpt_start,
                excerpt_end=excerpt_end,
            )
        selected = list(candidates.values())
        selected = self._limit_layer(selected, 3, EVENT_CHAIN_LIMIT)
        selected = self._limit_layer(selected, 4, RECENT_SUMMARY_LIMIT)
        return sorted(
            selected,
            key=lambda item: (
                not item.required,
                item.layer,
                -item.relevance,
                item.temporal_distance,
                item.stable_key,
            ),
        )

    @staticmethod
    def _temporal_distance(
        row: ContextSource,
        artifact: WorkflowArtifact | None,
        step: WorkflowStep,
        max_version: int,
    ) -> int:
        if artifact is not None:
            if artifact.ordinal is not None and step.ordinal is not None:
                return abs(step.ordinal - artifact.ordinal)
            artifact_step_distance = abs(
                step.position
                - (artifact.ordinal if artifact.ordinal is not None else step.position)
            )
            return artifact_step_distance
        return max(0, max_version - row.source_version)

    @staticmethod
    def _excerpt_offsets(
        artifact: WorkflowArtifact | None,
    ) -> tuple[int | None, int | None]:
        if artifact is None:
            return None, None
        start = artifact.payload.get("excerpt_start")
        end = artifact.payload.get("excerpt_end")
        if start is None and end is None:
            return None, None
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
        ):
            raise ValueError("excerpt offsets must form a positive range")
        return start, end

    @staticmethod
    def _limit_layer(
        candidates: list[ContextCandidate], layer: int, limit: int
    ) -> list[ContextCandidate]:
        layer_items = sorted(
            (item for item in candidates if item.layer == layer),
            key=lambda item: (item.temporal_distance, item.stable_key),
        )[:limit]
        return [item for item in candidates if item.layer != layer] + layer_items


class ContextService:
    def __init__(
        self, session: Session, budgeter: ContextBudgeter | None = None
    ) -> None:
        self.session = session
        self.budgeter = budgeter or ContextBudgeter()

    def build_packet(
        self,
        workflow_id: str,
        step_id: str,
        required: list[ContextCandidate] | tuple[ContextCandidate, ...],
        optional: list[ContextCandidate] | tuple[ContextCandidate, ...],
        limits: ContextLimits | Mapping[str, int],
    ) -> ContextPacket:
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        step = self.session.get(WorkflowStep, step_id)
        if workflow is None:
            raise ValueError("workflow not found")
        if step is None:
            raise ValueError("workflow step not found")
        if step.workflow_id != workflow.id:
            raise ValueError("step does not belong to workflow")
        normalized_limits = self._limits(limits)
        candidates = self._deduplicate(required, optional)
        packed = self.budgeter.pack(
            candidates,
            normalized_limits.input_capacity_tokens,
            normalized_limits.reserved_output_tokens,
            normalized_limits.fixed_overhead_tokens,
        )
        packet = ContextPacket(
            id=str(uuid4()),
            workflow_id=workflow.id,
            step_id=step.id,
            max_input_tokens=packed.max_input_tokens,
            used_input_tokens=packed.used_tokens,
            fixed_overhead_tokens=normalized_limits.fixed_overhead_tokens,
            reserved_output_tokens=packed.reserved_output_tokens,
            status="ready",
        )
        selected_keys = {item.stable_key for item in packed.selected}
        trim_reasons = {
            item.item.stable_key: item.reason for item in packed.trimmed
        }
        ordered = sorted(
            candidates,
            key=lambda item: (
                not item.required,
                item.layer,
                -item.relevance,
                item.temporal_distance,
                item.stable_key,
            ),
        )
        self.session.add(packet)
        self.session.add_all(
            [
                ContextPacketItem(
                    id=str(uuid4()),
                    packet_id=packet.id,
                    source_id=item.source_id,
                    stable_source_key=item.stable_key,
                    layer=item.layer,
                    text_snapshot=item.text,
                    selected=item.stable_key in selected_keys,
                    required=item.required,
                    relevance=item.relevance,
                    temporal_distance=item.temporal_distance,
                    estimated_tokens=self.budgeter.estimator.estimate(item.text)
                    + ITEM_FRAMING_TOKENS,
                    excerpt_start=item.excerpt_start,
                    excerpt_end=item.excerpt_end,
                    trim_reason=trim_reasons.get(item.stable_key),
                    position=position,
                )
                for position, item in enumerate(ordered)
            ]
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return packet

    @staticmethod
    def _limits(limits: ContextLimits | Mapping[str, int]) -> ContextLimits:
        if isinstance(limits, ContextLimits):
            return limits
        try:
            return ContextLimits(
                input_capacity_tokens=limits["input_capacity_tokens"],
                reserved_output_tokens=limits["reserved_output_tokens"],
                fixed_overhead_tokens=limits.get("fixed_overhead_tokens", 0),
            )
        except KeyError as exc:
            raise ValueError(f"missing context limit: {exc.args[0]}") from exc

    @staticmethod
    def _deduplicate(
        required: list[ContextCandidate] | tuple[ContextCandidate, ...],
        optional: list[ContextCandidate] | tuple[ContextCandidate, ...],
    ) -> list[ContextCandidate]:
        result: dict[str, ContextCandidate] = {}
        for item in required:
            result.setdefault(item.stable_key, replace(item, required=True))
        for item in optional:
            result.setdefault(item.stable_key, replace(item, required=False))
        return list(result.values())

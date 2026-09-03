from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ainovel.models.outline import OutlineNode, OutlineVersion
from ainovel.models.project import NovelProject

VERSION_ALLOCATION_ATTEMPTS = 3
OUTLINE_APPROVAL_CONFLICT_MESSAGE = "outline approval conflict"


class OutlineApprovalConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class OutlineNodeInput:
    key: str
    parent_key: str | None
    kind: str
    title: str
    order: int
    status: str = "planned"
    payload: dict[str, Any] = field(default_factory=dict)
    author_locked: bool = False


@dataclass
class OutlineNodeView:
    key: str
    parent_key: str | None
    kind: str
    title: str
    order: int
    status: str
    payload: dict[str, Any]
    author_locked: bool
    children: list[OutlineNodeView] = field(default_factory=list)


@dataclass(frozen=True)
class OutlineDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]


class OutlineService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_candidate(
        self, project_id: str, nodes: list[OutlineNodeInput], reason: str
    ) -> OutlineVersion:
        self._validate_nodes(nodes)
        for attempt in range(VERSION_ALLOCATION_ATTEMPTS):
            project = self.session.get(NovelProject, project_id)
            if project is None:
                self.session.rollback()
                raise ValueError("project not found")
            self.session.refresh(project)
            base_version_id = project.official_outline_version_id
            if base_version_id is not None:
                self._official_version_for_project(base_version_id, project_id)
            version_number = self.session.scalar(
                select(func.coalesce(func.max(OutlineVersion.version_number), 0) + 1).where(
                    OutlineVersion.project_id == project_id
                )
            )
            version = OutlineVersion(
                id=str(uuid4()),
                project_id=project_id,
                version_number=version_number,
                status="candidate",
                base_version_id=base_version_id,
                reason=reason,
            )
            self.session.add(version)
            try:
                self.session.flush()
                self.session.add_all(
                    [
                        OutlineNode(
                            id=str(uuid4()),
                            outline_version_id=version.id,
                            stable_key=node.key,
                            parent_key=node.parent_key,
                            kind=node.kind,
                            title=node.title,
                            order=node.order,
                            status=node.status,
                            payload=deepcopy(node.payload),
                            author_locked=node.author_locked,
                        )
                        for node in nodes
                    ]
                )
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                if attempt == VERSION_ALLOCATION_ATTEMPTS - 1:
                    raise
                continue
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
            return version
        raise RuntimeError("outline version allocation exhausted")

    def approve(self, version_id: str) -> OutlineVersion:
        version = self.session.get(OutlineVersion, version_id)
        if version is None:
            self.session.rollback()
            raise ValueError("outline version not found")
        if version.status != "candidate":
            self.session.rollback()
            raise ValueError("only candidate outlines can be approved")
        project = self.session.get(NovelProject, version.project_id)
        if project is None:
            self.session.rollback()
            raise ValueError("project not found")
        base_version_id = version.base_version_id
        pointer_matches_base = (
            NovelProject.official_outline_version_id.is_(None)
            if base_version_id is None
            else NovelProject.official_outline_version_id == base_version_id
        )
        try:
            result = self.session.execute(
                update(NovelProject)
                .where(NovelProject.id == project.id, pointer_matches_base)
                .values(official_outline_version_id=version.id)
            )
            if result.rowcount != 1:
                raise OutlineApprovalConflict(OUTLINE_APPROVAL_CONFLICT_MESSAGE)
            if base_version_id is not None:
                previous = self._official_version_for_project(base_version_id, project.id)
                previous.status = "superseded"
            version.status = "official"
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return version

    def get_tree(self, version_id: str) -> list[OutlineNodeView]:
        self._get_version(version_id)
        nodes = self.session.scalars(
            select(OutlineNode)
            .where(OutlineNode.outline_version_id == version_id)
            .order_by(OutlineNode.order, OutlineNode.stable_key)
        ).all()
        views = {
            node.stable_key: OutlineNodeView(
                key=node.stable_key,
                parent_key=node.parent_key,
                kind=node.kind,
                title=node.title,
                order=node.order,
                status=node.status,
                payload=deepcopy(node.payload),
                author_locked=node.author_locked,
            )
            for node in nodes
        }
        roots: list[OutlineNodeView] = []
        for node in nodes:
            view = views[node.stable_key]
            if node.parent_key is None:
                roots.append(view)
            else:
                views[node.parent_key].children.append(view)
        return roots

    def compare(self, from_version_id: str, to_version_id: str) -> OutlineDiff:
        from_version = self._get_version(from_version_id)
        to_version = self._get_version(to_version_id)
        if from_version.project_id != to_version.project_id:
            raise ValueError("outline versions must belong to the same project")
        source = self._canonical_nodes(from_version_id)
        target = self._canonical_nodes(to_version_id)
        source_keys = set(source)
        target_keys = set(target)
        return OutlineDiff(
            added=sorted(target_keys - source_keys),
            removed=sorted(source_keys - target_keys),
            changed=sorted(
                key for key in source_keys & target_keys if source[key] != target[key]
            ),
        )

    def count_versions(self, project_id: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(OutlineVersion).where(OutlineVersion.project_id == project_id)
        ) or 0

    def current_official_for_project(self, project_id: str) -> OutlineVersion | None:
        project = self.session.get(NovelProject, project_id)
        if project is None:
            raise ValueError("project not found")
        if project.official_outline_version_id is None:
            return None
        return self._official_version_for_project(project.official_outline_version_id, project.id)

    def _get_version(self, version_id: str) -> OutlineVersion:
        version = self.session.get(OutlineVersion, version_id)
        if version is None:
            raise ValueError("outline version not found")
        return version

    def _official_version_for_project(self, version_id: str, project_id: str) -> OutlineVersion:
        version = self._get_version(version_id)
        if version.project_id != project_id:
            raise ValueError("official outline belongs to another project")
        if version.status != "official":
            raise ValueError("project outline pointer must reference an official outline")
        return version

    def _canonical_nodes(self, version_id: str) -> dict[str, dict[str, Any]]:
        nodes = self.session.scalars(
            select(OutlineNode).where(OutlineNode.outline_version_id == version_id)
        ).all()
        return {
            node.stable_key: {
                "parent_key": node.parent_key,
                "kind": node.kind,
                "title": node.title,
                "order": node.order,
                "status": node.status,
                "payload": node.payload,
                "author_locked": node.author_locked,
            }
            for node in nodes
        }

    @staticmethod
    def _validate_nodes(nodes: list[OutlineNodeInput]) -> None:
        if not nodes:
            raise ValueError("outline must contain exactly one root")
        keys = [node.key for node in nodes]
        if any(not key for key in keys):
            raise ValueError("outline node keys are required")
        if len(set(keys)) != len(keys):
            raise ValueError("outline node keys must be unique")
        key_set = set(keys)
        roots = [node for node in nodes if node.parent_key is None]
        if len(roots) != 1:
            raise ValueError("outline must contain exactly one root")
        for node in nodes:
            if node.parent_key is not None and node.parent_key not in key_set:
                raise ValueError("outline node parent does not exist")
        parents = {node.key: node.parent_key for node in nodes}
        for key in keys:
            seen: set[str] = set()
            current: str | None = key
            while current is not None:
                if current in seen:
                    raise ValueError("outline must not contain cycles")
                seen.add(current)
                current = parents[current]

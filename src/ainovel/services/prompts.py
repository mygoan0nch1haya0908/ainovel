from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from typing import cast
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ainovel.agents.prompts import AGENT_PARAMETERS, AGENT_SCHEMAS, BUILTIN_PROMPTS
from ainovel.models.prompt import PromptVersion, WorkflowPromptSnapshot


PROMPT_VERSION_ALLOCATION_ATTEMPTS = 3
PROMPT_ROLES = tuple(BUILTIN_PROMPTS)


class PromptService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def ensure_builtins(self) -> list[PromptVersion]:
        versions: list[PromptVersion] = []
        for role, body in BUILTIN_PROMPTS.items():
            content_hash = self._content_hash(body)
            version = self.session.scalar(
                select(PromptVersion).where(
                    PromptVersion.role == role,
                    PromptVersion.content_hash == content_hash,
                )
            )
            if version is None:
                version = self.create_version(role, body, "builtin")
            active = self.session.scalar(
                select(PromptVersion.id).where(
                    PromptVersion.role == role,
                    PromptVersion.active.is_(True),
                )
            )
            if active is None:
                version = self.activate(version.id)
            versions.append(version)
        return versions

    def create_version(self, role: str, body: str, source: str) -> PromptVersion:
        self._require_known_role(role)
        if not body.strip():
            raise ValueError("prompt body is required")
        for attempt in range(PROMPT_VERSION_ALLOCATION_ATTEMPTS):
            version_number = cast(
                int,
                self.session.scalar(
                    select(func.coalesce(func.max(PromptVersion.version_number), 0) + 1).where(
                        PromptVersion.role == role
                    )
                ),
            )
            version = PromptVersion(
                id=str(uuid4()),
                role=role,
                version_number=version_number,
                body=body,
                content_hash=self._content_hash(body),
                active=False,
                source=source,
            )
            self.session.add(version)
            try:
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                if attempt == PROMPT_VERSION_ALLOCATION_ATTEMPTS - 1:
                    raise
                continue
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
            return version
        raise RuntimeError("prompt version allocation exhausted")

    def activate(self, prompt_version_id: str) -> PromptVersion:
        self.session.expire_all()
        target = self.session.get(PromptVersion, prompt_version_id)
        if target is None:
            self.session.rollback()
            raise ValueError("prompt version not found")
        self._require_known_role(target.role)
        if target.active:
            return target
        current_active_id = self.session.scalar(
            select(PromptVersion.id).where(
                PromptVersion.role == target.role,
                PromptVersion.active.is_(True),
            )
        )
        try:
            if current_active_id is not None:
                deactivated = self.session.execute(
                    update(PromptVersion)
                    .where(
                        PromptVersion.role == target.role,
                        PromptVersion.id == current_active_id,
                        PromptVersion.active.is_(True),
                    )
                    .values(active=False)
                )
                if deactivated.rowcount != 1:
                    raise ValueError("prompt activation conflict")
            activated = self.session.execute(
                update(PromptVersion)
                .where(
                    PromptVersion.id == target.id,
                    PromptVersion.role == target.role,
                    PromptVersion.active.is_(False),
                )
                .values(active=True)
            )
            if activated.rowcount != 1:
                raise ValueError("prompt activation conflict")
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        self.session.expire_all()
        return self._get_version(target.id)

    def snapshot(
        self,
        workflow_id: str,
        schemas: Mapping[str, type[BaseModel]],
        parameters: Mapping[str, dict[str, object]],
    ) -> list[WorkflowPromptSnapshot]:
        self._require_complete_role_mapping(schemas, "prompt schemas")
        self._require_complete_role_mapping(parameters, "prompt parameters")
        existing_by_role = {
            row.role: row
            for row in self.list_snapshots(workflow_id)
        }
        snapshots = list(existing_by_role.values())
        for role in PROMPT_ROLES:
            if role in existing_by_role:
                continue
            active = self.session.scalar(
                select(PromptVersion).where(
                    PromptVersion.role == role,
                    PromptVersion.active.is_(True),
                )
            )
            if active is None:
                raise ValueError(f"no active prompt for role: {role}")
            snapshot = WorkflowPromptSnapshot(
                id=str(uuid4()),
                workflow_id=workflow_id,
                role=role,
                prompt_version_id=active.id,
                prompt_body=active.body,
                output_schema=deepcopy(schemas[role].model_json_schema()),
                parameters=deepcopy(parameters[role]),
            )
            self.session.add(snapshot)
            snapshots.append(snapshot)
        self.session.flush()
        return sorted(snapshots, key=lambda row: row.role)

    def list_snapshots(self, workflow_id: str) -> list[WorkflowPromptSnapshot]:
        return self.session.scalars(
            select(WorkflowPromptSnapshot)
            .where(WorkflowPromptSnapshot.workflow_id == workflow_id)
            .order_by(WorkflowPromptSnapshot.role)
        ).all()

    def _get_version(self, prompt_version_id: str) -> PromptVersion:
        version = self.session.get(PromptVersion, prompt_version_id)
        if version is None:
            raise ValueError("prompt version not found")
        return version

    @staticmethod
    def _content_hash(body: str) -> str:
        return sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def _require_known_role(role: str) -> None:
        if role not in PROMPT_ROLES:
            raise ValueError("unknown prompt role")

    @staticmethod
    def _require_complete_role_mapping(mapping: Mapping[str, object], label: str) -> None:
        if set(mapping) != set(PROMPT_ROLES):
            raise ValueError(f"{label} must cover all prompt roles")

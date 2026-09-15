from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ainovel.models.workflow import (
    ChapterDraftRepair,
    GenerationWorkflow,
    ModelAttempt,
    WorkflowArtifact,
    WorkflowStep,
)


MAX_CHAPTER_REPAIRS = 2
REPAIR_TARGET_CHARACTERS = 5200
MIN_CHAPTER_CHARACTERS = 4500


@dataclass(frozen=True)
class ChapterDraftRepairView:
    id: str
    writing_step_id: str
    ordinal: int
    latest_attempt_id: str
    title: str
    body: str
    visible_count: int
    repair_count: int
    repair_pending: bool
    draft_revision: int
    updated_at: datetime


def repair_payload(state: ChapterDraftRepair) -> dict[str, object]:
    return {
        "draft": deepcopy(state.latest_payload),
        "visible_count": state.visible_count,
        "minimum_gap": MIN_CHAPTER_CHARACTERS - state.visible_count,
        "target_characters": REPAIR_TARGET_CHARACTERS,
        "instruction": "在已批准场景内扩写，返回完整修订章；不得提前消耗后续事件或重复填充。",
    }


class DraftRepairService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_writing_step(self, writing_step_id: str) -> ChapterDraftRepair | None:
        return self.session.scalar(
            select(ChapterDraftRepair).where(
                ChapterDraftRepair.writing_step_id == writing_step_id
            )
        )

    def list_for_workflow(
        self, workflow_id: str
    ) -> tuple[ChapterDraftRepairView, ...]:
        rows = self.session.execute(
            select(ChapterDraftRepair, WorkflowStep.ordinal)
            .join(WorkflowStep, WorkflowStep.id == ChapterDraftRepair.writing_step_id)
            .where(ChapterDraftRepair.workflow_id == workflow_id)
            .order_by(WorkflowStep.ordinal)
        ).all()
        views: list[ChapterDraftRepairView] = []
        for state, ordinal in rows:
            title = state.latest_payload.get("title")
            body = state.latest_payload.get("body")
            if not isinstance(ordinal, int) or not isinstance(title, str) or not isinstance(body, str):
                raise ValueError("persisted chapter repair state is invalid")
            views.append(
                ChapterDraftRepairView(
                    id=state.id,
                    writing_step_id=state.writing_step_id,
                    ordinal=ordinal,
                    latest_attempt_id=state.latest_attempt_id,
                    title=title,
                    body=body,
                    visible_count=state.visible_count,
                    repair_count=state.repair_count,
                    repair_pending=state.repair_pending,
                    draft_revision=state.draft_revision,
                    updated_at=state.updated_at,
                )
            )
        return tuple(views)

    def reserve_repair_dispatch(
        self, step_id: str, worker_id: str, *, claim_revision: int
    ) -> bool:
        self.session.expire_all()
        step = self.session.get(WorkflowStep, step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        state = self.get_for_writing_step(step_id)
        if state is None:
            self.session.rollback()
            return True
        if (
            step is None
            or workflow is None
            or workflow.generation_version != 2
            or step.kind != "WRITING"
            or step.status != "RUNNING"
            or step.revision != claim_revision
            or step.lease_owner != worker_id
        ):
            self.session.rollback()
            raise ValueError("repair dispatch requires the active writing lease")
        if state.repair_pending:
            self.session.rollback()
            return True
        if state.repair_count >= MAX_CHAPTER_REPAIRS:
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                )
                .values(
                    status="PAUSED_REVIEW",
                    revision=workflow.revision + 1,
                    last_error_code="repair_attempts_exhausted",
                    last_error_detail="chapter remains below 4500 visible characters after two repairs",
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == claim_revision,
                    WorkflowStep.lease_owner == worker_id,
                )
                .values(
                    status="PAUSED",
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            if workflow_claim.rowcount != 1 or step_claim.rowcount != 1:
                self.session.rollback()
                raise ValueError("repair exhaustion pause conflict")
            self.session.commit()
            return False
        claimed = self.session.execute(
            update(ChapterDraftRepair)
            .where(
                ChapterDraftRepair.id == state.id,
                ChapterDraftRepair.draft_revision == state.draft_revision,
                ChapterDraftRepair.repair_count == state.repair_count,
            )
            .values(
                repair_count=state.repair_count + 1,
                repair_pending=True,
                draft_revision=state.draft_revision + 1,
            )
        )
        if claimed.rowcount != 1:
            self.session.rollback()
            raise ValueError("repair dispatch conflict")
        self.session.commit()
        return True


def persist_work_draft(
    session: Session,
    workflow: GenerationWorkflow,
    step: WorkflowStep,
    attempt: ModelAttempt,
    artifact: WorkflowArtifact,
) -> ChapterDraftRepair:
    state = session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.writing_step_id == step.id
        )
    )
    if state is None:
        state = ChapterDraftRepair(
            id=str(uuid4()),
            workflow_id=workflow.id,
            writing_step_id=step.id,
            latest_attempt_id=attempt.id,
            latest_payload=deepcopy(artifact.payload),
            visible_count=artifact.visible_char_count,
            repair_count=0,
            repair_pending=False,
            draft_revision=1,
        )
        session.add(state)
    else:
        state.latest_attempt_id = attempt.id
        state.latest_payload = deepcopy(artifact.payload)
        state.visible_count = artifact.visible_char_count
        state.repair_pending = False
        state.draft_revision += 1
    return state


def coverage_is_valid(
    session: Session, step: WorkflowStep, payload: dict[str, object]
) -> bool:
    state = _state_for_ordinal(session, step.workflow_id, step.ordinal)
    if state is None:
        return False
    body = state.latest_payload.get("body")
    if not isinstance(body, str):
        return False
    for key in ("goal", "ending_hook"):
        verdict = payload.get(key)
        if not isinstance(verdict, dict) or verdict.get("passed") is not True:
            return False
        excerpt = verdict.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt.strip() or excerpt not in body:
            return False
    return True


def promote_work_draft(
    session: Session,
    workflow: GenerationWorkflow,
    coverage_step: WorkflowStep,
) -> WorkflowArtifact:
    state = _state_for_ordinal(
        session, workflow.id, coverage_step.ordinal
    )
    if state is None or state.visible_count < MIN_CHAPTER_CHARACTERS:
        raise ValueError("coverage promotion requires an accepted work draft")
    existing = session.scalar(
        select(WorkflowArtifact).where(
            WorkflowArtifact.workflow_id == workflow.id,
            WorkflowArtifact.kind == "chapter_draft",
            WorkflowArtifact.ordinal == coverage_step.ordinal,
        )
    )
    if existing is not None:
        return existing
    body = state.latest_payload["body"]
    return WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=coverage_step.id,
        kind="chapter_draft",
        ordinal=coverage_step.ordinal,
        text_content=body,
        payload=deepcopy(state.latest_payload),
        visible_char_count=state.visible_count,
        content_hash=sha256(body.encode("utf-8")).hexdigest(),
    )


def _state_for_ordinal(
    session: Session, workflow_id: str, ordinal: int | None
) -> ChapterDraftRepair | None:
    writing_step = session.scalar(
        select(WorkflowStep).where(
            WorkflowStep.workflow_id == workflow_id,
            WorkflowStep.kind == "WRITING",
            WorkflowStep.ordinal == ordinal,
        )
    )
    if writing_step is None:
        return None
    return session.scalar(
        select(ChapterDraftRepair).where(
            ChapterDraftRepair.writing_step_id == writing_step.id
        )
    )

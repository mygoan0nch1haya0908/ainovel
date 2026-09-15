from __future__ import annotations
from ainovel.providers.diagnostics import ResponseFailure, safe_failure_detail

from collections.abc import Mapping, Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import func, select, update, or_
from sqlalchemy.orm import Session

from ainovel.agents.prompts import (
    AGENT_SCHEMAS,
    V2_AGENT_SCHEMAS,
    V2_BUILTIN_PROMPTS,
)
from ainovel.context import RequiredContextOverflow
from ainovel.models.audit import AuditEvent
from ainovel.models.batch import WritingBatch
from ainovel.models.outline import OutlineVersion
from ainovel.models.project import ConstitutionVersion, NovelProject
from ainovel.models.stage import StoryStage, StageWorkflow, StageRoadmapVersion
from ainovel.models.workflow import (
    GenerationWorkflow,
    ModelAttempt,
    PlanDecision,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.providers.contracts import (
    ModelResponse,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
)
from ainovel.services.counting import count_visible_characters
from ainovel.services.draft_repair import (
    coverage_is_valid,
    persist_work_draft,
    promote_work_draft,
)
from ainovel.services.prompts import PromptService


@dataclass(frozen=True)
class WorkflowBudgets:
    planner_input: int = 16_000
    planner_output: int = 4_000
    writer_input: int = 32_000
    writer_output: int = 12_000
    summarizer_input: int = 16_000
    summarizer_output: int = 4_000
    reviewer_input: int = 32_000
    reviewer_output: int = 6_000


DEFAULT_BUDGETS = WorkflowBudgets()
PROVIDER_NAMES = frozenset({"fake", "ollama", "openai", "qwen"})
WORKFLOW_STATUSES = frozenset(
    {
        "PREPARING",
        "PLANNING",
        "AWAITING_PLAN_APPROVAL",
        "GENERATING_CHAPTERS",
        "REVIEWING_BATCH",
        "CREATING_CANDIDATE_BATCH",
        "AWAITING_CONTENT_APPROVAL",
        "PAUSED_PROVIDER",
        "PAUSED_CONTEXT_OVERFLOW",
        "PAUSED_ATTEMPTS",
        "PAUSED_REVIEW",
        "PAUSED_STALE_VERSION",
        "COMPLETED",
        "REJECTED",
        "CANCELLED",
        "FAILED",
    }
)
STEP_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "PAUSED", "FAILED"})
RESUMABLE_WORKFLOW_STATUSES = frozenset(
    {"PAUSED_PROVIDER", "PAUSED_CONTEXT_OVERFLOW"}
)

MAX_STEP_ATTEMPTS = 2
EXECUTABLE_WORKFLOW_STATUSES = frozenset(
    {"PLANNING", "GENERATING_CHAPTERS", "REVIEWING_BATCH", "CREATING_CANDIDATE_BATCH"}
)
TERMINAL_WORKFLOW_STATUSES = frozenset(
    {"COMPLETED", "REJECTED", "CANCELLED", "FAILED"}
)

_ALLOWED_WORKFLOW_TRANSITIONS: dict[str, frozenset[str]] = {
    "PREPARING": frozenset({"PLANNING", "CANCELLED", "FAILED"}),
    "PLANNING": frozenset(
        {
            "PLANNING",
            "AWAITING_PLAN_APPROVAL",
            "PAUSED_PROVIDER",
            "PAUSED_CONTEXT_OVERFLOW",
            "PAUSED_ATTEMPTS",
            "PAUSED_STALE_VERSION",
            "CANCELLED",
            "FAILED",
        }
    ),
    "AWAITING_PLAN_APPROVAL": frozenset(
        {"GENERATING_CHAPTERS", "PAUSED_STALE_VERSION", "REJECTED", "CANCELLED", "FAILED"}
    ),
    "GENERATING_CHAPTERS": frozenset(
        {
            "GENERATING_CHAPTERS",
            "REVIEWING_BATCH",
            "PAUSED_PROVIDER",
            "PAUSED_CONTEXT_OVERFLOW",
            "PAUSED_ATTEMPTS",
            "PAUSED_REVIEW",
            "PAUSED_STALE_VERSION",
            "CANCELLED",
            "FAILED",
        }
    ),
    "REVIEWING_BATCH": frozenset(
        {
            "REVIEWING_BATCH",
            "CREATING_CANDIDATE_BATCH",
            "PAUSED_PROVIDER",
            "PAUSED_CONTEXT_OVERFLOW",
            "PAUSED_ATTEMPTS",
            "PAUSED_REVIEW",
            "PAUSED_STALE_VERSION",
            "CANCELLED",
            "FAILED",
        }
    ),
    "CREATING_CANDIDATE_BATCH": frozenset(
        {
            "CREATING_CANDIDATE_BATCH",
            "AWAITING_CONTENT_APPROVAL",
            "COMPLETED",
            "REJECTED",
            "PAUSED_STALE_VERSION",
            "CANCELLED",
            "FAILED",
        }
    ),
    "AWAITING_CONTENT_APPROVAL": frozenset(
        {"COMPLETED", "REJECTED", "CANCELLED", "FAILED"}
    ),
    "PAUSED_PROVIDER": frozenset(
        {"PLANNING", "GENERATING_CHAPTERS", "REVIEWING_BATCH", "CANCELLED", "FAILED"}
    ),
    "PAUSED_CONTEXT_OVERFLOW": frozenset(
        {"PLANNING", "GENERATING_CHAPTERS", "REVIEWING_BATCH", "CANCELLED", "FAILED"}
    ),
    "PAUSED_ATTEMPTS": frozenset({"CANCELLED", "FAILED"}),
    "PAUSED_REVIEW": frozenset({"CANCELLED", "FAILED"}),
    "PAUSED_STALE_VERSION": frozenset({"CANCELLED", "FAILED"}),
    "COMPLETED": frozenset(),
    "REJECTED": frozenset(),
    "CANCELLED": frozenset(),
    "FAILED": frozenset(),
}

_EXPECTED_ARTIFACT_KINDS = {
    "PLANNING": "batch_plan",
    "WRITING": "chapter_draft",
    "SUMMARIZING": "chapter_summary_delta",
    "REVIEWING": "batch_review",
    "VALIDATING_CHAPTER": "chapter_coverage",
}
_STEP_PROMPT_ROLES = {
    "PLANNING": "batch_planner",
    "WRITING": "chapter_writer",
    "SUMMARIZING": "chapter_summarizer",
    "REVIEWING": "batch_reviewer",
    "VALIDATING_CHAPTER": "chapter_coverage_reviewer",
}

_SAFE_PROVIDER_FAILURES: tuple[
    tuple[type[ProviderError], str, str, bool], ...
] = (
    (
        ProviderAuthenticationError,
        "provider_authentication",
        "provider authentication failed",
        False,
    ),
    (
        ProviderUnavailable,
        "provider_unavailable",
        "provider is unavailable",
        False,
    ),
    (ProviderTimeout, "provider_timeout", "provider request timed out", True),
    (
        ProviderProtocolError,
        "provider_protocol",
        "provider returned an invalid response",
        True,
    ),
)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


class StaleOutlineCompletion(RuntimeError):
    """The provider returned after the workflow's outline snapshot became stale."""


class _AttemptCompletionConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class _ArtifactValues:
    kind: str
    ordinal: int | None
    text_content: str | None
    payload: dict[str, Any]
    visible_char_count: int | None
    content_hash: str


class WorkflowService:
    def __init__(self, session: Session, clock: Clock | None = None) -> None:
        self.session = session
        self.clock = clock or SystemClock()

    def start(
        self,
        project_id: str,
        provider_name: str,
        model_name: str,
        requested_chapters: int,
        budgets: WorkflowBudgets,
        *,
        generation_version: int = 1,
        _before_commit: Callable[[GenerationWorkflow], None] | None = None,
    ) -> GenerationWorkflow:
        self._validate_start_arguments(
            provider_name, model_name, requested_chapters, budgets
        )
        if type(generation_version) is not int or generation_version not in {1, 2}:
            raise ValueError("generation version must be 1 or 2")
        try:
            initial_state = self._read_valid_start_state(project_id)
        except Exception:
            self.session.rollback()
            raise
        self.session.rollback()

        PromptService(self.session).ensure_builtins()

        try:
            current_state = self._read_valid_start_state(project_id)
        except Exception:
            self.session.rollback()
            raise
        self.session.rollback()
        if current_state != initial_state:
            raise ValueError("project state changed before workflow ownership")
        constitution_id, outline_id = current_state
        workflow_id = str(uuid4())

        try:
            ownership = self.session.execute(
                update(NovelProject)
                .where(
                    NovelProject.id == project_id,
                    NovelProject.active_workflow_id.is_(None),
                    NovelProject.active_batch_id.is_(None),
                    NovelProject.current_constitution_version_id == constitution_id,
                    NovelProject.official_outline_version_id == outline_id,
                )
                .values(active_workflow_id=workflow_id)
            )
            if ownership.rowcount != 1:
                self.session.rollback()
                self._raise_start_conflict(project_id)

            workflow = GenerationWorkflow(
                id=workflow_id,
                project_id=project_id,
                base_outline_version_id=outline_id,
                provider_name=provider_name,
                model_name=model_name.strip(),
                requested_chapters=requested_chapters,
                generation_version=generation_version,
                model_call_limit=(
                    4 + 10 * requested_chapters if generation_version == 2 else None
                ),
                total_input_token_limit=(
                    2 * budgets.planner_input
                    + requested_chapters
                    * (
                        6 * budgets.writer_input
                        + 2 * budgets.reviewer_input
                        + 2 * budgets.summarizer_input
                    )
                    + 2 * budgets.reviewer_input
                    if generation_version == 2
                    else None
                ),
                total_output_token_limit=(
                    2 * budgets.planner_output
                    + requested_chapters
                    * (
                        6 * budgets.writer_output
                        + 2 * budgets.reviewer_output
                        + 2 * budgets.summarizer_output
                    )
                    + 2 * budgets.reviewer_output
                    if generation_version == 2
                    else None
                ),
                model_calls_used=0,
                status="PLANNING",
                current_position=0,
                planner_input_tokens=budgets.planner_input,
                planner_output_tokens=budgets.planner_output,
                writer_input_tokens=budgets.writer_input,
                writer_output_tokens=budgets.writer_output,
                summarizer_input_tokens=budgets.summarizer_input,
                summarizer_output_tokens=budgets.summarizer_output,
                reviewer_input_tokens=budgets.reviewer_input,
                reviewer_output_tokens=budgets.reviewer_output,
                actual_input_tokens=0,
                actual_output_tokens=0,
                revision=1,
            )
            self.session.add(workflow)
            self.session.flush()
            prompt_service = PromptService(self.session)
            if generation_version == 1:
                prompt_service.snapshot(
                    workflow.id, AGENT_SCHEMAS, self._prompt_parameters(budgets)
                )
            else:
                prompt_service.snapshot_versioned(
                    workflow.id,
                    V2_BUILTIN_PROMPTS,
                    V2_AGENT_SCHEMAS,
                    self._v2_prompt_parameters(budgets),
                )
            self.session.add(
                WorkflowStep(
                    id=str(uuid4()),
                    workflow_id=workflow.id,
                    kind="PLANNING",
                    ordinal=None,
                    position=0,
                    status="PENDING",
                    attempt_count=0,
                    revision=1,
                )
            )
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "workflow_started",
                "author",
                {
                    "provider_name": provider_name,
                    "model_name": model_name.strip(),
                    "requested_chapters": requested_chapters,
                    "base_outline_version_id": outline_id,
                    "generation_version": generation_version,
                    "budgets": asdict(budgets),
                },
            )
            if _before_commit is not None:
                _before_commit(workflow)
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def claim_step(
        self,
        workflow_id: str,
        expected_statuses: set[str] | frozenset[str],
        worker_id: str,
        lease_seconds: int = 300,
    ) -> WorkflowStep | None:
        statuses = frozenset(expected_statuses)
        if not statuses or not statuses <= WORKFLOW_STATUSES:
            raise ValueError("expected workflow statuses are invalid")
        normalized_worker = worker_id.strip() if isinstance(worker_id, str) else ""
        if not normalized_worker:
            raise ValueError("worker id is required")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease seconds must be a positive integer")
        now = self._aware_utc(self.clock.now())
        lease_expires_at = now + timedelta(seconds=lease_seconds)

        self.session.expire_all()
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        if workflow is None:
            self.session.rollback()
            raise ValueError("workflow not found")
        if workflow.status not in statuses:
            self.session.rollback()
            return None
        project = self.session.get(NovelProject, workflow.project_id)
        if project is None or project.active_workflow_id != workflow.id:
            self.session.rollback()
            return None
        step = self.session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.position == workflow.current_position,
            )
        )
        if not self._inputs_are_current(workflow, project):
            if step is not None and step.status in {"PENDING", "RUNNING"}:
                self._pause_stale_workflow(workflow, step)
            else:
                self.session.rollback()
            return None
        if (
            step is None
            or step.status != "PENDING"
            or step.active_artifact_id is not None
            or step.lease_owner is not None
            or step.lease_expires_at is not None
        ):
            self.session.rollback()
            return None

        self._require_transition(workflow.status, workflow.status)
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
                GenerationWorkflow.current_position == workflow.current_position,
            )
            .values(revision=workflow.revision + 1)
        )
        if workflow_claim.rowcount != 1:
            self.session.rollback()
            return None
        step_claim = self.session.execute(
            update(WorkflowStep)
            .where(
                WorkflowStep.id == step.id,
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.status == "PENDING",
                WorkflowStep.revision == step.revision,
                WorkflowStep.active_artifact_id.is_(None),
                WorkflowStep.lease_owner.is_(None),
                WorkflowStep.lease_expires_at.is_(None),
            )
            .values(
                status="RUNNING",
                lease_owner=normalized_worker,
                lease_expires_at=lease_expires_at,
                revision=step.revision + 1,
            )
        )
        if step_claim.rowcount != 1:
            self.session.rollback()
            return None
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return step

    def recover_expired_claims(self, workflow_id: str, now: datetime) -> int:
        normalized_now = self._aware_utc(now)
        self.session.expire_all()
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        if workflow is None:
            self.session.rollback()
            raise ValueError("workflow not found")
        candidates = self.session.scalars(
            select(WorkflowStep)
            .where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.status == "RUNNING",
                WorkflowStep.active_artifact_id.is_(None),
                WorkflowStep.lease_owner.is_not(None),
                WorkflowStep.lease_expires_at.is_not(None),
                WorkflowStep.lease_expires_at <= normalized_now,
            )
            .order_by(WorkflowStep.position)
        ).all()
        if not candidates:
            self.session.rollback()
            return 0

        self._require_transition(workflow.status, workflow.status)
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
            )
            .values(revision=workflow.revision + 1)
        )
        if workflow_claim.rowcount != 1:
            self.session.rollback()
            return 0

        recovered_ids: list[str] = []
        for step in candidates:
            abandoned_attempt = self.session.scalar(
                select(ModelAttempt.id)
                .where(
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.status == "RUNNING",
                )
                .limit(1)
            )
            protocol_failure_count = step.protocol_failure_count
            if (
                workflow.generation_version == 2
                and step.kind == "WRITING"
                and abandoned_attempt is not None
            ):
                protocol_failure_count += 1
            recovered = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner == step.lease_owner,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                    WorkflowStep.lease_expires_at <= normalized_now,
                )
                .values(
                    status="PENDING",
                    lease_owner=None,
                    lease_expires_at=None,
                    protocol_failure_count=protocol_failure_count,
                    revision=step.revision + 1,
                )
            )
            if recovered.rowcount == 1:
                recovered_ids.append(step.id)
        if not recovered_ids:
            self.session.rollback()
            return 0
        self.session.execute(
            update(ModelAttempt)
            .where(
                ModelAttempt.step_id.in_(recovered_ids),
                ModelAttempt.status == "RUNNING",
            )
            .values(
                status="FAILED",
                error_code="lease_expired",
                error_detail="step lease expired before completion",
            )
        )
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return len(recovered_ids)

    def record_attempt_start(
        self, step_id: str, request_digest: str, *, claim_revision: int
    ) -> ModelAttempt:
        if (
            not isinstance(request_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", request_digest) is None
        ):
            raise ValueError("request digest must be a canonical SHA-256 hex digest")
        normalized_digest = request_digest
        now = self._aware_utc(self.clock.now())
        self.session.expire_all()
        step = self.session.get(WorkflowStep, step_id)
        if step is None:
            self.session.rollback()
            raise ValueError("workflow step not found")
        workflow = self.session.get(GenerationWorkflow, step.workflow_id)
        if workflow is None:
            self.session.rollback()
            raise ValueError("workflow not found")
        if (
            step.status != "RUNNING"
            or step.revision != claim_revision
            or step.lease_owner is None
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
        ):
            self.session.rollback()
            raise ValueError("attempt start requires an active step lease")
        attempt_limit_count = (
            step.protocol_failure_count
            if workflow.generation_version == 2 and step.kind == "WRITING"
            else step.attempt_count
        )
        if attempt_limit_count >= MAX_STEP_ATTEMPTS:
            self._pause_attempt_exhaustion(workflow, step)
            raise ValueError("maximum step attempts reached")
        if workflow.generation_version == 2:
            input_budget, output_budget = self._step_token_budget(workflow, step)
            if (
                workflow.model_call_limit is None
                or workflow.total_input_token_limit is None
                or workflow.total_output_token_limit is None
                or workflow.model_calls_used >= workflow.model_call_limit
                or workflow.actual_input_tokens + input_budget
                > workflow.total_input_token_limit
                or workflow.actual_output_tokens + output_budget
                > workflow.total_output_token_limit
            ):
                self._pause_workflow_budget_exhaustion(workflow, step)
                raise ValueError("workflow model budget exhausted")

        attempt_number = step.attempt_count + 1
        self._require_transition(workflow.status, workflow.status)
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
                GenerationWorkflow.current_position == step.position,
            )
            .values(
                revision=workflow.revision + 1,
                model_calls_used=(
                    workflow.model_calls_used + 1
                    if workflow.generation_version == 2
                    else workflow.model_calls_used
                ),
            )
        )
        step_claim = self.session.execute(
            update(WorkflowStep)
            .where(
                WorkflowStep.id == step.id,
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.status == "RUNNING",
                WorkflowStep.revision == step.revision,
                WorkflowStep.attempt_count == step.attempt_count,
                WorkflowStep.lease_owner == step.lease_owner,
                WorkflowStep.lease_expires_at == step.lease_expires_at,
                WorkflowStep.revision == claim_revision,
                WorkflowStep.lease_expires_at > now,
                ~select(ModelAttempt.id).where(
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.status == "RUNNING",
                ).exists(),
            )
            .values(
                attempt_count=attempt_number,
                revision=step.revision + 1,
            )
        )
        if workflow_claim.rowcount != 1 or step_claim.rowcount != 1:
            self.session.rollback()
            raise ValueError("attempt start conflict")
        attempt = ModelAttempt(
            id=str(uuid4()),
            step_id=step.id,
            attempt_number=attempt_number,
            status="RUNNING",
            request_digest=normalized_digest,
        )
        self.session.add(attempt)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return attempt

    def preflight_model_budget(
        self, step_id: str, worker_id: str, *, claim_revision: int
    ) -> bool:
        self.session.expire_all()
        step = self.session.get(WorkflowStep, step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        if (
            step is None
            or workflow is None
            or step.status != "RUNNING"
            or step.revision != claim_revision
            or step.lease_owner != worker_id
        ):
            self.session.rollback()
            raise ValueError("budget preflight requires the active step lease")
        if workflow.generation_version != 2:
            self.session.rollback()
            return True
        input_budget, output_budget = self._step_token_budget(workflow, step)
        available = (
            workflow.model_call_limit is not None
            and workflow.total_input_token_limit is not None
            and workflow.total_output_token_limit is not None
            and workflow.model_calls_used < workflow.model_call_limit
            and workflow.actual_input_tokens + input_budget
            <= workflow.total_input_token_limit
            and workflow.actual_output_tokens + output_budget
            <= workflow.total_output_token_limit
        )
        if available:
            self.session.rollback()
            return True
        self._pause_workflow_budget_exhaustion(workflow, step)
        return False

    def pause_context_overflow(
        self,
        step_id: str,
        worker_id: str,
        error: RequiredContextOverflow,
        *,
        claim_revision: int,
    ) -> GenerationWorkflow:
        normalized_worker = worker_id.strip() if isinstance(worker_id, str) else ""
        if not normalized_worker:
            raise ValueError("worker id is required")
        if not isinstance(error, RequiredContextOverflow):
            raise TypeError("context overflow pause requires RequiredContextOverflow")
        now = self._aware_utc(self.clock.now())
        self.session.expire_all()
        step = self.session.get(WorkflowStep, step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        running_attempt_id = (
            self.session.scalar(
                select(ModelAttempt.id)
                .where(
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.status == "RUNNING",
                )
                .limit(1)
            )
            if step is not None
            else None
        )
        if (
            step is None
            or workflow is None
            or workflow.status not in EXECUTABLE_WORKFLOW_STATUSES
            or workflow.current_position != step.position
            or step.kind not in _STEP_PROMPT_ROLES
            or step.status != "RUNNING"
            or step.revision != claim_revision
            or step.active_artifact_id is not None
            or step.lease_owner != normalized_worker
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
            or running_attempt_id is not None
        ):
            self.session.rollback()
            raise ValueError("context overflow pause conflict")

        self._require_transition(workflow.status, "PAUSED_CONTEXT_OVERFLOW")
        try:
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                )
                .values(
                    status="PAUSED_CONTEXT_OVERFLOW",
                    revision=workflow.revision + 1,
                    last_error_code="required_context_overflow",
                    last_error_detail=(
                        "required context exceeds the available input budget"
                    ),
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == step.attempt_count,
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner == normalized_worker,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                    WorkflowStep.revision == claim_revision,
                    WorkflowStep.lease_expires_at > now,
                    ~select(ModelAttempt.id)
                    .where(
                        ModelAttempt.step_id == step.id,
                        ModelAttempt.status == "RUNNING",
                    )
                    .exists(),
                )
                .values(
                    status="PAUSED",
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            if workflow_claim.rowcount != 1 or step_claim.rowcount != 1:
                raise ValueError("context overflow pause conflict")
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "workflow_paused",
                "system",
                {"reason": "required_context_overflow"},
            )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def pause_provider_failure(
        self,
        step_id: str,
        worker_id: str,
        error: ProviderError,
        *,
        claim_revision: int,
    ) -> GenerationWorkflow:
        normalized_worker = worker_id.strip() if isinstance(worker_id, str) else ""
        if not normalized_worker:
            raise ValueError("worker id is required")
        if not isinstance(error, ProviderError):
            raise TypeError("provider pause requires a typed ProviderError")
        code, detail, _retryable = self._provider_failure(error)
        now = self._aware_utc(self.clock.now())
        self.session.expire_all()
        step = self.session.get(WorkflowStep, step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        running_attempt_id = (
            self.session.scalar(
                select(ModelAttempt.id)
                .where(
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.status == "RUNNING",
                )
                .limit(1)
            )
            if step is not None
            else None
        )
        if (
            step is None
            or workflow is None
            or workflow.status not in EXECUTABLE_WORKFLOW_STATUSES
            or workflow.current_position != step.position
            or step.kind not in _STEP_PROMPT_ROLES
            or step.status != "RUNNING"
            or step.revision != claim_revision
            or step.active_artifact_id is not None
            or step.lease_owner != normalized_worker
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
            or running_attempt_id is not None
        ):
            self.session.rollback()
            raise ValueError("provider pause conflict")

        self._require_transition(workflow.status, "PAUSED_PROVIDER")
        try:
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                )
                .values(
                    status="PAUSED_PROVIDER",
                    revision=workflow.revision + 1,
                    last_error_code=code,
                    last_error_detail=detail,
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == step.attempt_count,
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner == normalized_worker,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                    WorkflowStep.revision == claim_revision,
                    WorkflowStep.lease_expires_at > now,
                    ~select(ModelAttempt.id)
                    .where(
                        ModelAttempt.step_id == step.id,
                        ModelAttempt.status == "RUNNING",
                    )
                    .exists(),
                )
                .values(
                    status="PAUSED",
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            if workflow_claim.rowcount != 1 or step_claim.rowcount != 1:
                raise ValueError("provider pause conflict")
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "workflow_paused",
                "system",
                {"reason": code},
            )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def complete_attempt(
        self,
        attempt_id: str,
        response: ModelResponse,
        artifact: BaseModel | Mapping[str, object],
        finalize_step: bool = True,
    ) -> WorkflowArtifact:
        self._validate_response(response)
        self.session.expire_all()
        attempt = self.session.get(ModelAttempt, attempt_id)
        if attempt is None:
            self.session.rollback()
            raise ValueError("model attempt not found")
        step = self.session.get(WorkflowStep, attempt.step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        project = (
            self.session.get(NovelProject, workflow.project_id)
            if workflow is not None
            else None
        )
        now = self._aware_utc(self.clock.now())
        if (
            step is None
            or workflow is None
            or project is None
            or project.active_workflow_id != workflow.id
            or attempt.status != "RUNNING"
            or attempt.attempt_number != step.attempt_count
            or step.status != "RUNNING"
            or step.active_artifact_id is not None
            or step.lease_owner is None
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
            or workflow.current_position != step.position
        ):
            self.session.rollback()
            raise ValueError("attempt completion conflict")
        if not finalize_step and not (
            step.kind == "REVIEWING"
            or workflow.generation_version == 2 and step.kind == "WRITING"
        ):
            self.session.rollback()
            raise ValueError("workflow step may not persist a non-final artifact")
        if not self._inputs_are_current(workflow, project):
            self.session.rollback()
            if self._pause_stale_attempt_completion(attempt_id, response):
                raise StaleOutlineCompletion(
                    "official outline changed during provider call"
                )
            raise ValueError("attempt completion conflict")

        try:
            artifact_values = self._artifact_values(workflow, step, artifact)
            effective_finalize = finalize_step
            if not finalize_step and step.kind == "REVIEWING":
                evidence_requested = (
                    artifact_values.payload.get("passed") is False
                    and bool(artifact_values.payload.get("evidence_queries"))
                )
                if not evidence_requested:
                    raise ValueError("reviewer evidence request is invalid")
                if attempt.attempt_number != 1:
                    effective_finalize = True
            target_status, target_position, target_step_status, make_active = (
                self._completion_transition(
                    workflow, step, artifact_values, effective_finalize
                )
            )
            self._require_transition(workflow.status, target_status)
        except Exception:
            self.session.rollback()
            raise
        artifact_row = WorkflowArtifact(
            id=str(uuid4()),
            workflow_id=workflow.id,
            step_id=step.id,
            kind=artifact_values.kind,
            ordinal=artifact_values.ordinal,
            text_content=artifact_values.text_content,
            payload=deepcopy(artifact_values.payload),
            visible_char_count=artifact_values.visible_char_count,
            content_hash=artifact_values.content_hash,
        )
        self.session.add(artifact_row)
        try:
            self.session.flush()
            promoted_artifact = None
            if workflow.generation_version == 2 and step.kind == "WRITING":
                persist_work_draft(
                    self.session, workflow, step, attempt, artifact_row
                )
            if (
                workflow.generation_version == 2
                and step.kind == "VALIDATING_CHAPTER"
                and target_status != "PAUSED_REVIEW"
            ):
                promoted_artifact = promote_work_draft(
                    self.session, workflow, step
                )
                self.session.add(promoted_artifact)
                self.session.flush()
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                    self._stage_current_predicate(workflow.id),
                    select(NovelProject.id)
                    .where(
                        NovelProject.id == workflow.project_id,
                        NovelProject.active_workflow_id == workflow.id,
                        NovelProject.official_outline_version_id
                        == workflow.base_outline_version_id,
                    )
                    .exists(),
                )
                .values(
                    status=target_status,
                    current_position=target_position,
                    actual_input_tokens=GenerationWorkflow.actual_input_tokens
                    + (response.input_tokens or 0),
                    actual_output_tokens=GenerationWorkflow.actual_output_tokens
                    + (response.output_tokens or 0),
                    revision=workflow.revision + 1,
                    last_error_code=(
                        "review_blocked" if target_status == "PAUSED_REVIEW" else None
                    ),
                    last_error_detail=(
                        "batch review requires author attention"
                        if target_status == "PAUSED_REVIEW"
                        else None
                    ),
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == attempt.attempt_number,
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner == step.lease_owner,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                )
                .values(
                    status=target_step_status,
                    active_artifact_id=(
                        (promoted_artifact or artifact_row).id
                        if make_active
                        else None
                    ),
                    protocol_failure_count=(
                        0
                        if workflow.generation_version == 2
                        and step.kind == "WRITING"
                        else step.protocol_failure_count
                    ),
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            attempt_claim = self.session.execute(
                update(ModelAttempt)
                .where(
                    ModelAttempt.id == attempt.id,
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.attempt_number == step.attempt_count,
                    ModelAttempt.status == "RUNNING",
                )
                .values(
                    status="COMPLETED",
                    provider_response_id=response.provider_response_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    error_code=None,
                    error_detail=None,
                )
            )
            if (
                workflow_claim.rowcount != 1
                or step_claim.rowcount != 1
                or attempt_claim.rowcount != 1
            ):
                raise _AttemptCompletionConflict
            self.session.commit()
            return artifact_row
        except _AttemptCompletionConflict:
            self.session.rollback()
            if self._pause_stale_attempt_completion(attempt_id, response):
                raise StaleOutlineCompletion(
                    "official outline changed during attempt completion"
                )
            raise ValueError("attempt completion conflict") from None
        except Exception:
            self.session.rollback()
            raise

    def _pause_stale_attempt_completion(
        self, attempt_id: str, response: ModelResponse
    ) -> bool:
        self.session.expire_all()
        attempt = self.session.get(ModelAttempt, attempt_id)
        step = (
            self.session.get(WorkflowStep, attempt.step_id)
            if attempt is not None
            else None
        )
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        project = (
            self.session.get(NovelProject, workflow.project_id)
            if workflow is not None
            else None
        )
        now = self._aware_utc(self.clock.now())
        if (
            attempt is None
            or step is None
            or workflow is None
            or project is None
            or project.active_workflow_id != workflow.id
            or self._inputs_are_current(workflow, project)
            or attempt.status != "RUNNING"
            or attempt.attempt_number != step.attempt_count
            or step.status != "RUNNING"
            or step.active_artifact_id is not None
            or step.lease_owner is None
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
            or workflow.current_position != step.position
        ):
            self.session.rollback()
            return False
        self._require_transition(workflow.status, "PAUSED_STALE_VERSION")
        try:
            stale_project = (
                select(NovelProject.id)
                .where(
                    NovelProject.id == workflow.project_id,
                    NovelProject.active_workflow_id == workflow.id,
                    or_(NovelProject.official_outline_version_id != workflow.base_outline_version_id,
                        ~self._stage_current_predicate(workflow.id)),
                )
                .exists()
            )
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                    stale_project,
                )
                .values(
                    status="PAUSED_STALE_VERSION",
                    actual_input_tokens=GenerationWorkflow.actual_input_tokens
                    + (response.input_tokens or 0),
                    actual_output_tokens=GenerationWorkflow.actual_output_tokens
                    + (response.output_tokens or 0),
                    revision=workflow.revision + 1,
                    last_error_code="stale_outline",
                    last_error_detail=(
                        "the official outline changed during the provider call"
                    ),
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == attempt.attempt_number,
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner == step.lease_owner,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                    WorkflowStep.lease_expires_at > now,
                )
                .values(
                    status="PAUSED",
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            attempt_claim = self.session.execute(
                update(ModelAttempt)
                .where(
                    ModelAttempt.id == attempt.id,
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.attempt_number == step.attempt_count,
                    ModelAttempt.status == "RUNNING",
                )
                .values(
                    status="FAILED",
                    provider_response_id=response.provider_response_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    error_code="stale_outline",
                    error_detail="official outline changed during provider call",
                )
            )
            if (
                workflow_claim.rowcount != 1
                or step_claim.rowcount != 1
                or attempt_claim.rowcount != 1
            ):
                self.session.rollback()
                return False
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "workflow_paused",
                "system",
                {"reason": "stale_outline", "during": "attempt_completion"},
            )
            self.session.commit()
            return True
        except Exception:
            self.session.rollback()
            raise

    def fail_attempt(
        self,
        attempt_id: str,
        error: ProviderError,
        *,
        response: ModelResponse | None = None,
    ) -> GenerationWorkflow:
        if not isinstance(error, ProviderError):
            raise TypeError("attempt failures must be typed ProviderError instances")
        if response is not None:
            self._validate_response(response)
        code, detail, retryable = self._provider_failure(error)
        now = self._aware_utc(self.clock.now())
        self.session.expire_all()
        attempt = self.session.get(ModelAttempt, attempt_id)
        if attempt is None:
            self.session.rollback()
            raise ValueError("model attempt not found")
        step = self.session.get(WorkflowStep, attempt.step_id)
        workflow = (
            self.session.get(GenerationWorkflow, step.workflow_id)
            if step is not None
            else None
        )
        if (
            step is None
            or workflow is None
            or attempt.status != "RUNNING"
            or attempt.attempt_number != step.attempt_count
            or step.status != "RUNNING"
            or step.lease_owner is None
            or step.lease_expires_at is None
            or step.lease_expires_at <= now
            or workflow.current_position != step.position
        ):
            self.session.rollback()
            raise ValueError("attempt failure conflict")

        protocol_failure_count = (
            step.protocol_failure_count + 1
            if workflow.generation_version == 2 and step.kind == "WRITING"
            else step.attempt_count
        )
        if retryable and protocol_failure_count < MAX_STEP_ATTEMPTS:
            target_status = workflow.status
            target_step_status = "PENDING"
        elif retryable:
            target_status = "PAUSED_ATTEMPTS"
            target_step_status = "PAUSED"
        else:
            target_status = "PAUSED_PROVIDER"
            target_step_status = "PAUSED"
        self._require_transition(workflow.status, target_status)
        try:
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                )
                .values(
                    status=target_status,
                    actual_input_tokens=GenerationWorkflow.actual_input_tokens
                    + ((response.input_tokens or 0) if response is not None else 0),
                    actual_output_tokens=GenerationWorkflow.actual_output_tokens
                    + ((response.output_tokens or 0) if response is not None else 0),
                    revision=workflow.revision + 1,
                    last_error_code=code,
                    last_error_detail=detail,
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.status == "RUNNING",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == attempt.attempt_number,
                    WorkflowStep.lease_owner == step.lease_owner,
                    WorkflowStep.lease_expires_at == step.lease_expires_at,
                    WorkflowStep.lease_expires_at > now,
                )
                .values(
                    status=target_step_status,
                    protocol_failure_count=(
                        protocol_failure_count
                        if workflow.generation_version == 2
                        and step.kind == "WRITING"
                        else step.protocol_failure_count
                    ),
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            attempt_claim = self.session.execute(
                update(ModelAttempt)
                .where(
                    ModelAttempt.id == attempt.id,
                    ModelAttempt.step_id == step.id,
                    ModelAttempt.attempt_number == step.attempt_count,
                    ModelAttempt.status == "RUNNING",
                )
                .values(
                    status="FAILED",
                    provider_response_id=(
                        response.provider_response_id if response is not None else None
                    ),
                    input_tokens=(response.input_tokens if response is not None else None),
                    output_tokens=(response.output_tokens if response is not None else None),
                    latency_ms=(response.latency_ms if response is not None else None),
                    error_code=code,
                    error_detail=detail,
                )
            )
            if (
                workflow_claim.rowcount != 1
                or step_claim.rowcount != 1
                or attempt_claim.rowcount != 1
            ):
                raise ValueError("attempt failure conflict")
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def approve_plan(self, workflow_id: str, actor: str) -> GenerationWorkflow:
        normalized_actor = actor.strip() if isinstance(actor, str) else ""
        if not normalized_actor:
            raise ValueError("plan decision actor is required")
        self.session.expire_all()
        workflow = self._get_workflow(workflow_id)
        if workflow.status != "AWAITING_PLAN_APPROVAL":
            self.session.rollback()
            raise ValueError("plan approval conflict")
        project = self.session.get(NovelProject, workflow.project_id)
        planning_step = self.session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.kind == "PLANNING",
                WorkflowStep.ordinal.is_(None),
            )
        )
        plan_artifact = (
            self.session.get(WorkflowArtifact, planning_step.active_artifact_id)
            if planning_step is not None
            and planning_step.active_artifact_id is not None
            else None
        )
        if (
            planning_step is None
            or planning_step.status != "COMPLETED"
            or plan_artifact is None
            or plan_artifact.workflow_id != workflow.id
            or plan_artifact.step_id != planning_step.id
            or plan_artifact.kind != "batch_plan"
        ):
            self.session.rollback()
            raise ValueError("plan approval requires an active plan artifact")
        self._validate_plan_payload(plan_artifact.payload, workflow.requested_chapters)
        if (
            project is None
            or project.active_workflow_id != workflow.id
            or not self._inputs_are_current(workflow, project)
        ):
            self.session.rollback()
            raise ValueError("plan approval conflict")
        if self.session.scalar(
            select(func.count())
            .select_from(WorkflowStep)
            .where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.position > 0,
            )
        ):
            self.session.rollback()
            raise ValueError("plan approval conflict")

        self._require_transition(workflow.status, "GENERATING_CHAPTERS")
        try:
            claimed = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == "AWAITING_PLAN_APPROVAL",
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == planning_step.position,
                )
                .values(
                    status="GENERATING_CHAPTERS",
                    current_position=1,
                    revision=workflow.revision + 1,
                    last_error_code=None,
                    last_error_detail=None,
                )
            )
            if claimed.rowcount != 1:
                raise ValueError("plan approval conflict")
            steps: list[WorkflowStep] = []
            position = 1
            for ordinal in range(1, workflow.requested_chapters + 1):
                steps.append(
                    self._new_step(workflow.id, "WRITING", ordinal, position)
                )
                position += 1
                if workflow.generation_version == 2:
                    steps.append(
                        self._new_step(
                            workflow.id, "VALIDATING_CHAPTER", ordinal, position
                        )
                    )
                    position += 1
                steps.append(
                    self._new_step(workflow.id, "SUMMARIZING", ordinal, position)
                )
                position += 1
            steps.append(self._new_step(workflow.id, "REVIEWING", None, position))
            position += 1
            steps.append(
                self._new_step(
                    workflow.id, "CREATING_CANDIDATE_BATCH", None, position
                )
            )
            self.session.add_all(steps)
            self.session.add(
                PlanDecision(
                    id=str(uuid4()),
                    workflow_id=workflow.id,
                    decision="approved",
                    reason="",
                    actor=normalized_actor,
                )
            )
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "plan_approved",
                normalized_actor,
                {
                    "plan_artifact_id": plan_artifact.id,
                    "requested_chapters": workflow.requested_chapters,
                },
            )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def reject_plan(
        self, workflow_id: str, reason: str, actor: str
    ) -> GenerationWorkflow:
        normalized_reason = reason.strip() if isinstance(reason, str) else ""
        normalized_actor = actor.strip() if isinstance(actor, str) else ""
        if not normalized_reason:
            raise ValueError("rejection reason is required")
        if not normalized_actor:
            raise ValueError("plan decision actor is required")
        self.session.expire_all()
        workflow = self._get_workflow(workflow_id)
        if workflow.status != "AWAITING_PLAN_APPROVAL":
            self.session.rollback()
            raise ValueError("plan rejection conflict")
        self._require_transition(workflow.status, "REJECTED")
        try:
            claimed = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == "AWAITING_PLAN_APPROVAL",
                    GenerationWorkflow.revision == workflow.revision,
                )
                .values(
                    status="REJECTED",
                    revision=workflow.revision + 1,
                    last_error_code=None,
                    last_error_detail=None,
                )
            )
            if claimed.rowcount != 1:
                raise ValueError("plan rejection conflict")
            self.session.execute(
                update(NovelProject)
                .where(
                    NovelProject.id == workflow.project_id,
                    NovelProject.active_workflow_id == workflow.id,
                )
                .values(active_workflow_id=None)
            )
            self.session.add(
                PlanDecision(
                    id=str(uuid4()),
                    workflow_id=workflow.id,
                    decision="rejected",
                    reason=normalized_reason,
                    actor=normalized_actor,
                )
            )
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "plan_rejected",
                normalized_actor,
                {"reason": normalized_reason},
            )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def can_resume(self, workflow_id: str) -> bool:
        with self.session.no_autoflush:
            workflow = self.session.get(GenerationWorkflow, workflow_id)
            if workflow is None:
                return False
            try:
                self._resumable_step(workflow)
            except ValueError:
                return False
            return True

    def resume(self, workflow_id: str) -> GenerationWorkflow:
        self.session.expire_all()
        workflow = self._get_workflow(workflow_id)
        source_status = workflow.status
        try:
            step = self._resumable_step(workflow)
        except ValueError:
            self.session.rollback()
            raise
        target_status = self._workflow_status_for_step(step.kind)
        self._require_transition(workflow.status, target_status)
        try:
            resumable_project = (
                select(NovelProject.id)
                .where(
                    NovelProject.id == workflow.project_id,
                    NovelProject.active_workflow_id == workflow.id,
                    NovelProject.official_outline_version_id
                    == workflow.base_outline_version_id,
                )
                .exists()
            )
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == step.position,
                    resumable_project,
                )
                .values(
                    status=target_status,
                    revision=workflow.revision + 1,
                    last_error_code=None,
                    last_error_detail=None,
                )
            )
            step_claim = self.session.execute(
                update(WorkflowStep)
                .where(
                    WorkflowStep.id == step.id,
                    WorkflowStep.status == "PAUSED",
                    WorkflowStep.revision == step.revision,
                    WorkflowStep.attempt_count == step.attempt_count,
                    (
                        WorkflowStep.protocol_failure_count < MAX_STEP_ATTEMPTS
                        if workflow.generation_version == 2
                        and step.kind == "WRITING"
                        else WorkflowStep.attempt_count < MAX_STEP_ATTEMPTS
                    ),
                    WorkflowStep.active_artifact_id.is_(None),
                    WorkflowStep.lease_owner.is_(None),
                    WorkflowStep.lease_expires_at.is_(None),
                )
                .values(
                    status="PENDING",
                    lease_owner=None,
                    lease_expires_at=None,
                    revision=step.revision + 1,
                )
            )
            if workflow_claim.rowcount != 1 or step_claim.rowcount != 1:
                raise ValueError("workflow resume conflict")
            self._add_audit(
                workflow.project_id,
                workflow.id,
                "workflow_resumed",
                "author",
                {"from_status": source_status, "to_status": target_status},
            )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def _resumable_step(self, workflow: GenerationWorkflow) -> WorkflowStep:
        if workflow.status not in RESUMABLE_WORKFLOW_STATUSES:
            raise ValueError("workflow status is not resumable")
        project = self.session.get(NovelProject, workflow.project_id)
        step = self.session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.position == workflow.current_position,
            )
        )
        if (
            project is None
            or project.active_workflow_id != workflow.id
            or not self._inputs_are_current(workflow, project)
            or step is None
            or step.status != "PAUSED"
            or step.lease_owner is not None
            or step.lease_expires_at is not None
            or step.active_artifact_id is not None
            or (
                step.protocol_failure_count >= MAX_STEP_ATTEMPTS
                if workflow.generation_version == 2 and step.kind == "WRITING"
                else step.attempt_count >= MAX_STEP_ATTEMPTS
            )
        ):
            raise ValueError("workflow cannot be resumed")
        return step

    def reconcile_batch_decision(self, workflow_id: str) -> GenerationWorkflow:
        self.session.expire_all()
        workflow = self._get_workflow(workflow_id)
        batch = None
        if workflow.candidate_batch_id is not None:
            batch = self.session.get(WritingBatch, workflow.candidate_batch_id)
        if batch is None:
            batch = self.session.scalar(
                select(WritingBatch).where(
                    WritingBatch.source_workflow_id == workflow.id
                )
            )
        if batch is None:
            self.session.rollback()
            return workflow
        if (
            batch.project_id != workflow.project_id
            or batch.source_workflow_id != workflow.id
        ):
            self.session.rollback()
            raise ValueError("candidate batch does not belong to workflow")

        if batch.status == "approved":
            target_status = "COMPLETED"
            action = "workflow_completed"
        elif batch.status == "rejected":
            target_status = "REJECTED"
            action = "workflow_rejected"
        elif batch.status == "ready_for_review":
            target_status = "AWAITING_CONTENT_APPROVAL"
            action = "candidate_batch_linked"
        elif batch.status == "draft":
            target_status = "CREATING_CANDIDATE_BATCH"
            action = "candidate_batch_linked"
        else:
            self.session.rollback()
            return workflow

        workflow_already_reconciled = (
            workflow.status == target_status
            and workflow.candidate_batch_id == batch.id
        )
        candidate_step = self.session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.position == workflow.current_position,
                WorkflowStep.kind == "CREATING_CANDIDATE_BATCH",
            )
        )
        if (
            batch.status == "ready_for_review"
            and workflow_already_reconciled
            and candidate_step is not None
            and candidate_step.status == "COMPLETED"
            and candidate_step.active_artifact_id is None
            and candidate_step.lease_owner is None
            and candidate_step.lease_expires_at is None
        ):
            self.session.rollback()
            return workflow
        if workflow.status in TERMINAL_WORKFLOW_STATUSES:
            if workflow_already_reconciled:
                self.session.rollback()
                return workflow
            self.session.rollback()
            raise ValueError("terminal workflow conflicts with batch decision")
        paused_candidate_decision = (
            workflow.status == "PAUSED_STALE_VERSION"
            and batch.status in {"approved", "rejected"}
            and candidate_step is not None
            and candidate_step.status == "PAUSED"
            and candidate_step.active_artifact_id is None
            and candidate_step.lease_owner is None
            and candidate_step.lease_expires_at is None
        )
        # Only a persisted Phase 1 decision may close a stale candidate pause.
        # This does not make paused workflows resumable or bypass either gate.
        if not paused_candidate_decision:
            self._require_transition(workflow.status, target_status)
        should_update_step = (
            batch.status != "draft"
            and candidate_step is not None
            and (candidate_step.status in {"PENDING", "RUNNING"} or paused_candidate_decision)
        )
        if workflow_already_reconciled and not should_update_step:
            self.session.rollback()
            return workflow
        try:
            workflow_claim = self.session.execute(
                update(GenerationWorkflow)
                .where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.status == workflow.status,
                    GenerationWorkflow.revision == workflow.revision,
                    GenerationWorkflow.current_position == workflow.current_position,
                    select(WritingBatch.id).where(
                        WritingBatch.id == batch.id,
                        WritingBatch.project_id == workflow.project_id,
                        WritingBatch.source_workflow_id == workflow.id,
                        WritingBatch.status == batch.status,
                    ).exists(),
                )
                .values(
                    status=target_status,
                    candidate_batch_id=batch.id,
                    revision=workflow.revision + 1,
                    last_error_code=None,
                    last_error_detail=None,
                )
            )
            if workflow_claim.rowcount != 1:
                raise ValueError("batch reconciliation conflict")
            if should_update_step:
                assert candidate_step is not None
                step_predicates = [
                    WorkflowStep.id == candidate_step.id,
                    WorkflowStep.workflow_id == workflow.id,
                    WorkflowStep.status == candidate_step.status,
                    WorkflowStep.revision == candidate_step.revision,
                    WorkflowStep.active_artifact_id.is_(None),
                ]
                if candidate_step.status == "RUNNING":
                    step_predicates.extend(
                        [
                            WorkflowStep.lease_owner == candidate_step.lease_owner,
                            WorkflowStep.lease_expires_at
                            == candidate_step.lease_expires_at,
                        ]
                    )
                else:
                    step_predicates.extend(
                        [
                            WorkflowStep.lease_owner.is_(None),
                            WorkflowStep.lease_expires_at.is_(None),
                        ]
                    )
                step_claim = self.session.execute(
                    update(WorkflowStep)
                    .where(*step_predicates)
                    .values(
                        status="COMPLETED",
                        lease_owner=None,
                        lease_expires_at=None,
                        revision=candidate_step.revision + 1,
                    )
                )
                if step_claim.rowcount != 1:
                    raise ValueError("batch reconciliation conflict")
            if target_status in {"COMPLETED", "REJECTED"}:
                self.session.execute(
                    update(NovelProject)
                    .where(
                        NovelProject.id == workflow.project_id,
                        NovelProject.active_workflow_id == workflow.id,
                    )
                    .values(active_workflow_id=None)
                )
            if not workflow_already_reconciled:
                self._add_audit(
                    workflow.project_id,
                    workflow.id,
                    action,
                    "system",
                    {"candidate_batch_id": batch.id, "batch_status": batch.status},
                )
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    def can_cancel(self, workflow_id: str) -> bool:
        with self.session.no_autoflush:
            workflow = self.session.get(GenerationWorkflow, workflow_id)
            if workflow is None:
                return False
            return self.session.scalar(
                select(GenerationWorkflow.id).where(
                    GenerationWorkflow.id == workflow.id,
                    *self._cancel_predicates(workflow),
                )
            ) is not None

    @staticmethod
    def _cancel_predicates(workflow: GenerationWorkflow) -> tuple:
        return (
            GenerationWorkflow.status.in_(
                [status for status in WORKFLOW_STATUSES if status.startswith("PAUSED_")]
            ),
            GenerationWorkflow.candidate_batch_id.is_(None),
            ~select(WritingBatch.id).where(
                WritingBatch.source_workflow_id == workflow.id,
            ).exists(),
            select(NovelProject.id).where(
                NovelProject.id == workflow.project_id,
                NovelProject.active_workflow_id == workflow.id,
                NovelProject.active_batch_id.is_(None),
            ).exists(),
        )

    def cancel(self, workflow_id: str, actor: str) -> GenerationWorkflow:
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("cancel actor is required")
        self.session.expire_all()
        workflow = self._get_workflow(workflow_id)
        try:
            cancelled = self.session.execute(
                update(GenerationWorkflow).where(
                    GenerationWorkflow.id == workflow.id,
                    GenerationWorkflow.revision == workflow.revision,
                    *self._cancel_predicates(workflow),
                ).values(status="CANCELLED", revision=workflow.revision + 1)
            )
            if cancelled.rowcount != 1:
                raise ValueError("cancel conflict: owner changed or candidate requires batch decision")
            released = self.session.execute(
                update(NovelProject).where(
                    NovelProject.id == workflow.project_id,
                    NovelProject.active_workflow_id == workflow.id,
                    NovelProject.active_batch_id.is_(None),
                ).values(active_workflow_id=None)
            )
            if released.rowcount != 1:
                raise ValueError("cancel owner conflict")
            self._add_audit(workflow.project_id, workflow.id, "workflow_cancelled", actor.strip(), {})
            self.session.commit()
            return workflow
        except Exception:
            self.session.rollback()
            raise

    @staticmethod
    def _stage_current_predicate(workflow_id):
        mapped = select(StageWorkflow.workflow_id).where(StageWorkflow.workflow_id == workflow_id).exists()
        valid = (select(StageWorkflow.workflow_id)
                 .join(StoryStage, StoryStage.id == StageWorkflow.stage_id)
                 .join(StageRoadmapVersion, StageRoadmapVersion.id == StageWorkflow.roadmap_id)
                 .join(NovelProject, NovelProject.id == StoryStage.project_id)
                 .where(StageWorkflow.workflow_id == workflow_id,
                        StoryStage.approved_roadmap_id == StageWorkflow.roadmap_id,
                        StoryStage.confirmed_chapters == StageWorkflow.confirmed_start,
                        StageRoadmapVersion.status == "APPROVED",
                        NovelProject.current_constitution_version_id == StageRoadmapVersion.constitution_version_id)
                 .exists())
        return or_(~mapped, valid)

    def _inputs_are_current(self, workflow, project):
        return (project.official_outline_version_id == workflow.base_outline_version_id
                and bool(self.session.scalar(select(self._stage_current_predicate(workflow.id)))))

    def _read_valid_start_state(self, project_id: str) -> tuple[str, str]:
        self.session.expire_all()
        project = self.session.get(NovelProject, project_id)
        if project is None:
            raise ValueError("project not found")
        if project.active_batch_id is not None:
            raise ValueError("project already has an active batch")
        if project.active_workflow_id is not None:
            raise ValueError("project already has an active workflow")
        if project.current_constitution_version_id is None:
            raise ValueError("project requires an approved constitution")
        constitution = self.session.scalar(
            select(ConstitutionVersion).where(
                ConstitutionVersion.id == project.current_constitution_version_id,
                ConstitutionVersion.project_id == project.id,
                ConstitutionVersion.author_approved.is_(True),
            )
        )
        if constitution is None:
            raise ValueError("project requires an approved constitution")
        if project.official_outline_version_id is None:
            raise ValueError("project requires a current official outline")
        outline = self.session.scalar(
            select(OutlineVersion).where(
                OutlineVersion.id == project.official_outline_version_id,
                OutlineVersion.project_id == project.id,
                OutlineVersion.status == "official",
            )
        )
        if outline is None:
            raise ValueError("project requires a current official outline")
        return constitution.id, outline.id

    def _raise_start_conflict(self, project_id: str) -> None:
        self.session.expire_all()
        project = self.session.get(NovelProject, project_id)
        if project is None:
            self.session.rollback()
            raise ValueError("project not found")
        if project.active_workflow_id is not None:
            self.session.rollback()
            raise ValueError("project already has an active workflow")
        if project.active_batch_id is not None:
            self.session.rollback()
            raise ValueError("project already has an active batch")
        self.session.rollback()
        raise ValueError("project state changed before workflow ownership")

    @staticmethod
    def _validate_start_arguments(
        provider_name: str,
        model_name: str,
        requested_chapters: int,
        budgets: WorkflowBudgets,
    ) -> None:
        if provider_name not in PROVIDER_NAMES:
            raise ValueError("unknown provider")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model name is required")
        if type(requested_chapters) is not int or not 1 <= requested_chapters <= 5:
            raise ValueError("requested chapters must be between 1 and 5")
        if not isinstance(budgets, WorkflowBudgets) or any(
            type(getattr(budgets, field.name)) is not int
            or getattr(budgets, field.name) <= 0
            for field in fields(WorkflowBudgets)
        ):
            raise ValueError("workflow budgets must be positive integers")

    @staticmethod
    def _prompt_parameters(
        budgets: WorkflowBudgets,
    ) -> dict[str, dict[str, object]]:
        return {
            "batch_planner": {
                "max_input_tokens": budgets.planner_input,
                "max_output_tokens": budgets.planner_output,
            },
            "chapter_writer": {
                "max_input_tokens": budgets.writer_input,
                "max_output_tokens": budgets.writer_output,
            },
            "chapter_summarizer": {
                "max_input_tokens": budgets.summarizer_input,
                "max_output_tokens": budgets.summarizer_output,
            },
            "batch_reviewer": {
                "max_input_tokens": budgets.reviewer_input,
                "max_output_tokens": budgets.reviewer_output,
            },
        }

    @classmethod
    def _v2_prompt_parameters(
        cls, budgets: WorkflowBudgets
    ) -> dict[str, dict[str, object]]:
        parameters = cls._prompt_parameters(budgets)
        parameters["chapter_coverage_reviewer"] = {
            "max_input_tokens": budgets.reviewer_input,
            "max_output_tokens": budgets.reviewer_output,
        }
        return parameters

    def _pause_stale_workflow(
        self, workflow: GenerationWorkflow, step: WorkflowStep
    ) -> None:
        self._require_transition(workflow.status, "PAUSED_STALE_VERSION")
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
                GenerationWorkflow.current_position == step.position,
            )
            .values(
                status="PAUSED_STALE_VERSION",
                revision=workflow.revision + 1,
                last_error_code="stale_outline",
                last_error_detail="the official outline changed after workflow start",
            )
        )
        step_claim = self.session.execute(
            update(WorkflowStep)
            .where(
                WorkflowStep.id == step.id,
                WorkflowStep.status == step.status,
                WorkflowStep.revision == step.revision,
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
            return
        self._add_audit(
            workflow.project_id,
            workflow.id,
            "workflow_paused",
            "system",
            {"reason": "stale_outline"},
        )
        self.session.commit()

    def _pause_attempt_exhaustion(
        self, workflow: GenerationWorkflow, step: WorkflowStep
    ) -> None:
        self._require_transition(workflow.status, "PAUSED_ATTEMPTS")
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
            )
            .values(
                status="PAUSED_ATTEMPTS",
                revision=workflow.revision + 1,
                last_error_code="attempts_exhausted",
                last_error_detail="maximum model attempts reached",
            )
        )
        step_claim = self.session.execute(
            update(WorkflowStep)
            .where(
                WorkflowStep.id == step.id,
                WorkflowStep.status == "RUNNING",
                WorkflowStep.revision == step.revision,
                WorkflowStep.attempt_count == step.attempt_count,
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
            raise ValueError("maximum attempt pause conflict")
        self.session.commit()

    def _pause_workflow_budget_exhaustion(
        self, workflow: GenerationWorkflow, step: WorkflowStep
    ) -> None:
        self._require_transition(workflow.status, "PAUSED_ATTEMPTS")
        workflow_claim = self.session.execute(
            update(GenerationWorkflow)
            .where(
                GenerationWorkflow.id == workflow.id,
                GenerationWorkflow.status == workflow.status,
                GenerationWorkflow.revision == workflow.revision,
                GenerationWorkflow.current_position == step.position,
            )
            .values(
                status="PAUSED_ATTEMPTS",
                revision=workflow.revision + 1,
                last_error_code="workflow_budget_exhausted",
                last_error_detail="workflow model call or token budget exhausted",
            )
        )
        step_claim = self.session.execute(
            update(WorkflowStep)
            .where(
                WorkflowStep.id == step.id,
                WorkflowStep.status == "RUNNING",
                WorkflowStep.revision == step.revision,
                WorkflowStep.attempt_count == step.attempt_count,
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
            raise ValueError("workflow budget pause conflict")
        self.session.commit()

    @staticmethod
    def _step_token_budget(
        workflow: GenerationWorkflow, step: WorkflowStep
    ) -> tuple[int, int]:
        if step.kind == "PLANNING":
            return workflow.planner_input_tokens, workflow.planner_output_tokens
        if step.kind == "WRITING":
            return workflow.writer_input_tokens, workflow.writer_output_tokens
        if step.kind == "SUMMARIZING":
            return workflow.summarizer_input_tokens, workflow.summarizer_output_tokens
        if step.kind in {"VALIDATING_CHAPTER", "REVIEWING"}:
            return workflow.reviewer_input_tokens, workflow.reviewer_output_tokens
        raise ValueError("workflow step has no model token budget")

    def _completion_transition(
        self,
        workflow: GenerationWorkflow,
        step: WorkflowStep,
        artifact: _ArtifactValues,
        finalize_step: bool,
    ) -> tuple[str, int, str, bool]:
        if not finalize_step:
            return workflow.status, step.position, "PENDING", False
        if step.kind == "REVIEWING" and (
            artifact.payload.get("passed") is not True
            or bool(artifact.payload.get("evidence_queries"))
        ):
            return "PAUSED_REVIEW", step.position, "PAUSED", True
        if step.kind == "PLANNING":
            return "AWAITING_PLAN_APPROVAL", step.position, "COMPLETED", True
        next_step = self.session.scalar(
            select(WorkflowStep).where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.position == step.position + 1,
            )
        )
        if next_step is None:
            raise ValueError("workflow step sequence is incomplete")
        expected_after_writing = (
            "VALIDATING_CHAPTER"
            if workflow.generation_version == 2
            else "SUMMARIZING"
        )
        if step.kind == "WRITING" and (
            next_step.kind != expected_after_writing
            or next_step.ordinal != step.ordinal
        ):
            raise ValueError("workflow step sequence is invalid")
        if step.kind == "WRITING":
            return "GENERATING_CHAPTERS", next_step.position, "COMPLETED", True
        if step.kind == "VALIDATING_CHAPTER":
            if next_step.kind != "SUMMARIZING" or next_step.ordinal != step.ordinal:
                raise ValueError("workflow step sequence is invalid")
            if not coverage_is_valid(self.session, step, artifact.payload):
                return "PAUSED_REVIEW", step.position, "PAUSED", True
            return "GENERATING_CHAPTERS", next_step.position, "COMPLETED", True
        if step.kind == "SUMMARIZING" and next_step.kind == "WRITING":
            return "GENERATING_CHAPTERS", next_step.position, "COMPLETED", True
        if step.kind == "SUMMARIZING" and next_step.kind == "REVIEWING":
            return "REVIEWING_BATCH", next_step.position, "COMPLETED", True
        if step.kind == "REVIEWING" and next_step.kind == "CREATING_CANDIDATE_BATCH":
            return "CREATING_CANDIDATE_BATCH", next_step.position, "COMPLETED", True
        raise ValueError("workflow step sequence is invalid")

    def _artifact_values(
        self,
        workflow: GenerationWorkflow,
        step: WorkflowStep,
        artifact: BaseModel | Mapping[str, object],
    ) -> _ArtifactValues:
        expected_kind = _EXPECTED_ARTIFACT_KINDS.get(step.kind)
        if workflow.generation_version == 2 and step.kind == "WRITING":
            expected_kind = "chapter_work_draft"
        if expected_kind is None:
            raise ValueError("workflow step does not accept model artifacts")
        if isinstance(artifact, BaseModel):
            raw_payload: object = artifact.model_dump(mode="json")
            kind: object = expected_kind
            ordinal: object = step.ordinal
            text_content: object = None
            visible_char_count: object = None
        elif isinstance(artifact, Mapping):
            if "kind" in artifact or "payload" in artifact:
                kind = artifact.get("kind")
                raw_payload = artifact.get("payload")
                ordinal = artifact.get("ordinal", step.ordinal)
                text_content = artifact.get("text_content")
                visible_char_count = artifact.get("visible_char_count")
            else:
                kind = expected_kind
                raw_payload = artifact
                ordinal = step.ordinal
                text_content = None
                visible_char_count = None
        else:
            raise TypeError("artifact must be a Pydantic model or mapping")
        if kind != expected_kind:
            raise ValueError("artifact kind does not match workflow step")
        if not isinstance(raw_payload, Mapping):
            raise ValueError("artifact payload must be a mapping")
        if ordinal != step.ordinal:
            raise ValueError("artifact ordinal does not match workflow step")
        payload = deepcopy(dict(raw_payload))
        try:
            schemas = (
                V2_AGENT_SCHEMAS
                if workflow.generation_version == 2
                else AGENT_SCHEMAS
            )
            payload = schemas[_STEP_PROMPT_ROLES[step.kind]].model_validate(
                payload
            ).model_dump(mode="json")
        except (KeyError, ValueError, TypeError):
            raise ValueError("artifact payload failed validation") from None
        if step.kind == "WRITING":
            canonical_text = payload["body"]
            canonical_visible_count = count_visible_characters(canonical_text)
        elif step.kind == "SUMMARIZING":
            canonical_text = payload["summary"]
            canonical_visible_count = None
        else:
            canonical_text = None
            canonical_visible_count = None
        if (
            text_content is not None
            and text_content != canonical_text
            or visible_char_count is not None
            and visible_char_count != canonical_visible_count
        ):
            raise ValueError("artifact metadata does not match payload")
        text_content = canonical_text
        visible_char_count = canonical_visible_count
        if text_content is not None:
            content_hash = sha256(text_content.encode("utf-8")).hexdigest()
        else:
            canonical = json.dumps(
                {
                    "kind": kind,
                    "ordinal": ordinal,
                    "text_content": text_content,
                    "payload": payload,
                    "visible_char_count": visible_char_count,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            content_hash = sha256(canonical.encode("utf-8")).hexdigest()
        return _ArtifactValues(
            kind=expected_kind,
            ordinal=step.ordinal,
            text_content=text_content,
            payload=payload,
            visible_char_count=visible_char_count,
            content_hash=content_hash,
        )

    @staticmethod
    def _validate_response(response: ModelResponse) -> None:
        if not isinstance(response, ModelResponse):
            raise TypeError("response must be a ModelResponse")
        for value in (response.input_tokens, response.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("response token usage is invalid")
        if type(response.latency_ms) is not int or response.latency_ms < 0:
            raise ValueError("response latency is invalid")

    @staticmethod
    def _provider_failure(error: ProviderError) -> tuple[str, str, bool]:
        if isinstance(error, ResponseFailure):
            return "provider_protocol", safe_failure_detail(error), True
        for error_type, code, detail, retryable in _SAFE_PROVIDER_FAILURES:
            if isinstance(error, error_type):
                return code, detail, retryable
        return "provider_error", "provider request failed", False

    @staticmethod
    def _validate_plan_payload(payload: Mapping[str, Any], expected_count: int) -> None:
        chapters = payload.get("chapters")
        if not isinstance(chapters, list) or len(chapters) != expected_count:
            raise ValueError("active plan artifact has invalid chapter count")
        if [chapter.get("ordinal") for chapter in chapters if isinstance(chapter, dict)] != list(
            range(1, expected_count + 1)
        ):
            raise ValueError("active plan artifact has invalid ordinals")

    @staticmethod
    def _workflow_status_for_step(kind: str) -> str:
        if kind == "PLANNING":
            return "PLANNING"
        if kind in {"WRITING", "VALIDATING_CHAPTER", "SUMMARIZING"}:
            return "GENERATING_CHAPTERS"
        if kind == "REVIEWING":
            return "REVIEWING_BATCH"
        if kind == "CREATING_CANDIDATE_BATCH":
            return "CREATING_CANDIDATE_BATCH"
        raise ValueError("unknown workflow step kind")

    @staticmethod
    def _new_step(
        workflow_id: str, kind: str, ordinal: int | None, position: int
    ) -> WorkflowStep:
        return WorkflowStep(
            id=str(uuid4()),
            workflow_id=workflow_id,
            kind=kind,
            ordinal=ordinal,
            position=position,
            status="PENDING",
            attempt_count=0,
            revision=1,
        )

    @staticmethod
    def _require_transition(source: str, target: str) -> None:
        if target not in _ALLOWED_WORKFLOW_TRANSITIONS.get(source, frozenset()):
            raise ValueError(f"workflow transition is not allowed: {source} -> {target}")

    def _get_workflow(self, workflow_id: str) -> GenerationWorkflow:
        workflow = self.session.get(GenerationWorkflow, workflow_id)
        if workflow is None:
            self.session.rollback()
            raise ValueError("workflow not found")
        return workflow

    def _add_audit(
        self,
        project_id: str,
        workflow_id: str,
        action: str,
        actor: str,
        details: dict[str, object],
    ) -> None:
        self.session.add(
            AuditEvent(
                id=str(uuid4()),
                project_id=project_id,
                entity_type="generation_workflow",
                entity_id=workflow_id,
                action=action,
                actor=actor,
                details=deepcopy(details),
            )
        )

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("workflow clock requires a timezone-aware datetime")
        return value.astimezone(timezone.utc)

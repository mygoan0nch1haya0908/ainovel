from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ainovel.agents.contracts import (
    BatchPlanDraft,
    BatchReview,
    ChapterDraft,
    ChapterPlan,
    ChapterSummaryDelta,
)
from ainovel.agents.runner import AgentRunner
from ainovel.context import RequiredContextOverflow, effective_input_capacity
from ainovel.models.batch import Chapter, WritingBatch
from ainovel.models.context import ContextPacket, ContextPacketItem, ContextSource
from ainovel.models.outline import OutlineNode
from ainovel.models.project import ConstitutionVersion, NovelProject
from ainovel.models.prompt import WorkflowPromptSnapshot
from ainovel.models.workflow import (
    GenerationWorkflow,
    PlanDecision,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.providers.contracts import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderProtocolError,
    ProviderUnavailable,
)
from ainovel.providers.registry import ProviderRegistry
from ainovel.services.batches import BatchService
from ainovel.services.context import (
    EXCERPT_SOURCE_TYPES,
    MAX_L7_VISIBLE_CHARACTERS,
    SOURCE_LAYERS,
    ContextBuilder,
    ContextIndexService,
    ContextService,
)
from ainovel.services.counting import count_visible_characters
from ainovel.services.prompts import PromptService
from ainovel.services.workflows import (
    EXECUTABLE_WORKFLOW_STATUSES,
    Clock,
    StaleOutlineCompletion,
    SystemClock,
    WorkflowService,
)


_ROLE_BY_STEP = {
    "PLANNING": "batch_planner",
    "WRITING": "chapter_writer",
    "SUMMARIZING": "chapter_summarizer",
    "REVIEWING": "batch_reviewer",
}
_RESULT_BY_STEP: dict[str, type[BaseModel]] = {
    "PLANNING": BatchPlanDraft,
    "WRITING": ChapterDraft,
    "SUMMARIZING": ChapterSummaryDelta,
    "REVIEWING": BatchReview,
}
_SCHEMA_NAME_BY_STEP = {
    "PLANNING": "batch_plan",
    "WRITING": "chapter_draft",
    "SUMMARIZING": "chapter_summary_delta",
    "REVIEWING": "batch_review",
}
_WAITING_FOR = {
    "AWAITING_PLAN_APPROVAL": "plan_approval",
    "AWAITING_CONTENT_APPROVAL": "content_approval",
    "PAUSED_PROVIDER": "provider",
    "PAUSED_CONTEXT_OVERFLOW": "context_budget",
    "PAUSED_ATTEMPTS": "attempts",
    "PAUSED_REVIEW": "review",
    "PAUSED_STALE_VERSION": "stale_outline",
}


@dataclass(frozen=True)
class AdvanceResult:
    workflow_id: str
    status: str
    completed_step: str | None
    waiting_for: str | None
    candidate_batch_id: str | None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_request(request: ModelRequest) -> str:
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    return sha256(_canonical_json(asdict(request)).encode("utf-8")).hexdigest()


class WorkflowOrchestrator:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        provider_registry: ProviderRegistry,
        runner: AgentRunner,
        context_service: Callable[[Session], ContextService] = ContextService,
        prompt_service: Callable[[Session], PromptService] = PromptService,
        *,
        clock: Clock | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._registry = provider_registry
        self._runner = runner
        self._context_service_factory = context_service
        self._prompt_service_factory = prompt_service
        self._clock = clock or SystemClock()
        self._worker_id = worker_id or f"orchestrator-{uuid4()}"
        self._providers: dict[tuple[str, str, str], ModelProvider] = {}
        self._attempt_responses: dict[str, ModelResponse] = {}
        self._attempt_validations: dict[str, dict[str, object]] = {}

    def advance(self, workflow_id: str) -> AdvanceResult:
        service = self._workflow_service()
        try:
            service.recover_expired_claims(workflow_id, self._clock.now())
            claim = service.claim_step(
                workflow_id, EXECUTABLE_WORKFLOW_STATUSES, self._worker_id
            )
        finally:
            service.session.close()
        if claim is None:
            return self._current_result(workflow_id)
        if claim.kind == "CREATING_CANDIDATE_BATCH":
            return self._create_candidate_batch(claim)

        try:
            request = self._build_request(claim)
        except RequiredContextOverflow as error:
            overflow_service = self._workflow_service()
            try:
                overflow_service.pause_context_overflow(
                    claim.id, self._worker_id, error, claim_revision=claim.revision
                )
            finally:
                overflow_service.session.close()
            return self._current_result(workflow_id)
        except ProviderError as error:
            provider_service = self._workflow_service()
            try:
                provider_service.pause_provider_failure(
                    claim.id, self._worker_id, error, claim_revision=claim.revision
                )
            finally:
                provider_service.session.close()
            return self._current_result(workflow_id)

        attempt_service = self._workflow_service()
        try:
            try:
                attempt = attempt_service.record_attempt_start(
                    claim.id, digest_request(request), claim_revision=claim.revision
                )
            except ValueError:
                current = self._current_result(workflow_id)
                if current.status == "PAUSED_ATTEMPTS":
                    return current
                raise
        finally:
            attempt_service.session.close()
        try:
            run = self._runner.run_with_response(
                self._provider(claim), request, self._result_type(claim)
            )
            validations = self._validate_business_result(claim, run.result)
        except ProviderError as error:
            return self._record_failure(attempt.id, error)

        self._attempt_responses[attempt.id] = run.response
        self._attempt_validations[attempt.id] = validations
        try:
            return self._save_result_and_advance(attempt.id, claim, run.result)
        finally:
            self._attempt_responses.pop(attempt.id, None)
            self._attempt_validations.pop(attempt.id, None)

    def run_until_blocked(
        self, workflow_id: str, max_steps: int = 32
    ) -> AdvanceResult:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max steps must be a positive integer")
        result = self._current_result(workflow_id)
        for _ in range(max_steps):
            if result.status not in EXECUTABLE_WORKFLOW_STATUSES:
                return result
            result = self.advance(workflow_id)
        return result

    def _workflow_service(self) -> WorkflowService:
        return WorkflowService(self._session_factory(), clock=self._clock)

    def _build_request(self, step: WorkflowStep) -> ModelRequest:
        self._ensure_summary_indexes(step.workflow_id)
        if step.kind == "REVIEWING":
            self._ensure_review_evidence(step.id)
        with self._session_factory() as session:
            persisted_step = session.get(WorkflowStep, step.id)
            if persisted_step is None:
                raise ValueError("workflow step not found")
            workflow = session.get(GenerationWorkflow, persisted_step.workflow_id)
            if workflow is None:
                raise ValueError("workflow not found")
            role = _ROLE_BY_STEP.get(persisted_step.kind)
            if role is None:
                raise ValueError("workflow step does not use a model")
            snapshot = next(
                (
                    item
                    for item in self._prompt_service_factory(session).list_snapshots(
                        workflow.id
                    )
                    if item.role == role
                ),
                None,
            )
            if snapshot is None:
                raise ValueError("workflow prompt snapshot not found")
            payload = self._task_payload(session, workflow, persisted_step)
            configured_input = self._positive_snapshot_parameter(
                snapshot, "max_input_tokens"
            )
            configured_output = self._positive_snapshot_parameter(
                snapshot, "max_output_tokens"
            )
            provider = self._provider(persisted_step)
            try:
                capabilities = provider.capabilities(workflow.model_name)
                output_tokens = min(
                    configured_output, capabilities.max_output_tokens
                )
            except ProviderError:
                raise
            except Exception:
                raise ProviderUnavailable("provider is unavailable") from None
            try:
                input_capacity = effective_input_capacity(
                    configured_input,
                    capabilities.context_window,
                    output_tokens,
                )
            except ValueError:
                raise RequiredContextOverflow(
                    "provider_capacity", configured_input, 0
                ) from None

            uses_context = persisted_step.kind != "SUMMARIZING"
            if uses_context:
                payload["context_packet"] = {
                    "packet_id": "0" * 36,
                    "items": [],
                }
            provisional = self._request(
                workflow,
                persisted_step,
                snapshot,
                payload,
                input_capacity,
                output_tokens,
            )
            context_service = self._context_service_factory(session)
            estimator = context_service.budgeter.estimator
            fixed_overhead = estimator.estimate(_canonical_json(asdict(provisional)))
            if fixed_overhead > input_capacity:
                session.rollback()
                raise RequiredContextOverflow(
                    "request_framing", fixed_overhead, input_capacity
                )

            if not uses_context:
                session.rollback()
                return provisional
            ContextIndexService(session).rebuild_official(workflow.project_id)
            candidates = ContextBuilder(session).candidates_for_step(
                workflow.id, persisted_step.id
            )
            packet = context_service.build_packet(
                workflow.id,
                persisted_step.id,
                [candidate for candidate in candidates if candidate.required],
                [candidate for candidate in candidates if not candidate.required],
                {
                    "input_capacity_tokens": input_capacity,
                    "reserved_output_tokens": output_tokens,
                    "fixed_overhead_tokens": fixed_overhead,
                },
            )
            payload["context_packet"] = self._packet_payload(session, packet.id)
            request = self._request(
                workflow,
                persisted_step,
                snapshot,
                payload,
                input_capacity,
                output_tokens,
            )
            request = self._trim_to_serialized_capacity(
                session, packet, request, estimator, input_capacity
            )
            session.rollback()
            return request

    def _provider(self, step: WorkflowStep) -> ModelProvider:
        with self._session_factory() as session:
            workflow = session.get(GenerationWorkflow, step.workflow_id)
            if workflow is None:
                raise ValueError("workflow not found")
            key = (workflow.id, workflow.provider_name, workflow.model_name)
            provider = self._providers.get(key)
            if provider is None:
                provider = self._registry.get(workflow.provider_name)
                self._providers[key] = provider
            session.rollback()
            return provider

    def _result_type(self, step: WorkflowStep) -> type[BaseModel]:
        result_type = _RESULT_BY_STEP.get(step.kind)
        if result_type is None:
            raise ValueError("workflow step does not have a result schema")
        return result_type

    def _record_failure(
        self, attempt_id: str, error: ProviderError
    ) -> AdvanceResult:
        service = self._workflow_service()
        try:
            workflow = service.fail_attempt(attempt_id, error)
            workflow_id = workflow.id
        finally:
            service.session.close()
        return self._current_result(workflow_id)

    def _save_result_and_advance(
        self, attempt_id: str, step: WorkflowStep, result: BaseModel
    ) -> AdvanceResult:
        response = self._attempt_responses.get(attempt_id)
        if response is None:
            raise ValueError("model response metadata is unavailable")
        finalize_step = not (
            step.kind == "REVIEWING"
            and isinstance(result, BatchReview)
            and result.passed is False
            and bool(result.evidence_queries)
        )
        service = self._workflow_service()
        try:
            try:
                artifact = service.complete_attempt(
                    attempt_id, response, result, finalize_step=finalize_step
                )
            except StaleOutlineCompletion:
                return self._current_result(step.workflow_id)
        finally:
            service.session.close()

        if step.kind == "WRITING":
            self._persist_validation_artifact(
                artifact.id, self._attempt_validations.get(attempt_id, {})
            )
        elif step.kind == "SUMMARIZING":
            with self._session_factory() as session:
                ContextIndexService(session).index_workflow_artifact(artifact.id)
        elif step.kind == "REVIEWING" and not finalize_step:
            self._ensure_review_evidence(step.id)

        current = self._current_result(step.workflow_id)
        return AdvanceResult(
            workflow_id=current.workflow_id,
            status=current.status,
            completed_step=step.kind if finalize_step else None,
            waiting_for=current.waiting_for,
            candidate_batch_id=current.candidate_batch_id,
        )

    def _current_result(self, workflow_id: str) -> AdvanceResult:
        with self._session_factory() as session:
            workflow = session.get(GenerationWorkflow, workflow_id)
            if workflow is None:
                raise ValueError("workflow not found")
            result = AdvanceResult(
                workflow_id=workflow.id,
                status=workflow.status,
                completed_step=None,
                waiting_for=_WAITING_FOR.get(workflow.status),
                candidate_batch_id=workflow.candidate_batch_id,
            )
            session.rollback()
            return result

    def _task_payload(
        self,
        session: Session,
        workflow: GenerationWorkflow,
        step: WorkflowStep,
    ) -> dict[str, object]:
        if step.kind == "PLANNING":
            project = session.get(NovelProject, workflow.project_id)
            constitution = (
                session.get(
                    ConstitutionVersion, project.current_constitution_version_id
                )
                if project is not None
                and project.current_constitution_version_id is not None
                else None
            )
            if project is None or constitution is None or not constitution.author_approved:
                raise ValueError("workflow project constitution is invalid")
            outline_nodes = session.scalars(
                select(OutlineNode)
                .where(OutlineNode.outline_version_id == workflow.base_outline_version_id)
                .order_by(OutlineNode.order, OutlineNode.stable_key)
            ).all()
            chapter_count, visible_count = session.execute(
                select(
                    func.count(Chapter.id),
                    func.coalesce(func.sum(Chapter.visible_char_count), 0),
                ).where(
                    Chapter.project_id == workflow.project_id,
                    Chapter.official_chapter_number.is_not(None),
                )
            ).one()
            return {
                "project_constitution": deepcopy(constitution.content),
                "official_outline_id": workflow.base_outline_version_id,
                "official_outline_tree": [
                    {
                        "stable_key": node.stable_key,
                        "parent_key": node.parent_key,
                        "kind": node.kind,
                        "title": node.title,
                        "order": node.order,
                        "payload": deepcopy(node.payload),
                        "author_locked": node.author_locked,
                    }
                    for node in outline_nodes
                ],
                "requested_chapters": workflow.requested_chapters,
                "official_chapter_statistics": {
                    "chapter_count": chapter_count,
                    "visible_character_count": visible_count,
                },
                "budgets": self._budget_payload(workflow),
            }
        if step.kind == "WRITING":
            chapter_plan = self._chapter_plan(session, workflow, step.ordinal)
            return {
                "ordinal": step.ordinal,
                "chapter_plan": chapter_plan,
                "previous_candidate_summaries": self._prior_summaries(
                    session, workflow, step.ordinal
                ),
            }
        if step.kind == "SUMMARIZING":
            chapter = self._active_artifact_for_ordinal(
                session, workflow.id, "chapter_draft", step.ordinal
            )
            if chapter is None:
                raise ValueError("summarizer requires the generated chapter")
            return {
                "chapter": deepcopy(chapter.payload),
                "chapter_plan": self._chapter_plan(
                    session, workflow, step.ordinal
                ),
            }
        if step.kind == "REVIEWING":
            return self._reviewer_payload(session, workflow, step)
        raise ValueError("workflow step does not use a task payload")

    def _request(
        self,
        workflow: GenerationWorkflow,
        step: WorkflowStep,
        snapshot: WorkflowPromptSnapshot,
        payload: dict[str, object],
        input_capacity: int,
        output_tokens: int,
    ) -> ModelRequest:
        return ModelRequest(
            model=workflow.model_name,
            system_prompt=snapshot.prompt_body,
            input_payload=deepcopy(payload),
            output_schema=deepcopy(snapshot.output_schema),
            max_input_tokens=input_capacity,
            max_output_tokens=output_tokens,
            timeout_seconds=60.0,
            metadata={
                "agent_role": _ROLE_BY_STEP[step.kind],
                "schema_name": _SCHEMA_NAME_BY_STEP[step.kind],
                "workflow_id": workflow.id,
                "step_id": step.id,
                "ordinal": "" if step.ordinal is None else str(step.ordinal),
            },
        )

    @staticmethod
    def _positive_snapshot_parameter(
        snapshot: WorkflowPromptSnapshot, name: str
    ) -> int:
        value = snapshot.parameters.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError("workflow prompt snapshot token limit is invalid")
        return value

    @staticmethod
    def _budget_payload(workflow: GenerationWorkflow) -> dict[str, int]:
        return {
            "planner_input": workflow.planner_input_tokens,
            "planner_output": workflow.planner_output_tokens,
            "writer_input": workflow.writer_input_tokens,
            "writer_output": workflow.writer_output_tokens,
            "summarizer_input": workflow.summarizer_input_tokens,
            "summarizer_output": workflow.summarizer_output_tokens,
            "reviewer_input": workflow.reviewer_input_tokens,
            "reviewer_output": workflow.reviewer_output_tokens,
        }

    @staticmethod
    def _plan_artifact(
        session: Session, workflow_id: str
    ) -> WorkflowArtifact:
        artifact = session.scalar(
            select(WorkflowArtifact)
            .join(
                WorkflowStep,
                WorkflowStep.active_artifact_id == WorkflowArtifact.id,
            )
            .where(
                WorkflowArtifact.workflow_id == workflow_id,
                WorkflowArtifact.kind == "batch_plan",
                WorkflowStep.kind == "PLANNING",
                WorkflowStep.status == "COMPLETED",
            )
        )
        if artifact is None:
            raise ValueError("approved batch plan is unavailable")
        decision = session.scalar(
            select(PlanDecision.id).where(
                PlanDecision.workflow_id == workflow_id,
                PlanDecision.decision == "approved",
            )
        )
        if decision is None:
            raise ValueError("batch plan has not been approved")
        return artifact

    def _chapter_plan(
        self,
        session: Session,
        workflow: GenerationWorkflow,
        ordinal: int | None,
    ) -> dict[str, object]:
        if ordinal is None:
            raise ValueError("chapter ordinal is required")
        plan = self._plan_artifact(session, workflow.id)
        for chapter in plan.payload.get("chapters", []):
            if isinstance(chapter, dict) and chapter.get("ordinal") == ordinal:
                return deepcopy(chapter)
        raise ValueError("approved chapter plan is unavailable")

    @staticmethod
    def _active_artifact_for_ordinal(
        session: Session,
        workflow_id: str,
        kind: str,
        ordinal: int | None,
    ) -> WorkflowArtifact | None:
        return session.scalar(
            select(WorkflowArtifact)
            .join(
                WorkflowStep,
                WorkflowStep.active_artifact_id == WorkflowArtifact.id,
            )
            .where(
                WorkflowArtifact.workflow_id == workflow_id,
                WorkflowArtifact.kind == kind,
                WorkflowArtifact.ordinal == ordinal,
                WorkflowStep.status == "COMPLETED",
            )
        )

    def _prior_summaries(
        self,
        session: Session,
        workflow: GenerationWorkflow,
        ordinal: int | None,
    ) -> list[dict[str, object]]:
        if ordinal is None:
            raise ValueError("writer ordinal is required")
        rows = session.scalars(
            select(WorkflowArtifact)
            .join(
                WorkflowStep,
                WorkflowStep.active_artifact_id == WorkflowArtifact.id,
            )
            .where(
                WorkflowArtifact.workflow_id == workflow.id,
                WorkflowArtifact.kind == "chapter_summary_delta",
                WorkflowArtifact.ordinal.is_not(None),
                WorkflowArtifact.ordinal < ordinal,
                WorkflowStep.status == "COMPLETED",
            )
            .order_by(WorkflowArtifact.ordinal)
        ).all()
        if len(rows) != ordinal - 1:
            raise ValueError("writer requires every previous candidate summary")
        return [
            {
                "ordinal": row.ordinal,
                "summary": row.payload["summary"],
                "state_delta": deepcopy(row.payload["state_delta"]),
            }
            for row in rows
        ]

    def _reviewer_payload(
        self,
        session: Session,
        workflow: GenerationWorkflow,
        step: WorkflowStep,
    ) -> dict[str, object]:
        plan = self._plan_artifact(session, workflow.id)
        reports: list[dict[str, object]] = []
        for ordinal in range(1, workflow.requested_chapters + 1):
            chapter = self._active_artifact_for_ordinal(
                session, workflow.id, "chapter_draft", ordinal
            )
            summary = self._active_artifact_for_ordinal(
                session, workflow.id, "chapter_summary_delta", ordinal
            )
            if chapter is None or summary is None:
                raise ValueError("reviewer requires completed chapter reports")
            validation = self._validation_payload(session, chapter)
            reports.append(
                {
                    "ordinal": ordinal,
                    "title": chapter.payload["title"],
                    "summary": summary.payload["summary"],
                    "state_delta": deepcopy(summary.payload["state_delta"]),
                    "visible_char_count": chapter.visible_char_count,
                    "validation_results": validation,
                }
            )
        evidence_rows = session.scalars(
            select(WorkflowArtifact)
            .where(
                WorkflowArtifact.workflow_id == workflow.id,
                WorkflowArtifact.step_id == step.id,
                WorkflowArtifact.kind == "review_evidence_excerpt",
            )
            .order_by(WorkflowArtifact.created_at, WorkflowArtifact.id)
        ).all()
        return {
            "approved_batch_plan": deepcopy(plan.payload),
            "chapter_reports": reports,
            "evidence": [
                {
                    "query": artifact.payload["query"],
                    "canonical_source_type": artifact.payload[
                        "canonical_source_type"
                    ],
                    "canonical_source_id": artifact.payload[
                        "canonical_source_id"
                    ],
                    "excerpt_start": artifact.payload["excerpt_start"],
                    "excerpt_end": artifact.payload["excerpt_end"],
                    "text": artifact.text_content,
                }
                for artifact in evidence_rows
            ],
        }

    @staticmethod
    def _packet_payload(session: Session, packet_id: str) -> dict[str, object]:
        items = session.scalars(
            select(ContextPacketItem)
            .where(
                ContextPacketItem.packet_id == packet_id,
                ContextPacketItem.selected.is_(True),
            )
            .order_by(ContextPacketItem.position)
        ).all()
        return {
            "packet_id": packet_id,
            "items": [
                {
                    "stable_key": item.stable_source_key,
                    "layer": item.layer,
                    "source_type": item.source_type,
                    "source_version": item.source_version,
                    "state_scope": item.state_scope,
                    "text": item.text_snapshot,
                    "excerpt_start": item.excerpt_start,
                    "excerpt_end": item.excerpt_end,
                }
                for item in items
            ],
        }

    def _trim_to_serialized_capacity(
        self,
        session: Session,
        packet: ContextPacket,
        request: ModelRequest,
        estimator: Any,
        input_capacity: int,
    ) -> ModelRequest:
        current = request
        while estimator.estimate(_canonical_json(asdict(current))) > input_capacity:
            removable = session.scalar(
                select(ContextPacketItem)
                .where(
                    ContextPacketItem.packet_id == packet.id,
                    ContextPacketItem.selected.is_(True),
                    ContextPacketItem.required.is_(False),
                )
                .order_by(
                    ContextPacketItem.layer.desc(),
                    ContextPacketItem.relevance,
                    ContextPacketItem.temporal_distance.desc(),
                    ContextPacketItem.stable_source_key.desc(),
                )
            )
            if removable is None:
                used = estimator.estimate(_canonical_json(asdict(current)))
                session.rollback()
                raise RequiredContextOverflow(
                    "serialized_request", used, input_capacity
                )
            session.execute(
                update(ContextPacketItem)
                .where(
                    ContextPacketItem.id == removable.id,
                    ContextPacketItem.selected.is_(True),
                    ContextPacketItem.required.is_(False),
                )
                .values(selected=False, trim_reason="serialized_budget")
            )
            session.execute(
                update(ContextPacket)
                .where(ContextPacket.id == packet.id)
                .values(
                    used_input_tokens=ContextPacket.used_input_tokens
                    - removable.estimated_tokens
                )
            )
            session.commit()
            payload = deepcopy(current.input_payload)
            payload["context_packet"] = self._packet_payload(session, packet.id)
            current = ModelRequest(
                model=current.model,
                system_prompt=current.system_prompt,
                input_payload=payload,
                output_schema=deepcopy(current.output_schema),
                max_input_tokens=current.max_input_tokens,
                max_output_tokens=current.max_output_tokens,
                timeout_seconds=current.timeout_seconds,
                metadata=deepcopy(current.metadata),
            )
        return current

    def _validate_business_result(
        self, step: WorkflowStep, result: BaseModel
    ) -> dict[str, object]:
        if step.kind == "PLANNING":
            if not isinstance(result, BatchPlanDraft):
                raise ProviderProtocolError("provider returned an invalid plan")
            with self._session_factory() as session:
                workflow = session.get(GenerationWorkflow, step.workflow_id)
                if workflow is None or len(result.chapters) != workflow.requested_chapters:
                    raise ProviderProtocolError("provider returned an invalid plan")
                session.rollback()
            return {}
        if step.kind != "WRITING":
            return {}
        if not isinstance(result, ChapterDraft) or step.ordinal is None:
            raise ProviderProtocolError("provider returned an invalid chapter")
        with self._session_factory() as session:
            workflow = session.get(GenerationWorkflow, step.workflow_id)
            if workflow is None:
                raise ValueError("workflow not found")
            validations = self._chapter_validation_results(
                session,
                workflow,
                step.ordinal,
                result.title,
                result.body,
            )
            session.rollback()
        if (
            not validations["nonblank_title"]
            or not validations["nonblank_body"]
            or not validations["visible_length_valid"]
            or not validations["approved_plan_ordinal"]
            or not validations["approved_goal_present"]
            or not validations["approved_key_event_present"]
            or validations["obvious_repeated_blocks"]
            or not validations["ordinal_continuity"]
        ):
            raise ProviderProtocolError("provider chapter failed deterministic validation")
        return validations

    def _chapter_validation_results(
        self,
        session: Session,
        workflow: GenerationWorkflow,
        ordinal: int,
        title: str,
        body: str,
    ) -> dict[str, object]:
        chapter_plan = ChapterPlan.model_validate(
            self._chapter_plan(session, workflow, ordinal)
        )
        completed_writers = session.scalar(
            select(func.count())
            .select_from(WorkflowStep)
            .where(
                WorkflowStep.workflow_id == workflow.id,
                WorkflowStep.kind == "WRITING",
                WorkflowStep.ordinal < ordinal,
                WorkflowStep.status == "COMPLETED",
                WorkflowStep.active_artifact_id.is_not(None),
            )
        )
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", body)
            if len(paragraph.strip()) >= 80
        ]
        normalized_body = self._coverage_text(body)
        visible_count = count_visible_characters(body)
        return {
            "nonblank_title": bool(title.strip()),
            "nonblank_body": bool(body.strip()),
            "visible_character_count": visible_count,
            "visible_length_valid": 4500 <= visible_count <= 6000,
            "approved_plan_ordinal": chapter_plan.ordinal == ordinal,
            "approved_goal_present": self._coverage_text(chapter_plan.goal)
            in normalized_body,
            "approved_key_event_present": self._coverage_text(
                chapter_plan.ending_hook
            )
            in normalized_body,
            "obvious_repeated_blocks": len(paragraphs) != len(set(paragraphs)),
            "ordinal_continuity": completed_writers == ordinal - 1,
        }

    @staticmethod
    def _coverage_text(value: str) -> str:
        return re.sub(r"\s+", "", value).casefold()

    def _persist_validation_artifact(
        self, chapter_artifact_id: str, validations: dict[str, object]
    ) -> None:
        with self._session_factory() as session:
            chapter = session.get(WorkflowArtifact, chapter_artifact_id)
            if chapter is None:
                raise ValueError("chapter artifact not found")
            existing = session.scalar(
                select(WorkflowArtifact).where(
                    WorkflowArtifact.workflow_id == chapter.workflow_id,
                    WorkflowArtifact.step_id == chapter.step_id,
                    WorkflowArtifact.kind == "chapter_validation",
                    WorkflowArtifact.ordinal == chapter.ordinal,
                )
            )
            if existing is not None:
                session.rollback()
                return
            payload = deepcopy(validations)
            canonical = _canonical_json(payload)
            session.add(
                WorkflowArtifact(
                    id=str(uuid4()),
                    workflow_id=chapter.workflow_id,
                    step_id=chapter.step_id,
                    kind="chapter_validation",
                    ordinal=chapter.ordinal,
                    text_content=None,
                    payload=payload,
                    visible_char_count=chapter.visible_char_count,
                    content_hash=sha256(canonical.encode("utf-8")).hexdigest(),
                )
            )
            session.commit()

    def _validation_payload(
        self, session: Session, chapter: WorkflowArtifact
    ) -> dict[str, object]:
        row = session.scalar(
            select(WorkflowArtifact).where(
                WorkflowArtifact.workflow_id == chapter.workflow_id,
                WorkflowArtifact.step_id == chapter.step_id,
                WorkflowArtifact.kind == "chapter_validation",
                WorkflowArtifact.ordinal == chapter.ordinal,
            )
        )
        if row is None:
            workflow = session.get(GenerationWorkflow, chapter.workflow_id)
            if workflow is None or chapter.ordinal is None:
                raise ValueError("chapter validation context is unavailable")
            payload = self._chapter_validation_results(
                session,
                workflow,
                chapter.ordinal,
                chapter.payload["title"],
                chapter.payload["body"],
            )
            canonical = _canonical_json(payload)
            row = WorkflowArtifact(
                id=str(uuid4()),
                workflow_id=chapter.workflow_id,
                step_id=chapter.step_id,
                kind="chapter_validation",
                ordinal=chapter.ordinal,
                text_content=None,
                payload=payload,
                visible_char_count=chapter.visible_char_count,
                content_hash=sha256(canonical.encode("utf-8")).hexdigest(),
            )
            session.add(row)
            session.commit()
        return deepcopy(row.payload)

    def _ensure_review_evidence(self, step_id: str) -> None:
        to_index: list[str] = []
        with self._session_factory() as session:
            step = session.get(WorkflowStep, step_id)
            if step is None or step.kind != "REVIEWING":
                raise ValueError("review evidence requires a reviewer step")
            workflow = session.get(GenerationWorkflow, step.workflow_id)
            if workflow is None:
                raise ValueError("workflow not found")
            prior_review = session.scalar(
                select(WorkflowArtifact)
                .where(
                    WorkflowArtifact.workflow_id == workflow.id,
                    WorkflowArtifact.step_id == step.id,
                    WorkflowArtifact.kind == "batch_review",
                )
                .order_by(WorkflowArtifact.created_at, WorkflowArtifact.id)
            )
            if prior_review is None:
                session.rollback()
                return
            queries = prior_review.payload.get("evidence_queries")
            if not isinstance(queries, list) or not queries:
                session.rollback()
                return
            existing = session.scalars(
                select(WorkflowArtifact).where(
                    WorkflowArtifact.workflow_id == workflow.id,
                    WorkflowArtifact.step_id == step.id,
                    WorkflowArtifact.kind == "review_evidence_excerpt",
                )
            ).all()
            existing_queries = {
                artifact.payload["query"].strip()
                for artifact in existing
                if isinstance(artifact.payload.get("query"), str)
                and artifact.payload["query"].strip()
            }
            to_index.extend(artifact.id for artifact in existing)
            index = ContextIndexService(session)
            searchable = set(SOURCE_LAYERS) - EXCERPT_SOURCE_TYPES
            for query in queries:
                if not isinstance(query, str):
                    continue
                normalized_query = query.strip()
                if not normalized_query or normalized_query in existing_queries:
                    continue
                rows = index.search(
                    workflow.project_id, normalized_query, searchable, limit=2
                )
                for row in rows:
                    match_at = row.text.find(normalized_query)
                    center = match_at if match_at >= 0 else 0
                    start = max(0, center - MAX_L7_VISIBLE_CHARACTERS // 3)
                    end = min(len(row.text), start + MAX_L7_VISIBLE_CHARACTERS)
                    start = max(0, end - MAX_L7_VISIBLE_CHARACTERS)
                    excerpt = row.text[start:end]
                    if not excerpt:
                        continue
                    payload = {
                        "query": normalized_query,
                        "explicitly_requested": True,
                        "canonical_source_type": row.source_type,
                        "canonical_source_id": row.source_id,
                        "excerpt_start": start,
                        "excerpt_end": end,
                    }
                    artifact = WorkflowArtifact(
                        id=str(uuid4()),
                        workflow_id=workflow.id,
                        step_id=step.id,
                        kind="review_evidence_excerpt",
                        ordinal=None,
                        text_content=excerpt,
                        payload=payload,
                        visible_char_count=count_visible_characters(excerpt),
                        content_hash=sha256(excerpt.encode("utf-8")).hexdigest(),
                    )
                    session.add(artifact)
                    to_index.append(artifact.id)
                existing_queries.add(normalized_query)
            if session.new:
                session.commit()
            else:
                session.rollback()
        for artifact_id in dict.fromkeys(to_index):
            with self._session_factory() as session:
                ContextIndexService(session).index_workflow_artifact(artifact_id)

    def _ensure_summary_indexes(self, workflow_id: str) -> None:
        with self._session_factory() as session:
            artifact_ids = session.scalars(
                select(WorkflowArtifact.id)
                .join(
                    WorkflowStep,
                    WorkflowStep.active_artifact_id == WorkflowArtifact.id,
                )
                .where(
                    WorkflowArtifact.workflow_id == workflow_id,
                    WorkflowArtifact.kind == "chapter_summary_delta",
                    WorkflowStep.status == "COMPLETED",
                )
                .order_by(WorkflowArtifact.ordinal)
            ).all()
            indexed_ids = set(
                session.scalars(
                    select(ContextSource.source_id).where(
                        ContextSource.state_scope == f"workflow:{workflow_id}",
                        ContextSource.source_type == "chapter_summary_delta",
                        ContextSource.source_id.in_(artifact_ids),
                    )
                ).all()
            )
            session.rollback()
        for artifact_id in artifact_ids:
            if artifact_id in indexed_ids:
                continue
            with self._session_factory() as session:
                ContextIndexService(session).index_workflow_artifact(artifact_id)

    def _create_candidate_batch(self, step: WorkflowStep) -> AdvanceResult:
        with self._session_factory() as session:
            persisted_step = session.get(WorkflowStep, step.id)
            workflow = (
                session.get(GenerationWorkflow, persisted_step.workflow_id)
                if persisted_step is not None
                else None
            )
            if persisted_step is None or workflow is None:
                raise ValueError("candidate batch step is invalid")
            self._require_candidate_lease(persisted_step)
            batch = session.scalar(
                select(WritingBatch).where(
                    WritingBatch.source_workflow_id == workflow.id
                )
            )
            batch_id = batch.id if batch is not None else None
            batch_status = batch.status if batch is not None else None
            workflow_id = workflow.id
            project_id = workflow.project_id
            outline_id = workflow.base_outline_version_id
            planned_chapters = workflow.requested_chapters
            session.rollback()

        if batch_id is None:
            try:
                with self._session_factory() as session:
                    batch = BatchService(session).create(
                        project_id,
                        outline_id,
                        planned_chapters,
                        source_workflow_id=workflow_id,
                    )
                    batch_id = batch.id
                    batch_status = batch.status
            except Exception:
                with self._session_factory() as session:
                    batch = session.scalar(
                        select(WritingBatch).where(
                            WritingBatch.source_workflow_id == workflow_id
                        )
                    )
                    batch_id = batch.id if batch is not None else None
                    batch_status = batch.status if batch is not None else None
                    session.rollback()
                if batch_id is None:
                    raise

        if batch_status in {"approved", "rejected"}:
            return self._reconcile_terminal_candidate(workflow_id, step.kind)

        if batch_status == "draft":
            reconcile = self._workflow_service()
            try:
                reconcile.reconcile_batch_decision(workflow_id)
            finally:
                reconcile.session.close()

        with self._session_factory() as session:
            writers = session.scalars(
                select(WorkflowArtifact)
                .join(
                    WorkflowStep,
                    WorkflowStep.active_artifact_id == WorkflowArtifact.id,
                )
                .where(
                    WorkflowArtifact.workflow_id == workflow_id,
                    WorkflowArtifact.kind == "chapter_draft",
                    WorkflowStep.status == "COMPLETED",
                )
                .order_by(WorkflowArtifact.ordinal)
            ).all()
            summaries = {
                artifact.ordinal: artifact
                for artifact in session.scalars(
                    select(WorkflowArtifact)
                    .join(
                        WorkflowStep,
                        WorkflowStep.active_artifact_id == WorkflowArtifact.id,
                    )
                    .where(
                        WorkflowArtifact.workflow_id == workflow_id,
                        WorkflowArtifact.kind == "chapter_summary_delta",
                        WorkflowStep.status == "COMPLETED",
                    )
                ).all()
            }
            if [artifact.ordinal for artifact in writers] != list(
                range(1, planned_chapters + 1)
            ) or set(summaries) != set(range(1, planned_chapters + 1)):
                raise ValueError("candidate batch requires every validated chapter")
            copies = [
                (
                    artifact.ordinal,
                    artifact.payload["title"],
                    artifact.payload["body"],
                    deepcopy(summaries[artifact.ordinal].payload["state_delta"]),
                )
                for artifact in writers
            ]
            session.rollback()

        for ordinal, title, body, state_delta in copies:
            with self._session_factory() as session:
                existing = session.scalar(
                    select(Chapter).where(
                        Chapter.batch_id == batch_id,
                        Chapter.ordinal == ordinal,
                    )
                )
                if existing is not None:
                    if (
                        existing.title != title
                        or existing.body != body
                        or existing.state_delta != state_delta
                    ):
                        raise ValueError("candidate chapter conflicts with workflow artifact")
                    session.rollback()
                    continue
                BatchService(session).save_candidate_chapter(
                    batch_id, ordinal, title, body, state_delta
                )

        terminal_status = False
        with self._session_factory() as session:
            persisted_batch = BatchService(session).get(batch_id)
            if persisted_batch.status == "draft":
                persisted_batch = BatchService(session).mark_ready(batch_id)
            if persisted_batch.status in {"approved", "rejected"}:
                terminal_status = True
                session.rollback()
            elif persisted_batch.status != "ready_for_review":
                session.rollback()
                raise ValueError("candidate batch is not reconcilable")
        if terminal_status:
            return self._reconcile_terminal_candidate(workflow_id, step.kind)

        with self._session_factory() as session:
            current_step = session.get(WorkflowStep, step.id)
            if current_step is None:
                raise ValueError("candidate batch step is missing")
            self._require_candidate_lease(current_step)
            session.rollback()
        final_reconcile = self._workflow_service()
        try:
            final_reconcile.reconcile_batch_decision(workflow_id)
        finally:
            final_reconcile.session.close()
        current = self._current_result(workflow_id)
        return AdvanceResult(
            workflow_id=current.workflow_id,
            status=current.status,
            completed_step=step.kind,
            waiting_for=current.waiting_for,
            candidate_batch_id=current.candidate_batch_id,
        )

    def _reconcile_terminal_candidate(
        self, workflow_id: str, completed_step: str
    ) -> AdvanceResult:
        terminal_reconcile = self._workflow_service()
        try:
            terminal_reconcile.reconcile_batch_decision(workflow_id)
        finally:
            terminal_reconcile.session.close()
        current = self._current_result(workflow_id)
        return AdvanceResult(
            workflow_id=current.workflow_id,
            status=current.status,
            completed_step=completed_step,
            waiting_for=current.waiting_for,
            candidate_batch_id=current.candidate_batch_id,
        )

    def _require_candidate_lease(self, step: WorkflowStep) -> None:
        if (
            step.kind != "CREATING_CANDIDATE_BATCH"
            or step.status != "RUNNING"
            or step.lease_owner != self._worker_id
            or step.lease_expires_at is None
            or step.lease_expires_at <= self._clock.now()
            or step.active_artifact_id is not None
        ):
            raise ValueError("candidate batch lease conflict")

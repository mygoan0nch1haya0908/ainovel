from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from uuid import uuid4

from sqlalchemy import select, update, func

from ainovel.agents.stage_contracts import StageRoadmapDraft
from ainovel.agents.runner import AgentRunner
from ainovel.context import ConservativeEstimator, effective_input_capacity
from ainovel.models.audit import AuditEvent
from ainovel.models.outline import OutlineNode
from ainovel.models.project import NovelProject, ConstitutionVersion
from ainovel.models.stage import StoryStage, StageRoadmapVersion, StageModelAttempt, StageWorkflow, StageWorkflowNode
from ainovel.models.workflow import GenerationWorkflow
from ainovel.providers.contracts import ModelRequest
from ainovel.services.workflows import DEFAULT_BUDGETS, PROVIDER_NAMES, WorkflowService, WorkflowBudgets


STAGE_PROMPT = "根据作者阶段架构和锁定设定提出精简路线图。仅列有稳定标识的小章节节点；按因果顺序推进，依赖只能引用此前节点。不得生成正文或场景详情。保留已确认节点，必须覆盖关键事件和阶段终态，不得用章数代替节点列表。"


@dataclass(frozen=True)
class StageBudgets:
    input_tokens: int = 16000
    output_tokens: int = 8000
    attempt_limit: int = 2
    total_input_tokens: int = 32000
    total_output_tokens: int = 16000


@dataclass(frozen=True)
class StageBatchStart:
    workflow: GenerationWorkflow
    nodes: tuple[StageWorkflowNode, ...]


class StageService:
    def __init__(self, session):
        self.session = session

    def get(self, stage_id):
        stage = self.session.get(StoryStage, stage_id)
        if stage is None:
            raise ValueError("stage not found")
        return stage

    def create(self, project_id, architecture, actor="author"):
        project = self.session.get(NovelProject, project_id)
        if project is None or not project.official_outline_version_id:
            raise ValueError("official outline required")
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError("architecture required")
        stage = StoryStage(id=str(uuid4()), project_id=project_id,
                           base_outline_version_id=project.official_outline_version_id,
                           architecture=architecture.strip())
        self.session.add(stage)
        self.session.flush()
        self._audit(stage, "stage_created", actor, {})
        self.session.commit()
        return stage

    def list_for_project(self, project_id):
        return list(self.session.scalars(select(StoryStage).where(StoryStage.project_id == project_id).order_by(StoryStage.created_at)))

    def roadmap(self, roadmap_id):
        version = self.session.get(StageRoadmapVersion, roadmap_id)
        if version is None:
            raise ValueError("roadmap not found")
        return version

    def list_roadmaps(self, stage_id):
        return list(self.session.scalars(select(StageRoadmapVersion).where(StageRoadmapVersion.stage_id == stage_id).order_by(StageRoadmapVersion.version_number)))

    def propose_roadmap(self, stage_id, actor, provider_name, model_name, *, architecture=None, budgets=StageBudgets()):
        if provider_name not in PROVIDER_NAMES or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("invalid provider/model")
        if any(type(v) is not int or v <= 0 for v in asdict(budgets).values()) or budgets.attempt_limit > 2:
            raise ValueError("invalid finite stage budgets")
        self.session.expire_all()
        stage = self.get(stage_id)
        architecture = stage.architecture if architecture is None else architecture
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError("architecture required")
        try:
            project = self._lock_idle_project(stage)
            self._claim_stage(stage)
            constitution = self.session.get(ConstitutionVersion, project.current_constitution_version_id)
            if constitution is None or not constitution.author_approved:
                raise ValueError("approved constitution required")
            previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            outline_nodes = self.session.scalars(select(OutlineNode).where(OutlineNode.outline_version_id == stage.base_outline_version_id).order_by(OutlineNode.order)).all()
            version = StageRoadmapVersion(
                id=str(uuid4()), stage_id=stage.id,
                version_number=self.session.scalar(select(func.coalesce(func.max(StageRoadmapVersion.version_number), 0) + 1).where(StageRoadmapVersion.stage_id == stage.id)),
                input_revision=stage.revision, constitution_version_id=constitution.id,
                provider_name=provider_name, model_name=model_name.strip(), architecture=architecture.strip(),
                prompt_snapshot={"body": STAGE_PROMPT, "schema": StageRoadmapDraft.model_json_schema()},
                input_snapshot={"architecture": architecture.strip(), "constitution": deepcopy(constitution.content),
                                "outline": [{"key": n.stable_key, "kind": n.kind, "title": n.title, "payload": n.payload, "locked": n.author_locked} for n in outline_nodes],
                                "confirmed_chapters": stage.confirmed_chapters,
                                "previous_roadmap": deepcopy(previous.payload) if previous else None},
                input_token_limit=budgets.input_tokens, output_token_limit=budgets.output_tokens,
                total_input_token_limit=budgets.total_input_tokens, total_output_token_limit=budgets.total_output_tokens,
                attempt_limit=budgets.attempt_limit,
            )
            self.session.add(version)
            self._audit(stage, "stage_roadmap_requested", actor, {"roadmap_id": version.id})
            self.session.commit()
            return version
        except Exception:
            self.session.rollback()
            raise

    def generate_roadmap(self, roadmap_id, provider):
        """Dispatch one durable attempt; a retry must use the same roadmap ID."""
        self.session.expire_all()
        version = self.roadmap(roadmap_id)
        if version.status not in {"PENDING", "PAUSED_PROVIDER", "PAUSED_INVALID"}:
            raise ValueError("roadmap is not available for generation")
        stage = self.get(version.stage_id)
        if not self._inputs_current(stage, version):
            version.status = "PAUSED_STALE_VERSION"
            self.session.commit()
            return version
        number = version.attempts_used + 1
        # Keep the full reservation for missing usage, but never allow a known
        # actual overrun to disappear behind the smaller per-attempt reservation.
        reserved_input = max(
            version.attempts_used * version.input_token_limit,
            version.actual_input_tokens or 0,
        ) + version.input_token_limit
        reserved_output = max(
            version.attempts_used * version.output_token_limit,
            version.actual_output_tokens or 0,
        ) + version.output_token_limit
        if (
            number > version.attempt_limit
            or reserved_input > version.total_input_token_limit
            or reserved_output > version.total_output_token_limit
        ):
            version.status = "PAUSED_BUDGET"
            self.session.commit()
            return version
        try:
            capabilities = provider.capabilities(version.model_name)
            output = min(version.output_token_limit, capabilities.max_output_tokens)
            capacity = effective_input_capacity(version.input_token_limit, capabilities.context_window, output)
            request = ModelRequest(version.model_name, version.prompt_snapshot["body"], deepcopy(version.input_snapshot),
                                   deepcopy(version.prompt_snapshot["schema"]), capacity, output, 120.0,
                                   {"agent_role": "stage_planner", "schema_name": "stage_roadmap", "generation_version": "2", "roadmap_id": version.id})
            if ConservativeEstimator().estimate(json.dumps(asdict(request), ensure_ascii=False)) > capacity:
                version.status = "PAUSED_CONTEXT_OVERFLOW"
                self.session.commit()
                return version
        except Exception:
            version.status = "PAUSED_PROVIDER"
            self.session.commit()
            return version
        claim = self.session.execute(update(StageRoadmapVersion).where(
            StageRoadmapVersion.id == version.id, StageRoadmapVersion.status == version.status,
            StageRoadmapVersion.attempts_used == number - 1,
        ).values(status="RUNNING", attempts_used=number))
        if claim.rowcount != 1:
            self.session.rollback()
            raise ValueError("roadmap generation conflict")
        attempt = StageModelAttempt(id=str(uuid4()), roadmap_id=version.id, number=number, status="RUNNING")
        self.session.add(attempt)
        self.session.commit()
        response = None
        payload = None
        status = "PROPOSED"
        try:
            result = AgentRunner().run_with_response(provider, request, StageRoadmapDraft)
            response = result.response
            payload = result.result.model_dump()
        except Exception as error:
            response = getattr(error, "response", None)
            status = "PAUSED_INVALID" if response is not None else "PAUSED_PROVIDER"
        # A schema failure can still carry billable usage. Apply the same ceiling
        # checks to both successful and failed responses before saving either.
        if response is not None and (
            response.input_tokens is not None
            and response.input_tokens > request.max_input_tokens
            or response.output_tokens is not None
            and response.output_tokens > request.max_output_tokens
        ):
            status, payload = "PAUSED_BUDGET", None
        self.session.expire_all()
        version = self.roadmap(roadmap_id)
        stage = self.get(version.stage_id)
        try:
            # Acquire the stage row before reading version pointers again. This fences
            # outline/constitution changes and concurrent approval/start transitions.
            self.session.execute(update(StoryStage).where(StoryStage.id == stage.id).values(revision=StoryStage.revision))
            self.session.expire_all()
            if not self._inputs_current(stage, version):
                status, payload = "PAUSED_STALE_VERSION", None
            if version.status != "RUNNING" or version.attempts_used != number:
                raise ValueError("roadmap completion conflict")
            attempt = self.session.get(StageModelAttempt, attempt.id)
            attempt.input_tokens = response.input_tokens if response else None
            attempt.output_tokens = response.output_tokens if response else None
            version.actual_input_tokens = self._usage_total(version.actual_input_tokens, attempt.input_tokens)
            version.actual_output_tokens = self._usage_total(version.actual_output_tokens, attempt.output_tokens)
            if (
                version.actual_input_tokens is not None
                and version.actual_input_tokens > version.total_input_token_limit
                or version.actual_output_tokens is not None
                and version.actual_output_tokens > version.total_output_token_limit
            ):
                status, payload = "PAUSED_BUDGET", None
            attempt.status = status
            attempt.error_code = None if status == "PROPOSED" else status.lower()
            version.status, version.payload = status, payload
            self.session.commit()
            return version
        except Exception:
            self.session.rollback()
            raise

    def approve_roadmap(self, stage_id, roadmap_id, actor):
        self.session.expire_all()
        stage, version = self.get(stage_id), self.roadmap(roadmap_id)
        if version.stage_id != stage.id:
            raise ValueError("roadmap does not belong to stage")
        if stage.approved_roadmap_id == roadmap_id and version.status == "APPROVED":
            return version
        try:
            self._lock_idle_project(stage)
            if version.status != "PROPOSED" or not self._inputs_current(stage, version):
                raise ValueError("roadmap approval conflict or stale version")
            draft = StageRoadmapDraft.model_validate(version.payload)
            previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            if previous and previous.payload["nodes"][:stage.confirmed_chapters] != version.payload["nodes"][:stage.confirmed_chapters]:
                raise ValueError("revision cannot change confirmed nodes")
            self._claim_stage(stage)
            stage.approved_roadmap_id = version.id
            stage.architecture = version.architecture
            version.status, version.approved_by = "APPROVED", actor
            self._audit(stage, "stage_roadmap_approved", actor, {"roadmap_id": version.id, "estimated_chapters": draft.estimated_chapters})
            self.session.commit()
            return version
        except Exception:
            self.session.rollback()
            raise

    def roadmap_diff(self, stage_id, roadmap_id):
        stage, version = self.get(stage_id), self.roadmap(roadmap_id)
        if version.stage_id != stage.id or version.payload is None:
            raise ValueError("roadmap does not belong to stage or has no payload")
        previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
        before = previous.payload if previous else {}
        return {key: {"before": deepcopy(before.get(key)), "after": deepcopy(value)} for key, value in version.payload.items() if before.get(key) != value}

    def start_next_batch(
        self, stage_id: str, actor: str, provider_name: str, model_name: str,
        requested_chapters: int = 5, *, budgets: WorkflowBudgets = DEFAULT_BUDGETS,
    ) -> StageBatchStart:
        if type(requested_chapters) is not int or not 1 <= requested_chapters <= 5:
            raise ValueError("requested chapters must be from 1 to 5")
        self.session.expire_all()
        stage = self.get(stage_id)
        if not stage.approved_roadmap_id:
            raise ValueError("approved roadmap required")
        version = self.roadmap(stage.approved_roadmap_id)
        if version.status != "APPROVED":
            raise ValueError("approved roadmap required")
        if not self._inputs_current(stage, version, check_revision=False):
            raise ValueError("stale stage inputs")
        revision, confirmed, roadmap_id = stage.revision, stage.confirmed_chapters, version.id
        selected = deepcopy(version.payload["nodes"][confirmed:confirmed + requested_chapters])
        if not selected:
            raise ValueError("stage is complete")
        project_id = stage.project_id
        nodes = []

        def attach(workflow):
            current = self.get(stage_id)
            project = self.session.get(NovelProject, project_id)
            if current.approved_roadmap_id != roadmap_id or not self._inputs_current(current, version, check_revision=False):
                raise ValueError("stale stage inputs")
            claim = self.session.execute(update(StoryStage).where(StoryStage.id == stage_id, StoryStage.revision == revision,
                StoryStage.approved_roadmap_id == roadmap_id, StoryStage.confirmed_chapters == confirmed).values(revision=revision + 1))
            if claim.rowcount != 1:
                raise ValueError("stage start conflict")
            self.session.add(StageWorkflow(workflow_id=workflow.id, stage_id=stage_id, roadmap_id=roadmap_id, confirmed_start=confirmed))
            self.session.flush()
            for ordinal, node in enumerate(selected, 1):
                mapping = StageWorkflowNode(workflow_id=workflow.id, ordinal=ordinal, node_id=node["node_id"],
                                           stage_ordinal=node["ordinal"], book_ordinal=project.next_official_chapter_number + ordinal - 1)
                self.session.add(mapping)
                nodes.append(mapping)
            self._audit(current, "stage_batch_started", actor, {"workflow_id": workflow.id, "roadmap_id": roadmap_id})
        workflow = WorkflowService(self.session).start(project_id, provider_name, model_name, len(selected), budgets,
                                                       generation_version=2, _before_commit=attach)
        return StageBatchStart(workflow, tuple(nodes))

    def workflow_nodes(self, workflow_id):
        return list(self.session.scalars(select(StageWorkflowNode).where(StageWorkflowNode.workflow_id == workflow_id).order_by(StageWorkflowNode.ordinal)))

    def workflow_context(self, workflow_id, ordinal=None, *, include_roadmap=False):
        mapping = self.session.get(StageWorkflow, workflow_id)
        if mapping is None:
            return None
        version = self.roadmap(mapping.roadmap_id)
        nodes = self.workflow_nodes(workflow_id)
        result = {"stage_id": mapping.stage_id, "roadmap_id": version.id,
                  "confirmed_chapters": mapping.confirmed_start,
                  "goal": version.payload["goal"],
                  "instruction": "只展开所选节点；保留节点标题和目标，场景不得提前消耗后续节点事件。",
                  "nodes": [{**deepcopy(version.payload["nodes"][n.stage_ordinal - 1]),
                             "ordinal": n.ordinal, "stage_ordinal": n.stage_ordinal, "book_ordinal": n.book_ordinal}
                            for n in nodes if ordinal is None or n.ordinal == ordinal]}
        if include_roadmap:
            result["roadmap"] = deepcopy(version.payload)
        return result

    def commit_batch_progress(self, batch, chapters, first_number, actor):
        """Called inside BatchService.approve; never commits independently."""
        mapping = self.session.get(StageWorkflow, batch.source_workflow_id) if batch.source_workflow_id else None
        if mapping is None:
            return
        stage = self.get(mapping.stage_id)
        nodes = self.workflow_nodes(mapping.workflow_id)
        version = self.roadmap(mapping.roadmap_id)
        if stage.approved_roadmap_id != mapping.roadmap_id or not self._inputs_current(stage, version, check_revision=False):
            raise ValueError("stale stage roadmap")
        if mapping.committed_batch_id or len(nodes) != len(chapters) or [n.book_ordinal for n in nodes] != list(range(first_number, first_number + len(chapters))) or [n.stage_ordinal for n in nodes] != list(range(mapping.confirmed_start + 1, mapping.confirmed_start + len(chapters) + 1)):
            raise ValueError("stage approval conflict")
        claim = self.session.execute(update(StoryStage).where(StoryStage.id == stage.id, StoryStage.revision == stage.revision,
            StoryStage.approved_roadmap_id == mapping.roadmap_id, StoryStage.confirmed_chapters == mapping.confirmed_start).values(
                confirmed_chapters=mapping.confirmed_start + len(chapters), revision=stage.revision + 1))
        if claim.rowcount != 1:
            raise ValueError("stage approval conflict")
        mapping.committed_batch_id = batch.id
        self._audit(stage, "stage_progress_confirmed", actor, {"batch_id": batch.id, "roadmap_id": mapping.roadmap_id, "confirmed_chapters": mapping.confirmed_start + len(chapters)})

    def _lock_idle_project(self, stage):
        project = self.session.get(NovelProject, stage.project_id)
        claim = self.session.execute(update(NovelProject).where(NovelProject.id == stage.project_id,
            NovelProject.active_workflow_id.is_(None), NovelProject.active_batch_id.is_(None),
            NovelProject.official_outline_version_id == stage.base_outline_version_id).values(next_batch_sequence=NovelProject.next_batch_sequence))
        if claim.rowcount != 1:
            raise ValueError("active batch/workflow or stale outline prevents stage edit")
        return project

    def _claim_stage(self, stage):
        claim = self.session.execute(update(StoryStage).where(StoryStage.id == stage.id, StoryStage.revision == stage.revision).values(revision=stage.revision + 1))
        if claim.rowcount != 1:
            raise ValueError("stage revision conflict")

    def _inputs_current(self, stage, version, check_revision=True):
        project = self.session.get(NovelProject, stage.project_id)
        return project is not None and project.official_outline_version_id == stage.base_outline_version_id and project.current_constitution_version_id == version.constitution_version_id and (not check_revision or stage.revision == version.input_revision)

    @staticmethod
    def _usage_total(current, value):
        return current + value if current is not None and value is not None else None

    def _audit(self, stage, action, actor, details):
        self.session.add(AuditEvent(id=str(uuid4()), project_id=stage.project_id, entity_type="story_stage", entity_id=stage.id,
                                   action=action, actor=actor, details=details))

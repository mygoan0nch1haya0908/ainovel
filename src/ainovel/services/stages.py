from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from uuid import uuid4

from sqlalchemy import select, update, func

from ainovel.agents.stage_contracts import StageRoadmapDraft, StagePlotRoadmapDraft
from ainovel.services.stage_planning import PLOT_FORMAT, parse_stage_roadmap, validate_locked_prefix, chapter_slots
from ainovel.agents.runner import AgentRunner
from ainovel.context import ConservativeEstimator, effective_input_capacity
from ainovel.models.audit import AuditEvent
from ainovel.models.outline import OutlineNode
from ainovel.models.project import NovelProject, ConstitutionVersion
from ainovel.models.stage import StoryStage, StageRoadmapVersion, StageModelAttempt, StageWorkflow, StageWorkflowNode
from ainovel.models.workflow import GenerationWorkflow
from ainovel.providers.contracts import ModelRequest
from ainovel.providers.diagnostics import safe_failure_code
from ainovel.services.llm_diagnostics import failure_details, enforce_cooldown
from ainovel.services.workflows import DEFAULT_BUDGETS, PROVIDER_NAMES, WorkflowService, WorkflowBudgets
from ainovel.services.provider_resolution import validate_profile_binding


STAGE_PROMPT = "根据作者阶段架构和锁定设定提出精简路线图。仅列有稳定标识的小章节节点；按因果顺序推进，依赖只能引用此前节点。不得生成正文或场景详情。保留已确认节点，必须覆盖关键事件和阶段终态，不得用章数代替节点列表。"
PLOT_PROMPT = "根据作者阶段架构和锁定设定提出剧情点规划，每个剧情点分配 chapter_count 章，不列单章或场景详情。按因果顺序安排目标、关键事件、人物变化和伏笔；稳定 point_id，依赖只能引用此前点。60章可安排约8—12个剧情点，不强制凑数；总章数遵守作者要求。已开始并确认正文的剧情点整体锁定内容、顺序和章数。返回完整 plot_points_v1 JSON。"
PLOT_COMPACT_INSTRUCTION = "精简输出但不删作者硬约束：顶层 goal/start_state/end_state 各用一句话；顶层 key_events 仅列3—5个全阶段转折，foreshadowing 仅列跨阶段未解钩子，不复述各点清单。每点 goal 一句话，key_events 通常2—4条，character_changes 和 foreshadowing 通常各0—2条；每条短句，不写对白、场景、战斗过程或解释性评论。必要约束可超出建议条数；已锁定点保持原样。先统筹全部剧情点的章数，包含高潮与收束，再输出完整JSON；章数总和必须符合作者要求，不为前半段耗尽章数。具体场景留到写作前批次细化。"
PLOT_REVISION = "本次根据 revision.source_payload 与 revision.feedback 修订，返回完整规划；保持未涉及点的稳定标识。previous_roadmap 中累计起始章号不大于 confirmed_chapters 的所有点整体禁止修改。不得改变已锁定设定、全书与阶段约束。"
HIERARCHY_INSTRUCTION = "遵守设定与锁定结局；总剧情大纲约束阶段大纲，阶段大纲约束单章展开。不得静默改写上层约束。first_chapter 或 author_chapter_outline 仅适用于当前阶段第1章，不得作为所有章节的任务；未提供时自行按阶段大纲拆章。路线图和章节计划均须作者核对三层一致性后批准。"
REVISION_INSTRUCTION = "本次是作者反馈修订：以 revision.source_payload 为原安排，按 revision.feedback 修改并返回完整新安排，不只返回修改片段。未涉及的节点尽量保持稳定标识与内容。previous_roadmap 的前 confirmed_chapters 个节点为已确认正文对应节点，禁止改动。意见不得覆盖已锁定设定、全书与阶段约束。"


def _outline_hierarchy(nodes):
    """Extract the new setup hierarchy; old outlines retain their existing path."""
    book = next((n for n in nodes if n["key"] == "book" and n["payload"].get("book_outline")), None)
    stage = next((n for n in nodes if n["key"] == "stage-1" and n.get("parent_key") == "book"), None)
    if book is None or stage is None:
        return None
    result = {"book_outline": book["payload"]["book_outline"],
              "stage_architecture": stage["payload"].get("stage_architecture", "")}
    chapter = next((n for n in nodes if n.get("parent_key") == stage["key"]
                    and n["payload"].get("stage_ordinal") == 1
                    and n["payload"].get("chapter_outline")), None)
    if chapter:
        result["first_chapter"] = {"outline_key": chapter["key"], "title": chapter["title"], **deepcopy(chapter["payload"])}
    return result


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


def compact_stage_input(snapshot):
    """Lossless references to repeated long strings; never edit the audit snapshot."""
    seen = {}

    def visit(value, path):
        if isinstance(value, str) and len(value) >= 256:
            if value in seen:
                return {"$context_ref": seen[value]}
            seen[value] = path
        if isinstance(value, dict):
            return {key: visit(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, path + "/" + str(index)) for index, item in enumerate(value)]
        return value

    return visit(snapshot, "")


def stage_request(version, context_window, max_output_tokens):
    output = min(version.output_token_limit, max_output_tokens)
    capacity = effective_input_capacity(version.input_token_limit, context_window, output)
    return ModelRequest(
        version.model_name,
        version.prompt_snapshot["body"] + '\n{"$context_ref":"/path"} represents the exact same full text at that JSON Pointer in this input; apply all constraints there, not a missing value.',
        compact_stage_input(version.input_snapshot), deepcopy(version.prompt_snapshot["schema"]),
        capacity, output, 600.0,
        {"agent_role": "stage_planner", "schema_name": "stage_roadmap", "generation_version": "2", "roadmap_id": version.id},
    )


def stage_context_report(version, context_window, max_output_tokens):
    request = stage_request(version, context_window, max_output_tokens)
    estimated = ConservativeEstimator().estimate(json.dumps(asdict(request), ensure_ascii=False))
    return {"estimated": estimated, "capacity": request.max_input_tokens,
            "output": request.max_output_tokens, "overflow": estimated > request.max_input_tokens}


class StageService:
    def __init__(self, session, *, provider_resolver=None):
        self.session = session
        self.provider_resolver = provider_resolver

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

    def outline_context(self, stage_id):
        stage = self.get(stage_id)
        nodes = self.session.scalars(select(OutlineNode).where(OutlineNode.outline_version_id == stage.base_outline_version_id)).all()
        return _outline_hierarchy([{"key": n.stable_key, "parent_key": n.parent_key,
                                    "title": n.title, "payload": n.payload} for n in nodes])

    def roadmap(self, roadmap_id):
        version = self.session.get(StageRoadmapVersion, roadmap_id)
        if version is None:
            raise ValueError("roadmap not found")
        return version

    def list_roadmaps(self, stage_id):
        return list(self.session.scalars(select(StageRoadmapVersion).where(StageRoadmapVersion.stage_id == stage_id).order_by(StageRoadmapVersion.version_number)))

    def propose_roadmap(self, stage_id, actor, provider_name, model_name, *, architecture=None,
                        budgets=None, model_profile_version_id=None, context_window=32000,
                        max_output_tokens=8000, source_roadmap_id=None, feedback=None, expected_revision=None,
                        roadmap_format=PLOT_FORMAT, requested_output_tokens=None):
        if provider_name not in PROVIDER_NAMES or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("invalid provider/model")
        if budgets is not None and (any(type(v) is not int or v <= 0 for v in asdict(budgets).values()) or budgets.attempt_limit > 2):
            raise ValueError("invalid finite stage budgets")
        if requested_output_tokens is not None and (type(requested_output_tokens) is not int
                or not 1 <= requested_output_tokens <= 2147483647 or budgets is not None):
            raise ValueError('invalid or conflicting stage output budget')
        self.session.expire_all()
        stage = self.get(stage_id)
        architecture = stage.architecture if architecture is None else architecture
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError("architecture required")
        try:
            project = self._lock_idle_project(stage)
            revision = None
            if source_roadmap_id is not None:
                source = self.roadmap(source_roadmap_id)
                if (source.stage_id != stage.id or source.payload is None
                        or not isinstance(feedback, str) or not feedback.strip()
                        or (expected_revision is not None and expected_revision != stage.revision)
                        or not ((source.status == 'PROPOSED' and self._inputs_current(stage, source))
                                or (source.status == 'APPROVED' and stage.approved_roadmap_id == source.id
                                    and self._inputs_current(stage, source, check_revision=False)))):
                    raise ValueError('feedback source is unavailable or stale')
                revision = {'source_roadmap_id': source.id, 'source_version': source.version_number,
                            'source_payload': deepcopy(source.payload), 'feedback': feedback.strip()}
                roadmap_format = source.payload.get('format', 'legacy')
            if roadmap_format not in {'legacy', PLOT_FORMAT}:
                raise ValueError('unknown roadmap format')
            previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            if previous and stage.confirmed_chapters and previous.payload.get('format', 'legacy') != roadmap_format:
                raise ValueError('cannot change roadmap format after confirmed chapters / 已确认正文的阶段不能转换规划格式')
            schema = StagePlotRoadmapDraft if roadmap_format == PLOT_FORMAT else StageRoadmapDraft
            prompt = PLOT_PROMPT + PLOT_COMPACT_INSTRUCTION if roadmap_format == PLOT_FORMAT else STAGE_PROMPT
            revision_prompt = PLOT_REVISION if roadmap_format == PLOT_FORMAT else REVISION_INSTRUCTION
            profile = validate_profile_binding(self.session, provider_name, model_name, model_profile_version_id)
            if budgets is None:
                window = profile.context_limit if profile is not None else context_window
                maximum_output = profile.output_limit if profile is not None else max_output_tokens
                if type(window) is not int or not 1 <= window <= 2147483647 or type(maximum_output) is not int or maximum_output < 1:
                    raise ValueError("invalid model capacity")
                output = min(32000, maximum_output) if requested_output_tokens is None else requested_output_tokens
                if output > maximum_output:
                    raise ValueError('stage output budget exceeds model output capacity')
                incoming = effective_input_capacity(window, window, output)
                budgets = StageBudgets(input_tokens=incoming, output_tokens=output,
                                       total_input_tokens=incoming * 2, total_output_tokens=output * 2)
            if profile is not None:
                output = min(budgets.output_tokens, profile.output_limit)
                incoming = min(budgets.input_tokens, profile.context_limit - output)
                if incoming < 1:
                    raise ValueError("model profile context limit is too small")
                budgets = replace(budgets, input_tokens=incoming, output_tokens=output)
            self._claim_stage(stage)
            constitution = self.session.get(ConstitutionVersion, project.current_constitution_version_id)
            if constitution is None or not constitution.author_approved:
                raise ValueError("approved constitution required")
            previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            outline_nodes = self.session.scalars(select(OutlineNode).where(OutlineNode.outline_version_id == stage.base_outline_version_id).order_by(OutlineNode.order)).all()
            frozen_outline = [{"key": n.stable_key, "parent_key": n.parent_key, "kind": n.kind,
                               "title": n.title, "payload": deepcopy(n.payload), "locked": n.author_locked} for n in outline_nodes]
            hierarchy = _outline_hierarchy(frozen_outline)
            version = StageRoadmapVersion(
                id=str(uuid4()), stage_id=stage.id,
                version_number=self.session.scalar(select(func.coalesce(func.max(StageRoadmapVersion.version_number), 0) + 1).where(StageRoadmapVersion.stage_id == stage.id)),
                input_revision=stage.revision, constitution_version_id=constitution.id,
                provider_name=provider_name, model_name=model_name if profile is not None else model_name.strip(),
                model_profile_version_id=model_profile_version_id, architecture=architecture.strip(),
                prompt_snapshot={"body": prompt + (HIERARCHY_INSTRUCTION if hierarchy else "") + (revision_prompt if revision else ""), "schema": schema.model_json_schema()},
                input_snapshot={"architecture": architecture.strip(), "constitution": deepcopy(constitution.content),
                                "outline": frozen_outline,
                                **({"outline_hierarchy": hierarchy} if hierarchy else {}),
                                **({"revision": revision} if revision else {}),
                                "confirmed_chapters": stage.confirmed_chapters,
                                "previous_roadmap": deepcopy(previous.payload) if previous else None},
                input_token_limit=budgets.input_tokens, output_token_limit=budgets.output_tokens,
                total_input_token_limit=budgets.total_input_tokens, total_output_token_limit=budgets.total_output_tokens,
                attempt_limit=budgets.attempt_limit,
            )
            self.session.add(version)
            self._audit(stage, "stage_roadmap_requested", actor,
                        {"roadmap_id": version.id, "model_profile_version_id": model_profile_version_id})
            if revision:
                self._audit(stage, 'stage_roadmap_feedback_saved', actor,
                            {'roadmap_id': version.id, 'source_roadmap_id': source_roadmap_id})
            self.session.commit()
            return version
        except Exception:
            self.session.rollback()
            raise

    def revise_roadmap(self, stage_id, roadmap_id, feedback, actor='author', *, expected_revision=None):
        """Save an immutable feedback proposal; never call a provider here."""
        source = self.roadmap(roadmap_id)
        budgets = StageBudgets(source.input_token_limit, source.output_token_limit, source.attempt_limit,
                               source.total_input_token_limit, source.total_output_token_limit)
        return self.propose_roadmap(stage_id, actor, source.provider_name, source.model_name,
            architecture=source.architecture, budgets=budgets, model_profile_version_id=source.model_profile_version_id,
            source_roadmap_id=roadmap_id, feedback=feedback, expected_revision=expected_revision)

    def generate_roadmap(self, roadmap_id, provider):
        """Dispatch one durable attempt; a retry must use the same roadmap ID."""
        self.session.expire_all()
        version = self.roadmap(roadmap_id)
        if version.provider_name == "compatible" and self.provider_resolver is None:
            raise ValueError("compatible roadmap requires profile resolver")
        if version.status not in {"PENDING", "PAUSED_PROVIDER", "PAUSED_INVALID"}:
            raise ValueError("roadmap is not available for generation")
        stage = self.get(version.stage_id)
        enforce_cooldown(self.session,stage.id)
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
            if version.provider_name == "compatible":
                provider = self.provider_resolver.resolve(
                    version.provider_name, version.model_name,
                    model_profile_version_id=version.model_profile_version_id,
                )
            capabilities = provider.capabilities(version.model_name)
            request = stage_request(version, capabilities.context_window, capabilities.max_output_tokens)
            if ConservativeEstimator().estimate(json.dumps(asdict(request), ensure_ascii=False)) > request.max_input_tokens:
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
        failure_code = None
        schema_issues = []
        diagnostic_details = None
        request = replace(request, metadata={**request.metadata, 'round': str(version.version_number), 'attempt': str(number)})
        try:
            schema = StagePlotRoadmapDraft if 'points' in version.prompt_snapshot['schema'].get('properties', {}) else StageRoadmapDraft
            from sqlalchemy.orm import sessionmaker
            from ainovel.services.project_llm_guard import project_call
            with project_call(sessionmaker(bind=self.session.bind), stage.project_id, attempt.id):
                result = AgentRunner().run_with_response(provider, request, schema)
            response = result.response
            payload = result.result.model_dump()
        except Exception as error:
            response = getattr(error, "response", None)
            diagnostic_details = failure_details(error,roadmap_id=version.id,attempt_number=number)
            failure_code = safe_failure_code(error)
            if failure_code == 'schema_mismatch':
                from ainovel.agents.schema_diagnostics import safe_schema_issues
                schema_issues = safe_schema_issues(getattr(error, 'schema_issues', []), version.prompt_snapshot['schema'])
            status = "PAUSED_INVALID" if response is not None else "PAUSED_PROVIDER"
            if failure_code in {"provider_quota", "provider_api", "provider_rate_limit", "provider_temporary"}:
                status = "PAUSED_PROVIDER"
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
            attempt.error_code = (None if status == "PROPOSED" else
                                  failure_code if status in {"PAUSED_INVALID", "PAUSED_PROVIDER"} and failure_code
                                  else status.lower())
            version.status, version.payload = status, payload
            if diagnostic_details:
                self._audit(stage,'llm_call_failed','system',diagnostic_details)
            if schema_issues:
                self._audit(stage, 'stage_schema_validation_failed', 'system',
                            {'roadmap_id': version.id, 'attempt_number': number, 'issues': schema_issues})
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
            draft = parse_stage_roadmap(version.payload)
            previous = self.roadmap(stage.approved_roadmap_id) if stage.approved_roadmap_id else None
            if previous:
                validate_locked_prefix(previous.payload, version.payload, stage.confirmed_chapters)
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
        revision = version.input_snapshot.get('revision')
        before = revision['source_payload'] if revision else previous.payload if previous else {}
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
        if version.model_profile_version_id is not None and (
            provider_name != version.provider_name or model_name != version.model_name
        ):
            raise ValueError("stage batch model binding mismatch")
        if not self._inputs_current(stage, version, check_revision=False):
            raise ValueError("stale stage inputs")
        revision, confirmed, roadmap_id = stage.revision, stage.confirmed_chapters, version.id
        selected = chapter_slots(version.payload, confirmed, requested_chapters)
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
                                           stage_ordinal=node["stage_ordinal"], book_ordinal=project.next_official_chapter_number + ordinal - 1)
                self.session.add(mapping)
                nodes.append(mapping)
            self._audit(current, "stage_batch_started", actor, {"workflow_id": workflow.id, "roadmap_id": roadmap_id})
        workflow = WorkflowService(self.session).start(project_id, provider_name, model_name, len(selected), budgets,
                                                       generation_version=2,
                                                       _planning_format=version.payload.get('format'),
                                                       model_profile_version_id=version.model_profile_version_id,
                                                       _before_commit=attach)
        return StageBatchStart(workflow, tuple(nodes))

    def workflow_nodes(self, workflow_id):
        return list(self.session.scalars(select(StageWorkflowNode).where(StageWorkflowNode.workflow_id == workflow_id).order_by(StageWorkflowNode.ordinal)))

    def workflow_context(self, workflow_id, ordinal=None, *, include_roadmap=False, scoped_memory=False):
        mapping = self.session.get(StageWorkflow, workflow_id)
        if mapping is None:
            return None
        version = self.roadmap(mapping.roadmap_id)
        nodes = self.workflow_nodes(workflow_id)
        result = {"stage_id": mapping.stage_id, "roadmap_id": version.id,
                  "confirmed_chapters": mapping.confirmed_start,
                  "goal": version.payload["goal"]}
        if version.payload.get('format') == PLOT_FORMAT:
            slots = {slot['stage_ordinal']: slot for slot in chapter_slots(version.payload, mapping.confirmed_start, len(nodes))}
            selected = [{**slots[n.stage_ordinal], 'ordinal': n.ordinal, 'book_ordinal': n.book_ordinal}
                        for n in nodes if ordinal is None or n.ordinal == ordinal]
            point_ids = {slot['point_id'] for slot in selected}
            result.update(format=PLOT_FORMAT, slots=selected,
                          points=[deepcopy(p) for p in version.payload['points'] if p['point_id'] in point_ids],
                          instruction='只细化预留章节位置，按点内进度分配不同的单章目标、场景与悬念，不得每章完成整个剧情点或提前消耗后续剧情点。')
        else:
            result.update({
                  "instruction": "只展开所选节点；保留节点标题和目标，场景不得提前消耗后续节点事件。",
                  "nodes": [{**deepcopy(version.payload["nodes"][n.stage_ordinal - 1]),
                             "ordinal": n.ordinal, "stage_ordinal": n.stage_ordinal, "book_ordinal": n.book_ordinal}
                            for n in nodes if ordinal is None or n.ordinal == ordinal]})
        if include_roadmap and not scoped_memory:
            result["roadmap"] = deepcopy(version.payload)
        hierarchy = version.input_snapshot.get("outline_hierarchy")
        if hierarchy:
            if not scoped_memory:
                result["outline_constraints"] = {key: deepcopy(value) for key, value in hierarchy.items() if key != "first_chapter"}
                result["instruction"] += HIERARCHY_INSTRUCTION
            first_chapter = hierarchy.get("first_chapter")
            if first_chapter:
                # This internal key lets the caller exclude the duplicate raw
                # outline source from both its tree and retrieval packet.
                result["_scoped_outline_keys"] = [first_chapter["outline_key"]]
                for node in result.get('slots', result.get('nodes', [])):
                    if node["stage_ordinal"] == 1:
                        node["author_chapter_outline"] = deepcopy(first_chapter)
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

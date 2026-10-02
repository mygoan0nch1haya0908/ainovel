from dataclasses import replace

import pytest
from sqlalchemy import select

from ainovel.models import ModelAttempt, WorkflowArtifact, WorkflowStep, WritingBatch
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.fake import FakeProvider
from ainovel.providers.qwen import QwenProvider
from ainovel.services.stages import StageService
from ainovel.services.workflows import DEFAULT_BUDGETS, WorkflowService
from test_draft_repair import body_with_evidence, start_approved_v2, v2_plan_payload
from test_orchestrator import clock, make_orchestrator, ready_project, response, session_factory
from test_qwen_provider import FakeQwenClient, chat_response, summary_request


@pytest.mark.parametrize("dimension", ["input", "output"])
@pytest.mark.parametrize("excess", [0, 1])
def test_response_budget_boundary_pauses_before_plan_artifact(
    session, ready_project, session_factory, clock, dimension, excess
):
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS, generation_version=2
    )
    limit = getattr(workflow, f"total_{dimension}_token_limit")
    allowance = getattr(workflow, f"planner_{dimension}_tokens")
    setattr(workflow, f"actual_{dimension}_tokens", limit - allowance)
    session.commit()
    reply = response(v2_plan_payload(), 1, **{f"{dimension}_tokens": allowance + excess})
    result = make_orchestrator(session_factory, FakeProvider([reply]), clock).advance(workflow.id)
    session.expire_all()
    assert getattr(workflow, f"actual_{dimension}_tokens") == limit + excess
    assert result.status == ("PAUSED_ATTEMPTS" if excess else "AWAITING_PLAN_APPROVAL")
    artifacts = session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == workflow.id)).all()
    assert len(artifacts) == (0 if excess else 1)
    attempt = session.scalar(select(ModelAttempt).join(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    assert getattr(attempt, f"{dimension}_tokens") == allowance + excess
    if excess:
        assert attempt.status == "FAILED"
        assert workflow.current_position == 0
        assert workflow.last_error_code == "workflow_budget_exhausted"
        assert not WorkflowService(session).can_resume(workflow.id)


def test_final_reviewer_cannot_create_candidate_after_response_exceeds_total(
    session, ready_project, session_factory, clock
):
    class ReviewerOverrun(DemoFakeProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.metadata["agent_role"] == "batch_reviewer":
                return replace(result, output_tokens=6001)
            return result

    service = WorkflowService(session, clock=clock)
    workflow = service.start(ready_project.id, "fake", "demo", 1, DEFAULT_BUDGETS, generation_version=2)
    orchestrator = make_orchestrator(session_factory, ReviewerOverrun(), clock)
    assert orchestrator.advance(workflow.id).status == "AWAITING_PLAN_APPROVAL"
    service.approve_plan(workflow.id, "author")
    for _ in range(3):
        orchestrator.advance(workflow.id)
    session.expire_all()
    assert workflow.status == "REVIEWING_BATCH"
    workflow.total_output_token_limit = 112000
    workflow.actual_output_tokens = 106000
    position = workflow.current_position
    session.commit()
    result = orchestrator.run_until_blocked(workflow.id)
    session.expire_all()
    assert result.status == "PAUSED_ATTEMPTS"
    assert workflow.actual_output_tokens == 112001
    assert workflow.current_position == position
    assert workflow.candidate_batch_id is None
    assert session.scalar(select(WritingBatch).where(WritingBatch.source_workflow_id == workflow.id)) is None
    assert session.scalar(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == workflow.id, WorkflowArtifact.kind == "batch_review")) is None


@pytest.mark.parametrize("invalid", [None, True, -1, "311"])
def test_qwen_failure_preserves_valid_dimension_when_other_metadata_is_invalid(invalid):
    from ainovel.providers.diagnostics import ResponseFailure
    reply = chat_response(content="private-invalid-json")
    reply.id = None
    reply.usage.prompt_tokens = invalid
    reply.usage.completion_tokens = 322
    with pytest.raises(ResponseFailure) as caught:
        QwenProvider(FakeQwenClient(reply), allow_real_calls=True).generate(summary_request())
    metadata = caught.value.response
    assert metadata.input_tokens is None
    assert metadata.output_tokens == 322
    assert metadata.provider_response_id is None
    assert metadata.structured is None and metadata.text is None
    assert "private-invalid-json" not in str(caught.value)


@pytest.mark.parametrize("version", [1, 2])
def test_completion_keeps_unknown_usage_unknown_and_legacy_limits_compatible(
    session, ready_project, session_factory, clock, version
):
    from test_orchestrator import plan_payload
    workflow = WorkflowService(session, clock=clock).start(
        ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS, generation_version=version
    )
    reply = replace(response(v2_plan_payload() if version == 2 else plan_payload(1), 1), input_tokens=None)
    result = make_orchestrator(session_factory, FakeProvider([reply]), clock).advance(workflow.id)
    session.expire_all()
    assert result.status == "AWAITING_PLAN_APPROVAL"
    assert workflow.actual_input_tokens == 0
    assert workflow.actual_output_tokens == 50
    attempt = session.scalar(select(ModelAttempt).join(WorkflowStep).where(WorkflowStep.workflow_id == workflow.id))
    assert attempt.input_tokens is None and attempt.output_tokens == 50
    if version == 1:
        assert workflow.total_input_token_limit is None


@pytest.mark.parametrize("failure", ["json", "refusal", "length"])
@pytest.mark.parametrize("consumer", ["workflow", "stage"])
def test_actual_qwen_parse_failure_retains_usage_without_body(
    session, ready_project, session_factory, clock, failure, consumer
):
    secret = "<script>credential-author-secret</script>"
    reply = chat_response(content=secret, finish_reason="length" if failure == "length" else "stop",
                          refusal=secret if failure == "refusal" else None)
    reply.usage.prompt_tokens = 311
    reply.usage.completion_tokens = 322
    provider = QwenProvider(FakeQwenClient(reply), allow_real_calls=True)
    if consumer == "stage":
        from ainovel.models import StageModelAttempt
        service = StageService(session)
        stage = service.create(ready_project.id, "查案", "author")
        version = service.propose_roadmap(stage.id, "author", "fake", "scripted")
        record = service.generate_roadmap(version.id, provider)
        attempt = session.scalar(select(StageModelAttempt).where(StageModelAttempt.roadmap_id == version.id))
        assert record.payload is None
    else:
        record = WorkflowService(session, clock=clock).start(
            ready_project.id, "fake", "scripted", 1, DEFAULT_BUDGETS, generation_version=2
        )
        make_orchestrator(session_factory, provider, clock).advance(record.id)
        session.expire_all()
        attempt = session.scalar(select(ModelAttempt).join(WorkflowStep).where(WorkflowStep.workflow_id == record.id))
        assert session.scalar(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == record.id)) is None
        assert secret not in (record.last_error_detail or "")
    assert (attempt.input_tokens, attempt.output_tokens) == (311, 322)
    assert (record.actual_input_tokens, record.actual_output_tokens) == (311, 322)


@pytest.mark.parametrize("invalid_excerpt", [False, True])
def test_coverage_page_explains_target_verdict_and_exact_excerpt_failure(
    client, session, ready_project, session_factory, clock, invalid_excerpt
):
    issue = "<script>缺少进城行动</script>"
    coverage = {
        "goal": {"passed": invalid_excerpt, "excerpt": "正文没有这句话" if invalid_excerpt else "守卫终于让开了城门", "issues": [issue]},
        "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起", "issues": []},
    }
    provider = FakeProvider([
        response(v2_plan_payload(), 1),
        response({"title": "城门夜变", "body": body_with_evidence(5200)}, 2),
        response(coverage, 3),
    ])
    workflow, orchestrator = start_approved_v2(session_factory, session, ready_project, clock, provider)
    orchestrator.advance(workflow.id)
    result = orchestrator.advance(workflow.id)
    assert result.status == "PAUSED_REVIEW"
    session.expire_all()
    assert workflow.last_error_detail == "chapter coverage requires author attention"
    page = client.get(f"/workflows/{workflow.id}").text
    assert "目标覆盖" in page and "章末钩子覆盖" in page
    assert "&lt;script&gt;缺少进城行动&lt;/script&gt;" in page
    assert issue not in page
    assert ("摘录无效：不是正文中的非空连续原文" if invalid_excerpt else "语义判定：未通过") in page
    assert "摘录有效：已核对正文原文" in page
    assert coverage["goal"]["excerpt"] in page
    assert session.scalar(select(WorkflowArtifact).where(WorkflowArtifact.workflow_id == workflow.id, WorkflowArtifact.kind == "chapter_draft")) is None
    assert workflow.candidate_batch_id is None


def test_new_coverage_wire_requires_issues_but_frozen_v2_snapshot_remains_usable(
    session, ready_project, session_factory, clock
):
    from copy import deepcopy
    from ainovel.models import WorkflowPromptSnapshot
    coverage = {
        "goal": {"passed": True, "excerpt": "守卫终于让开了城门"},
        "ending_hook": {"passed": True, "excerpt": "暗处的铃声忽然响起"},
    }
    provider = FakeProvider([
        response(v2_plan_payload(), 1),
        response({"title": "城门夜变", "body": body_with_evidence(5200)}, 2),
        response(coverage, 3),
    ])
    workflow, orchestrator = start_approved_v2(session_factory, session, ready_project, clock, provider)
    snapshot = session.scalar(select(WorkflowPromptSnapshot).where(
        WorkflowPromptSnapshot.workflow_id == workflow.id,
        WorkflowPromptSnapshot.role == "chapter_coverage_reviewer",
    ))
    schema = deepcopy(snapshot.output_schema)
    verdict = schema["$defs"]["CoverageVerdict"]
    assert set(verdict["required"]) == set(verdict["properties"])
    # Reconstruct the historical persisted contract, then run against it unchanged.
    verdict["properties"].pop("issues")
    verdict["required"].remove("issues")
    snapshot.output_schema = schema
    snapshot.prompt_body = "Historical coverage prompt: passed and exact excerpt only."
    frozen_prompt = snapshot.prompt_body
    session.commit()
    orchestrator.advance(workflow.id)
    assert orchestrator.advance(workflow.id).completed_step == "VALIDATING_CHAPTER"
    assert provider.requests[-1].output_schema == schema
    assert provider.requests[-1].system_prompt == frozen_prompt
    session.expire_all()
    assert snapshot.output_schema == schema and snapshot.prompt_body == frozen_prompt
    artifact = session.scalar(select(WorkflowArtifact).where(
        WorkflowArtifact.workflow_id == workflow.id, WorkflowArtifact.kind == "chapter_coverage"
    ))
    assert artifact.payload == coverage

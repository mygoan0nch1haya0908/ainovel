from __future__ import annotations

import traceback

import pytest
from pydantic import ValidationError

from ainovel.agents.contracts import BatchPlanDraft, ChapterDraft
from ainovel.agents.runner import AgentRunner
from ainovel.providers.contracts import ModelRequest, ModelResponse, ProviderProtocolError
from ainovel.providers.demo import DemoFakeProvider
from ainovel.providers.fake import FakeProvider
from ainovel.services.counting import count_visible_characters


def chapter_request(ordinal: int) -> ModelRequest:
    return ModelRequest(
        model="demo",
        system_prompt="只返回结构化章节",
        input_payload={"ordinal": ordinal},
        output_schema=ChapterDraft.model_json_schema(),
        max_input_tokens=32000,
        max_output_tokens=12000,
        timeout_seconds=5.0,
        metadata={"schema_name": "chapter_draft", "agent_role": "chapter_writer"},
    )


def batch_plan_request() -> ModelRequest:
    return ModelRequest(
        model="scripted",
        system_prompt="只返回结构化计划",
        input_payload={"requested_chapters": 1},
        output_schema=BatchPlanDraft.model_json_schema(),
        max_input_tokens=16000,
        max_output_tokens=4000,
        timeout_seconds=5.0,
        metadata={"schema_name": "batch_plan", "agent_role": "batch_planner"},
    )


def test_runner_returns_a_validated_batch_plan() -> None:
    request = batch_plan_request()
    scripted = ModelResponse(
        structured={
            "chapters": [
                {
                    "ordinal": 1,
                    "title": "入局",
                    "goal": "主角接下委托",
                    "ending_hook": "发现追踪者",
                }
            ]
        },
        text=None,
        provider_response_id="fake-1",
        input_tokens=120,
        output_tokens=80,
        latency_ms=1,
    )
    provider = FakeProvider([scripted])

    result = AgentRunner().run(provider, request, BatchPlanDraft)

    assert result.chapters[0].ordinal == 1


def test_runner_can_return_validated_result_with_original_response_metadata() -> None:
    request = batch_plan_request()
    response = ModelResponse(
        structured={
            "chapters": [
                {
                    "ordinal": 1,
                    "title": "入局",
                    "goal": "主角接下委托",
                    "ending_hook": "发现追踪者",
                }
            ]
        },
        text=None,
        provider_response_id="provider-real-id",
        input_tokens=321,
        output_tokens=123,
        latency_ms=47,
    )

    run = AgentRunner().run_with_response(FakeProvider([response]), request, BatchPlanDraft)

    assert run.result.chapters[0].title == "入局"
    assert run.response is response
    assert run.response.input_tokens == 321
    assert run.response.output_tokens == 123
    assert run.response.latency_ms == 47
    assert run.response.provider_response_id == "provider-real-id"


def test_runner_rejects_invalid_structured_output() -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                structured={"chapters": []},
                text=None,
                provider_response_id=None,
                input_tokens=None,
                output_tokens=None,
                latency_ms=1,
            )
        ]
    )

    with pytest.raises(ProviderProtocolError):
        AgentRunner().run(provider, batch_plan_request(), BatchPlanDraft)


def test_demo_fake_can_generate_an_exact_length_chapter() -> None:
    response = DemoFakeProvider().generate(chapter_request(ordinal=3))
    draft = ChapterDraft.model_validate(response.structured)

    assert count_visible_characters(draft.body) == 4500



@pytest.mark.parametrize("length", [4499, 6001])
def test_chapter_draft_rejects_body_outside_visible_character_limits(length: int) -> None:
    with pytest.raises(ValidationError, match="4500.*6000"):
        ChapterDraft.model_validate({"title": "演示", "body": "演" * length})


def test_runner_hides_invalid_provider_payload_from_rendered_traceback() -> None:
    secret = "hidden-provider-reasoning-should-not-be-logged"
    provider = FakeProvider([
        ModelResponse(
            structured={"chapters": [], "hidden_reasoning": secret},
            text=None,
            provider_response_id=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=1,
        )
    ])

    with pytest.raises(ProviderProtocolError) as error:
        AgentRunner().run(provider, batch_plan_request(), BatchPlanDraft)

    rendered = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert secret not in rendered
def test_batch_plan_requires_contiguous_ordinals() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        BatchPlanDraft.model_validate(
            {
                "chapters": [
                    {"ordinal": 1, "title": "一", "goal": "甲", "ending_hook": "乙"},
                    {"ordinal": 3, "title": "二", "goal": "丙", "ending_hook": "丁"},
                ]
            }
        )


def test_fake_provider_copies_request_history() -> None:
    request = batch_plan_request()
    provider = FakeProvider(
        [
            ModelResponse(
                structured={"chapters": [{"ordinal": 1, "title": "一", "goal": "甲", "ending_hook": "乙"}]},
                text=None,
                provider_response_id=None,
                input_tokens=None,
                output_tokens=None,
                latency_ms=1,
            )
        ]
    )

    provider.generate(request)
    request.input_payload["requested_chapters"] = 5

    assert provider.requests[0].input_payload == {"requested_chapters": 1}


def test_demo_fake_rejects_an_unknown_agent_role() -> None:
    request = batch_plan_request()
    request.metadata["agent_role"] = "unknown"

    with pytest.raises(ProviderProtocolError, match="unknown"):
        DemoFakeProvider().generate(request)

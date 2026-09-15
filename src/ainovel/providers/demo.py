from __future__ import annotations

from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderProtocolError,
)
from ainovel.services.counting import count_visible_characters


class DemoFakeProvider:
    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128000, 16000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        role = request.metadata.get("agent_role")
        if role == "stage_planner":
            structured = self._stage_roadmap(request)
        elif role == "batch_planner":
            structured = self._batch_plan(request)
        elif role == "chapter_writer":
            structured = self._chapter_draft(request)
        elif role == "chapter_coverage_reviewer":
            structured = self._chapter_coverage(request)
        elif role == "chapter_summarizer":
            structured = {"summary": "演示章节摘要。", "state_delta": {"demo": True}}
        elif role == "batch_reviewer":
            structured = {"passed": True, "issues": [], "evidence_queries": []}
        elif role is None:
            raise ProviderProtocolError("demo fake provider requires agent_role metadata")
        else:
            raise ProviderProtocolError(f"demo fake provider does not support agent_role: {role}")
        return ModelResponse(
            structured=structured,
            text=None,
            provider_response_id="demo-fake",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )

    def diagnose(self, model: str | None = None) -> ProviderDiagnostic:
        models = (model,) if model is not None else ("demo",)
        return ProviderDiagnostic(True, "local deterministic demo provider available", models)

    @staticmethod
    def _stage_roadmap(request: ModelRequest) -> dict[str, object]:
        previous = request.input_payload.get("previous_roadmap")
        if isinstance(previous, dict):
            from copy import deepcopy
            return deepcopy(previous)
        return {"goal": "查明演示事件", "start_state": "疑点出现", "end_state": "真相公开",
                "key_events": ["调查线索", "揭露真相"], "foreshadowing": ["遗失的铜铃"],
                "nodes": [{"node_id": f"demo-{i}", "ordinal": i, "title": f"演示第{i}章",
                           "goal": f"推进演示调查第{i}步", "dependencies": [f"demo-{i-1}"] if i > 1 else []}
                          for i in range(1, 8)]}

    @staticmethod
    def _batch_plan(request: ModelRequest) -> dict[str, object]:
        requested = request.input_payload.get("requested_chapters")
        if not isinstance(requested, int) or not 1 <= requested <= 5:
            raise ProviderProtocolError("batch planner requires requested_chapters from 1 to 5")
        chapters = []
        for ordinal in range(1, requested + 1):
            chapter = {
                    "ordinal": ordinal,
                    "title": f"演示第{ordinal}章",
                    "goal": f"推进演示情节第{ordinal}步",
                    "ending_hook": f"演示悬念{ordinal}",
            }
            stage = request.input_payload.get("stage")
            if isinstance(stage, dict):
                node = stage["nodes"][ordinal - 1]
                chapter["title"], chapter["goal"] = node["title"], node["goal"]
            if request.metadata.get("generation_version") == "2":
                chapter["scenes"] = [
                    {
                        "ordinal": 1,
                        "description": f"完成演示情节第{ordinal}步",
                        "target_characters": 5200,
                    }
                ]
            chapters.append(chapter)
        return {"chapters": chapters}

    @staticmethod
    def _chapter_draft(request: ModelRequest) -> dict[str, object]:
        ordinal = request.input_payload.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ProviderProtocolError("chapter writer requires a positive ordinal")
        plan = request.input_payload.get("chapter_plan")
        if plan is None:
            goal = f"推进演示情节第{ordinal}步"
            ending_hook = f"演示悬念{ordinal}"
        elif not isinstance(plan, dict):
            raise ProviderProtocolError("chapter writer requires the approved chapter plan")
        else:
            goal = plan.get("goal")
            ending_hook = plan.get("ending_hook")
        if not isinstance(goal, str) or not isinstance(ending_hook, str):
            raise ProviderProtocolError("chapter plan requires goal and ending_hook")
        required = f"{goal}。{ending_hook}。"
        padding = 4500 - count_visible_characters(required)
        if padding < 0:
            raise ProviderProtocolError("chapter plan coverage exceeds chapter length")
        title = plan.get("title", f"演示第{ordinal}章") if isinstance(plan, dict) else f"演示第{ordinal}章"
        return {"title": title, "body": required + "演" * padding}

    @staticmethod
    def _chapter_coverage(request: ModelRequest) -> dict[str, object]:
        plan = request.input_payload.get("chapter_plan")
        draft = request.input_payload.get("draft")
        if not isinstance(plan, dict) or not isinstance(draft, dict):
            raise ProviderProtocolError(
                "chapter coverage requires plan and draft payloads"
            )
        body = draft.get("body")
        goal = plan.get("goal")
        ending_hook = plan.get("ending_hook")
        if not all(isinstance(value, str) for value in (body, goal, ending_hook)):
            raise ProviderProtocolError("chapter coverage payload is invalid")
        return {
            "goal": {"passed": goal in body, "excerpt": goal if goal in body else ""},
            "ending_hook": {
                "passed": ending_hook in body,
                "excerpt": ending_hook if ending_hook in body else "",
            },
        }

from __future__ import annotations

from ainovel.providers.contracts import (
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderProtocolError,
)


class DemoFakeProvider:
    def capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(128000, 16000, True, True, True, False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        role = request.metadata.get("agent_role")
        if role == "batch_planner":
            structured = self._batch_plan(request)
        elif role == "chapter_writer":
            structured = self._chapter_draft(request)
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
    def _batch_plan(request: ModelRequest) -> dict[str, object]:
        requested = request.input_payload.get("requested_chapters")
        if not isinstance(requested, int) or not 1 <= requested <= 5:
            raise ProviderProtocolError("batch planner requires requested_chapters from 1 to 5")
        return {
            "chapters": [
                {
                    "ordinal": ordinal,
                    "title": f"演示第{ordinal}章",
                    "goal": f"推进演示情节第{ordinal}步",
                    "ending_hook": f"演示悬念{ordinal}",
                }
                for ordinal in range(1, requested + 1)
            ]
        }

    @staticmethod
    def _chapter_draft(request: ModelRequest) -> dict[str, object]:
        ordinal = request.input_payload.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ProviderProtocolError("chapter writer requires a positive ordinal")
        return {"title": f"演示第{ordinal}章", "body": "演" * 4500}

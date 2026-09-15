from typing import Annotated

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from ainovel.agents.contracts import AgentSchema


CompactText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
NodeId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class StageChapterNode(AgentSchema):
    model_config = ConfigDict(extra="forbid", strict=True)
    node_id: NodeId
    ordinal: int = Field(ge=1)
    title: CompactText
    goal: CompactText
    dependencies: list[NodeId] = Field(max_length=100)


class StageRoadmapDraft(AgentSchema):
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: CompactText
    start_state: CompactText
    end_state: CompactText
    key_events: list[CompactText] = Field(min_length=1, max_length=100)
    foreshadowing: list[CompactText] = Field(max_length=100)
    nodes: list[StageChapterNode] = Field(min_length=1, max_length=100)

    @property
    def estimated_chapters(self) -> int:
        return len(self.nodes)

    @model_validator(mode="after")
    def ordered_dependencies(self):
        seen = set()
        for ordinal, node in enumerate(self.nodes, 1):
            if node.ordinal != ordinal or node.node_id in seen:
                raise ValueError("nodes must have unique IDs and contiguous ordering")
            if len(node.dependencies) != len(set(node.dependencies)) or not set(node.dependencies) <= seen:
                raise ValueError("dependencies must reference earlier nodes exactly once")
            seen.add(node.node_id)
        return self

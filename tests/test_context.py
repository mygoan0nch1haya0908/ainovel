from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text

from ainovel.context import (
    ITEM_FRAMING_TOKENS,
    ConservativeEstimator,
    ContextBudgeter,
    ContextCandidate,
    RequiredContextOverflow,
    effective_input_capacity,
)
from ainovel.models import (
    ContextPacketItem,
    ContextSource,
    GenerationWorkflow,
    WorkflowArtifact,
    WorkflowStep,
)
from ainovel.services.context import ContextBuilder, ContextIndexService, ContextService
from ainovel.services.projects import ProjectService


class FixedEstimator:
    def __init__(self, costs: Mapping[str, int]) -> None:
        self.costs = costs

    def estimate(self, text: str) -> int:
        return self.costs[text]


def candidate(
    text: str,
    layer: int,
    required: bool,
    relevance: int,
    temporal_distance: int = 0,
    *,
    stable_key: str | None = None,
    source_id: str | None = None,
    source_type: str = "test",
    source_version: str = "1",
    state_scope: str = "official",
    excerpt_start: int | None = None,
    excerpt_end: int | None = None,
) -> ContextCandidate:
    return ContextCandidate(
        stable_key=stable_key or text,
        layer=layer,
        text=text,
        required=required,
        relevance=relevance,
        temporal_distance=temporal_distance,
        source_id=source_id,
        source_type=source_type,
        source_version=source_version,
        state_scope=state_scope,
        excerpt_start=excerpt_start,
        excerpt_end=excerpt_end,
    )


def _add_fts(session) -> None:
    session.execute(
        text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS context_source_fts "
            "USING fts5(source_id UNINDEXED, project_id UNINDEXED, text)"
        )
    )
    session.commit()


def _workflow(session, project, official_outline, suffix: str = "one"):
    workflow = GenerationWorkflow(
        id=str(uuid4()),
        project_id=project.id,
        base_outline_version_id=official_outline.id,
        provider_name="fake",
        model_name="fake",
        requested_chapters=1,
        status="running",
        current_position=0,
        planner_input_tokens=32_000,
        planner_output_tokens=4_000,
        writer_input_tokens=32_000,
        writer_output_tokens=8_000,
        summarizer_input_tokens=16_000,
        summarizer_output_tokens=2_000,
        reviewer_input_tokens=16_000,
        reviewer_output_tokens=2_000,
    )
    step = WorkflowStep(
        id=str(uuid4()),
        workflow_id=workflow.id,
        kind=f"WRITING_{suffix}",
        ordinal=40,
        position=1,
        status="pending",
    )
    session.add_all([workflow, step])
    session.commit()
    return workflow, step


def _source(
    project_id: str,
    source_type: str,
    source_id: str,
    version: int,
    scope: str,
    layer: int,
    body: str,
) -> ContextSource:
    return ContextSource(
        id=str(uuid4()),
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        source_version=version,
        state_scope=scope,
        layer=layer,
        text=body,
        content_hash=sha256(body.encode("utf-8")).hexdigest(),
    )


def test_budget_never_trims_required_context() -> None:
    budgeter = ContextBudgeter(FixedEstimator({"constitution": 40, "old excerpt": 80}))
    packed = budgeter.pack(
        [
            candidate("constitution", layer=0, required=True, relevance=100),
            candidate("old excerpt", layer=7, required=False, relevance=1),
        ],
        input_capacity_tokens=48,
        reserved_output_tokens=8,
    )
    assert [item.text for item in packed.selected] == ["constitution"]
    assert packed.trimmed[0].reason == "budget"
    assert packed.used_tokens == 40 + ITEM_FRAMING_TOKENS


def test_required_overflow_fails_before_provider_call() -> None:
    with pytest.raises(RequiredContextOverflow) as raised:
        ContextBudgeter(FixedEstimator({"constitution": 65})).pack(
            [candidate("constitution", layer=0, required=True, relevance=100)], 64, 0
        )
    assert raised.value.stable_key == "constitution"
    assert raised.value.required_tokens == 65 + ITEM_FRAMING_TOKENS
    assert raised.value.capacity == 64


def test_fixed_overhead_can_make_required_context_overflow() -> None:
    with pytest.raises(RequiredContextOverflow) as raised:
        ContextBudgeter(FixedEstimator({"constitution": 40})).pack(
            [candidate("constitution", layer=0, required=True, relevance=100)],
            input_capacity_tokens=48,
            reserved_output_tokens=8,
            fixed_overhead_tokens=8,
        )
    assert raised.value.required_tokens == 52


def test_budget_order_is_deterministic_and_uses_temporal_distance_tiebreaker() -> None:
    items = [
        candidate("z", layer=7, required=False, relevance=1),
        candidate("older", layer=3, required=False, relevance=10, temporal_distance=2),
        candidate("newer", layer=3, required=False, relevance=10, temporal_distance=1),
        candidate("required", layer=6, required=True, relevance=0),
        candidate("a", layer=3, required=False, relevance=10, temporal_distance=1),
        candidate("high", layer=3, required=False, relevance=20),
    ]
    packed = ContextBudgeter(FixedEstimator({item.text: 1 for item in items})).pack(
        items, 100, 9
    )
    assert [item.text for item in packed.selected] == [
        "required",
        "high",
        "a",
        "newer",
        "older",
        "z",
    ]
    assert packed.reserved_output_tokens == 9


def test_budget_validates_limits_ranges_and_duplicate_keys() -> None:
    budgeter = ContextBudgeter(FixedEstimator({"x": 1, "y": 1}))
    with pytest.raises(ValueError, match="input capacity"):
        budgeter.pack([], 0, 0)
    with pytest.raises(ValueError, match="fixed overhead"):
        budgeter.pack([], 10, 0, 11)
    with pytest.raises(ValueError, match="excerpt"):
        budgeter.pack(
            [candidate("x", 7, False, 1, excerpt_start=4, excerpt_end=2)], 10, 0
        )
    with pytest.raises(ValueError, match="duplicate stable key"):
        budgeter.pack(
            [
                candidate("x", 0, False, 1, stable_key="same"),
                candidate("y", 1, False, 1, stable_key="same"),
            ],
            20,
            0,
        )


def test_capacity_and_conservative_estimation() -> None:
    assert effective_input_capacity(32_000, 40_000, 12_000, 1_024) == 26_976
    assert effective_input_capacity(32_000, 128_000, 12_000, 1_024) == 32_000
    with pytest.raises(ValueError, match="nonpositive"):
        effective_input_capacity(32_000, 12_000, 12_000, 1_024)
    estimator = ConservativeEstimator()
    assert estimator.estimate("") == 1
    assert estimator.estimate("abcdefg") == 3


def test_official_rebuild_is_project_scoped_idempotent_and_keeps_candidates(
    session, project, official_outline, approved_chapter
) -> None:
    _add_fts(session)
    constitution = ProjectService(session).add_constitution(
        project.id, {"prohibitions": ["no resurrection"]}, True
    )
    stale = _source(project.id, "outline_node", "obsolete", 1, "official", 2, "old")
    candidate_row = _source(
        project.id,
        "chapter_summary_delta",
        "candidate",
        1,
        "workflow:kept",
        6,
        "candidate survives",
    )
    other_project = ProjectService(session).create("other", 1, 2)
    other_official = _source(
        other_project.id, "outline_node", "other", 1, "official", 2, "other survives"
    )
    session.add_all([stale, candidate_row, other_official])
    session.flush()
    for row in (stale, candidate_row, other_official):
        session.execute(
            text(
                "INSERT INTO context_source_fts(source_id, project_id, text) "
                "VALUES (:source_id, :project_id, :body)"
            ),
            {"source_id": row.id, "project_id": row.project_id, "body": row.text},
        )
    session.commit()

    service = ContextIndexService(session)
    first_count = service.rebuild_official(project.id)
    second_count = service.rebuild_official(project.id)

    official_rows = session.scalars(
        select(ContextSource).where(
            ContextSource.project_id == project.id,
            ContextSource.state_scope == "official",
        )
    ).all()
    assert first_count == second_count == 3
    assert {row.source_id for row in official_rows} == {
        constitution.id,
        official_outline.id + ":book",
        approved_chapter.id,
    }
    assert session.get(ContextSource, candidate_row.id) is not None
    assert session.get(ContextSource, other_official.id) is not None
    mirrors = session.execute(
        text("SELECT source_id, project_id, text FROM context_source_fts")
    ).all()
    assert len([row for row in mirrors if row.project_id == project.id]) == 4
    assert stale.id not in {row.source_id for row in mirrors}


def test_fts_search_escapes_untrusted_syntax_filters_types_and_orders_stably(
    session, project
) -> None:
    _add_fts(session)
    first = _source(project.id, "world_rule", "b", 1, "official", 1, "alpha target")
    second = _source(project.id, "world_rule", "a", 1, "official", 1, "alpha target")
    wrong_type = _source(project.id, "character", "c", 1, "official", 5, "alpha target")
    candidate_row = _source(
        project.id, "world_rule", "candidate", 1, "workflow:x", 1, "alpha target"
    )
    session.add_all([first, second, wrong_type, candidate_row])
    session.flush()
    for row in (first, second, wrong_type, candidate_row):
        session.execute(
            text(
                "INSERT INTO context_source_fts(source_id, project_id, text) "
                "VALUES (:source_id, :project_id, :body)"
            ),
            {"source_id": row.id, "project_id": row.project_id, "body": row.text},
        )
    session.commit()

    service = ContextIndexService(session)
    assert [row.source_id for row in service.search(project.id, "alpha", {"world_rule"}, 10)] == [
        "a",
        "b",
    ]
    assert service.search(project.id, 'alpha" OR target', {"world_rule"}, 10) == []
    assert service.search(project.id, "\" OR * NOT (", {"world_rule"}, 10) == []
    assert service.search(
        project.id, " ".join(["alpha"] * 5_000), {"world_rule"}, 10
    ) == []


def test_workflow_artifact_indexing_is_hash_versioned_and_preserves_excerpt_offsets(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    digest = sha256(b"remember this exact line").hexdigest()
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=39,
        text_content="remember this exact line",
        payload={"explicitly_requested": True, "excerpt_start": 100, "excerpt_end": 124},
        visible_char_count=24,
        content_hash=digest,
    )
    session.add(artifact)
    session.commit()

    indexed = ContextIndexService(session).index_workflow_artifact(artifact.id)
    indexed_again = ContextIndexService(session).index_workflow_artifact(artifact.id)
    candidates = ContextBuilder(session).candidates_for_step(workflow.id, step.id)

    assert indexed_again.id == indexed.id
    assert indexed.content_hash == digest
    assert indexed.state_scope == f"workflow:{workflow.id}"
    assert indexed.source_type == "historical_excerpt"
    selected = next(item for item in candidates if item.source_id == indexed.id)
    assert selected.source_version == digest
    assert (selected.excerpt_start, selected.excerpt_end) == (100, 124)


def test_workflow_artifact_indexing_rejects_unvalidated_or_unrequested_content(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="chapter_draft",
        ordinal=1,
        text_content="draft must not enter retrieval",
        payload={},
        visible_char_count=30,
        content_hash="f" * 64,
    )
    session.add(artifact)
    session.commit()
    with pytest.raises(ValueError, match="not indexable"):
        ContextIndexService(session).index_workflow_artifact(artifact.id)


def test_builder_maps_l0_through_l7_with_exact_scope_versions_and_limits(
    session, project, official_outline
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    other_workflow, _ = _workflow(session, project, official_outline, "two")
    layer_types = {
        0: "constitution",
        1: "world_rule",
        2: "current_stage_goal",
        5: "critical_character_state",
        6: "approved_batch_plan",
        7: "historical_excerpt",
    }
    rows = [
        _source(project.id, source_type, f"layer-{layer}", 1, "official", 99, f"L{layer}")
        for layer, source_type in layer_types.items()
        if layer != 6
    ]
    rows.append(
        _source(
            project.id,
            "approved_batch_plan",
            "layer-6",
            1,
            f"workflow:{workflow.id}",
            99,
            "L6",
        )
    )
    rows.append(
        _source(
            project.id,
            "approved_batch_plan",
            "wrong-workflow",
            1,
            f"workflow:{other_workflow.id}",
            6,
            "must not leak",
        )
    )
    rows.extend(
        _source(project.id, "event_chain", f"event-{index:02}", index, "official", 3, f"E{index}")
        for index in range(1, 36)
    )
    rows.extend(
        _source(
            project.id,
            "chapter_summary",
            f"summary-{index:02}",
            index,
            "official",
            4,
            f"S{index}",
        )
        for index in range(1, 8)
    )
    rows.append(_source(project.id, "world_rule", "versioned", 1, "official", 1, "old"))
    rows.append(_source(project.id, "world_rule", "versioned", 2, "official", 1, "new"))
    other_project = ProjectService(session).create("isolated", 1, 2)
    rows.append(_source(other_project.id, "constitution", "foreign", 1, "official", 0, "leak"))
    session.add_all(rows)
    session.commit()

    candidates = ContextBuilder(session).candidates_for_step(workflow.id, step.id)

    assert {item.layer for item in candidates} == set(range(8))
    assert sum(item.layer == 3 for item in candidates) == 30
    assert sum(item.layer == 4 for item in candidates) == 5
    assert {item.text for item in candidates if item.layer == 3} == {f"E{i}" for i in range(6, 36)}
    assert {item.text for item in candidates if item.layer == 4} == {f"S{i}" for i in range(3, 8)}
    assert "old" not in {item.text for item in candidates}
    assert "new" in {item.text for item in candidates}
    assert "must not leak" not in {item.text for item in candidates}
    assert "leak" not in {item.text for item in candidates}
    required_types = {item.source_type for item in candidates if item.required}
    assert {
        "constitution",
        "current_stage_goal",
        "critical_character_state",
        "approved_batch_plan",
    } <= required_types


def test_builder_rejects_a_step_from_another_workflow(
    session, project, official_outline
) -> None:
    workflow, _ = _workflow(session, project, official_outline)
    _, other_step = _workflow(session, project, official_outline, "other")
    with pytest.raises(ValueError, match="step does not belong"):
        ContextBuilder(session).candidates_for_step(workflow.id, other_step.id)


def test_packet_persists_selected_trimmed_snapshots_and_deduplicates_overlap(
    session, project, official_outline
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    source = _source(project.id, "constitution", "constitution", 1, "official", 0, "required")
    session.add(source)
    session.commit()
    required = candidate(
        "required",
        0,
        True,
        100,
        source_id=source.id,
        source_type="constitution",
        source_version=source.content_hash,
    )
    duplicate_optional = candidate(
        "stale duplicate",
        0,
        False,
        1,
        stable_key="required",
        source_id=source.id,
    )
    trimmed = candidate(
        "trimmed",
        7,
        False,
        1,
        temporal_distance=9,
        excerpt_start=20,
        excerpt_end=27,
    )
    service = ContextService(
        session,
        ContextBudgeter(FixedEstimator({"required": 10, "stale duplicate": 1, "trimmed": 30})),
    )

    packet = service.build_packet(
        workflow.id,
        step.id,
        [required],
        [duplicate_optional, trimmed],
        {
            "input_capacity_tokens": 24,
            "reserved_output_tokens": 8,
            "fixed_overhead_tokens": 4,
        },
    )

    items = session.scalars(
        select(ContextPacketItem)
        .where(ContextPacketItem.packet_id == packet.id)
        .order_by(ContextPacketItem.position)
    ).all()
    assert packet.used_input_tokens == 18
    assert packet.fixed_overhead_tokens == 4
    assert len(items) == 2
    assert (
        items[0].text_snapshot,
        items[0].selected,
        items[0].trim_reason,
        items[0].estimated_tokens,
    ) == ("required", True, None, 14)
    assert (
        items[1].text_snapshot,
        items[1].selected,
        items[1].trim_reason,
        items[1].temporal_distance,
        items[1].excerpt_start,
        items[1].excerpt_end,
    ) == ("trimmed", False, "budget", 9, 20, 27)
    assert session.scalar(
        select(func.count()).select_from(ContextPacketItem).where(
            ContextPacketItem.packet_id == packet.id
        )
    ) == 2


def test_packet_required_overflow_persists_nothing(
    session, project, official_outline
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    service = ContextService(
        session, ContextBudgeter(FixedEstimator({"constitution": 40}))
    )
    with pytest.raises(RequiredContextOverflow):
        service.build_packet(
            workflow.id,
            step.id,
            [candidate("constitution", 0, True, 100)],
            [],
            {
                "input_capacity_tokens": 48,
                "reserved_output_tokens": 8,
                "fixed_overhead_tokens": 8,
            },
        )
    assert session.scalar(text("SELECT count(*) FROM context_packets")) == 0

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
        source_version=str(version),
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
    canonical_text = "xxremember this exact lineyy"
    canonical = _source(
        project.id, "official_chapter", "chapter", 1, "official", 7, canonical_text
    )
    session.add(canonical)
    session.commit()
    digest = sha256(b"remember this exact line").hexdigest()
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=39,
        text_content="remember this exact line",
        payload={
            "explicitly_requested": True,
            "canonical_source_type": "official_chapter",
            "canonical_source_id": canonical.source_id,
            "excerpt_start": 2,
            "excerpt_end": 26,
        },
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
    assert indexed.source_version == digest
    assert indexed.state_scope == f"workflow:{workflow.id}"
    assert indexed.source_type == "historical_excerpt"
    selected = next(item for item in candidates if item.source_id == indexed.id)
    assert selected.source_version == digest
    assert (selected.excerpt_start, selected.excerpt_end) == (2, 26)


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
    plan_text = "L6"
    plan_artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="approved_batch_plan",
        ordinal=1,
        text_content=plan_text,
        payload={"chapters": []},
        visible_char_count=len(plan_text),
        content_hash=sha256(plan_text.encode("utf-8")).hexdigest(),
    )
    step.active_artifact_id = plan_artifact.id
    step.status = "validated"
    session.add(plan_artifact)
    canonical_l7 = _source(
        project.id, "official_chapter", "canonical-l7", 1, "official", 7, "L7"
    )
    excerpt_artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=1,
        text_content="L7",
        payload={
            "explicitly_requested": True,
            "canonical_source_type": "official_chapter",
            "canonical_source_id": canonical_l7.source_id,
            "excerpt_start": 0,
            "excerpt_end": 2,
        },
        visible_char_count=2,
        content_hash=sha256(b"L7").hexdigest(),
    )
    session.add_all([canonical_l7, excerpt_artifact])
    layer_types = {
        0: "constitution",
        1: "world_rule",
        2: "current_stage_goal",
        5: "critical_character_state",
        6: "approved_batch_plan",
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
            plan_artifact.id,
            plan_artifact.content_hash,
            f"workflow:{workflow.id}",
            99,
            plan_text,
        )
    )
    indexed_excerpt = _source(
        project.id,
        "historical_excerpt",
        excerpt_artifact.id,
        excerpt_artifact.content_hash,
        f"workflow:{workflow.id}",
        7,
        "L7",
    )
    indexed_excerpt.explicitly_requested = True
    indexed_excerpt.canonical_source_type = "official_chapter"
    indexed_excerpt.canonical_source_id = canonical_l7.source_id
    indexed_excerpt.excerpt_start = 0
    indexed_excerpt.excerpt_end = 2
    rows.append(indexed_excerpt)
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
    _add_fts(session)
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
        stable_key="constitution:constitution",
        source_type="constitution",
        source_version="1",
    )
    duplicate_optional = candidate(
        "stale duplicate",
        0,
        False,
        1,
        stable_key="constitution:constitution",
        source_id=source.id,
    )
    trimmed = candidate(
        "trimmed",
        6,
        False,
        1,
        temporal_distance=9,
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
    ) == ("trimmed", False, "budget", 9, None, None)
    assert session.scalar(
        select(func.count()).select_from(ContextPacketItem).where(
            ContextPacketItem.packet_id == packet.id
        )
    ) == 2

    ContextIndexService(session).rebuild_official(project.id)
    session.expire_all()
    items = session.scalars(
        select(ContextPacketItem)
        .where(ContextPacketItem.packet_id == packet.id)
        .order_by(ContextPacketItem.position)
    ).all()
    assert (
        items[0].source_id,
        items[0].source_type,
        items[0].source_version,
        items[0].state_scope,
        items[0].source_content_hash,
    ) == (None, "constitution", "1", "official", source.content_hash)
    assert (
        items[1].source_id,
        items[1].source_type,
        items[1].source_version,
        items[1].state_scope,
        items[1].source_content_hash,
    ) == (
        None,
        "test",
        "1",
        "official",
        sha256(b"trimmed").hexdigest(),
    )


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


@pytest.mark.parametrize("chapter_count", [1, 5])
def test_packet_rejects_raw_full_official_chapter_candidates_even_when_they_fit(
    session, project, official_outline, chapter_count
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    rows = [
        _source(
            project.id,
            "official_chapter",
            f"chapter-{number}",
            1,
            "official",
            7,
            "章名\n" + "甲" * 4_500,
        )
        for number in range(chapter_count)
    ]
    session.add_all(rows)
    session.commit()
    candidates = [
        candidate(
            row.text,
            7,
            False,
            10,
            stable_key=f"official_chapter:{row.source_id}",
            source_id=row.id,
            source_type="official_chapter",
            source_version="1",
        )
        for row in rows
    ]

    with pytest.raises(ValueError, match="bounded canonical excerpt"):
        ContextService(session).build_packet(
            workflow.id,
            step.id,
            [],
            candidates,
            {"input_capacity_tokens": 100_000, "reserved_output_tokens": 0},
        )
    assert session.scalar(text("SELECT count(*) FROM context_packets")) == 0


def test_exact_explicit_bounded_canonical_excerpt_can_be_packed(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    canonical_text = "prefix:" + "甲" * 1_300 + ":suffix"
    canonical = _source(
        project.id,
        "official_chapter",
        "chapter-1",
        1,
        "official",
        7,
        canonical_text,
    )
    session.add(canonical)
    session.commit()
    start, end = 7, 1_207
    excerpt = canonical_text[start:end]
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="review_evidence_excerpt",
        ordinal=39,
        text_content=excerpt,
        payload={
            "explicitly_requested": True,
            "canonical_source_type": "official_chapter",
            "canonical_source_id": canonical.source_id,
            "excerpt_start": start,
            "excerpt_end": end,
        },
        visible_char_count=1_200,
        content_hash=sha256(excerpt.encode("utf-8")).hexdigest(),
    )
    session.add(artifact)
    session.commit()

    indexed = ContextIndexService(session).index_workflow_artifact(artifact.id)
    assert indexed.explicitly_requested is True
    assert indexed.canonical_source_type == "official_chapter"
    assert indexed.canonical_source_id == canonical.source_id
    assert (indexed.excerpt_start, indexed.excerpt_end) == (start, end)
    excerpt_candidate = next(
        item
        for item in ContextBuilder(session).candidates_for_step(workflow.id, step.id)
        if item.source_id == indexed.id
    )
    packet = ContextService(session).build_packet(
        workflow.id,
        step.id,
        [],
        [excerpt_candidate],
        {"input_capacity_tokens": 2_000, "reserved_output_tokens": 0},
    )
    stored = session.scalar(
        select(ContextPacketItem).where(ContextPacketItem.packet_id == packet.id)
    )
    assert stored is not None
    assert stored.explicitly_requested is True
    assert stored.canonical_source_type == "official_chapter"
    assert stored.canonical_source_id == canonical.source_id
    assert stored.text_snapshot == canonical_text[start:end]
    assert (stored.excerpt_start, stored.excerpt_end) == (start, end)


@pytest.mark.parametrize(
    ("payload_update", "text_update"),
    [
        ({"explicitly_requested": False}, None),
        ({"canonical_source_id": None}, None),
        ({"excerpt_start": None}, None),
        ({"excerpt_end": 1_208}, None),
        ({"excerpt_end": 1_209}, "甲" * 1_202),
        ({}, "caller supplied text"),
    ],
)
def test_excerpt_indexing_rejects_missing_forged_or_oversized_provenance(
    session, project, official_outline, payload_update, text_update
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    canonical_text = "prefix:" + "甲" * 1_300
    canonical = _source(
        project.id, "official_chapter", "chapter", 1, "official", 7, canonical_text
    )
    session.add(canonical)
    session.commit()
    payload = {
        "explicitly_requested": True,
        "canonical_source_type": "official_chapter",
        "canonical_source_id": canonical.source_id,
        "excerpt_start": 7,
        "excerpt_end": 1_207,
    }
    payload.update(payload_update)
    body = canonical_text[7:1_207] if text_update is None else text_update
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=39,
        text_content=body,
        payload=payload,
        visible_char_count=len(body),
        content_hash=sha256(body.encode("utf-8")).hexdigest(),
    )
    session.add(artifact)
    session.commit()
    with pytest.raises(ValueError, match="canonical excerpt"):
        ContextIndexService(session).index_workflow_artifact(artifact.id)


@pytest.mark.parametrize("invalid_ownership", ["project", "scope"])
def test_excerpt_indexing_rejects_nonofficial_or_cross_project_canonical_source(
    session, project, official_outline, invalid_ownership
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    source_project = project
    if invalid_ownership == "project":
        source_project = ProjectService(session).create("other canonical", 1, 2)
    canonical = _source(
        source_project.id,
        "official_chapter",
        "foreign",
        1,
        "workflow:unapproved" if invalid_ownership == "scope" else "official",
        7,
        "甲" * 100,
    )
    session.add(canonical)
    session.commit()
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=1,
        text_content="甲" * 10,
        payload={
            "explicitly_requested": True,
            "canonical_source_type": "official_chapter",
            "canonical_source_id": "foreign",
            "excerpt_start": 0,
            "excerpt_end": 10,
        },
        visible_char_count=10,
        content_hash=sha256(("甲" * 10).encode("utf-8")).hexdigest(),
    )
    session.add(artifact)
    session.commit()
    with pytest.raises(ValueError, match="canonical excerpt"):
        ContextIndexService(session).index_workflow_artifact(artifact.id)


def test_summary_artifact_requires_completed_active_ownership(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    body = "validated summary"
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="chapter_summary",
        ordinal=1,
        text_content=body,
        payload={"summary": body},
        visible_char_count=len(body),
        content_hash=sha256(body.encode("utf-8")).hexdigest(),
    )
    session.add(artifact)
    session.commit()
    with pytest.raises(ValueError, match="accepted active artifact"):
        ContextIndexService(session).index_workflow_artifact(artifact.id)
    step.active_artifact_id = artifact.id
    step.status = "validated"
    session.commit()
    assert ContextIndexService(session).index_workflow_artifact(artifact.id).source_id == artifact.id


def test_accepted_artifact_rejects_a_content_hash_that_does_not_match_indexed_text(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="chapter_summary",
        ordinal=1,
        text_content="summary",
        payload={"summary": "summary"},
        visible_char_count=7,
        content_hash="f" * 64,
    )
    step.active_artifact_id = artifact.id
    step.status = "completed"
    session.add(artifact)
    session.commit()
    with pytest.raises(ValueError, match="content hash"):
        ContextIndexService(session).index_workflow_artifact(artifact.id)


def test_approved_plan_is_only_admitted_from_exact_accepted_workflow_scope(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    official_plan = _source(
        project.id, "approved_batch_plan", "official-plan", 1, "official", 6, "leak"
    )
    session.add(official_plan)
    body = "accepted plan"
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="approved_batch_plan",
        ordinal=1,
        text_content=body,
        payload={"chapters": []},
        visible_char_count=len(body),
        content_hash=sha256(body.encode("utf-8")).hexdigest(),
    )
    step.active_artifact_id = artifact.id
    step.status = "completed"
    session.add(artifact)
    session.commit()
    indexed = ContextIndexService(session).index_workflow_artifact(artifact.id)

    candidates = ContextBuilder(session).candidates_for_step(workflow.id, step.id)
    assert indexed.id in {item.source_id for item in candidates}
    assert "leak" not in {item.text for item in candidates}


@pytest.mark.parametrize(
    "mismatch",
    [
        "cross_project",
        "missing",
        "type",
        "version",
        "scope",
        "stable_key",
        "text",
        "hash",
    ],
)
def test_packet_rejects_invalid_linked_source_and_rolls_back(
    session, project, official_outline, mismatch
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    source_project = project
    if mismatch == "cross_project":
        source_project = ProjectService(session).create("foreign", 1, 2)
    row = _source(
        source_project.id, "world_rule", "rule", 1, "official", 1, "canonical"
    )
    if mismatch == "hash":
        row.content_hash = "0" * 64
    session.add(row)
    session.commit()
    source_id = "00000000-0000-0000-0000-000000000000" if mismatch == "missing" else row.id
    item = candidate(
        "forged" if mismatch == "text" else row.text,
        1,
        False,
        10,
        stable_key=(
            "world_rule:forged"
            if mismatch == "stable_key"
            else f"world_rule:{row.source_id}"
        ),
        source_id=source_id,
        source_type="character" if mismatch == "type" else "world_rule",
        source_version="2" if mismatch == "version" else "1",
        state_scope="workflow:wrong" if mismatch == "scope" else "official",
    )
    with pytest.raises(ValueError, match="linked context source"):
        ContextService(session).build_packet(
            workflow.id,
            step.id,
            [],
            [item],
            {"input_capacity_tokens": 100, "reserved_output_tokens": 0},
        )
    assert not session.new
    assert session.scalar(text("SELECT count(*) FROM context_packets")) == 0


def test_packet_rejects_nullable_injected_candidate_claiming_another_workflow(
    session, project, official_outline
) -> None:
    workflow, step = _workflow(session, project, official_outline)
    injected = candidate(
        "injected",
        5,
        False,
        1,
        state_scope="workflow:another",
    )
    with pytest.raises(ValueError, match="well-formed non-L7"):
        ContextService(session).build_packet(
            workflow.id,
            step.id,
            [],
            [injected],
            {"input_capacity_tokens": 100, "reserved_output_tokens": 0},
        )
    assert not session.new


def test_changed_artifact_hash_clears_old_packet_fk_but_keeps_snapshot(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    first_text = "first summary"
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="chapter_summary",
        ordinal=1,
        text_content=first_text,
        payload={"summary": first_text},
        visible_char_count=len(first_text),
        content_hash=sha256(first_text.encode("utf-8")).hexdigest(),
    )
    step.active_artifact_id = artifact.id
    step.status = "validated"
    session.add(artifact)
    session.commit()
    first_source = ContextIndexService(session).index_workflow_artifact(artifact.id)
    item = next(
        value
        for value in ContextBuilder(session).candidates_for_step(workflow.id, step.id)
        if value.source_id == first_source.id
    )
    packet = ContextService(session).build_packet(
        workflow.id,
        step.id,
        [],
        [item],
        {"input_capacity_tokens": 100, "reserved_output_tokens": 0},
    )
    artifact.text_content = "second summary"
    artifact.payload = {"summary": "second summary"}
    artifact.content_hash = sha256(b"second summary").hexdigest()
    session.commit()

    second_source = ContextIndexService(session).index_workflow_artifact(artifact.id)
    stored = session.scalar(
        select(ContextPacketItem).where(ContextPacketItem.packet_id == packet.id)
    )
    assert second_source.id != first_source.id
    assert stored is not None
    assert stored.source_id is None
    assert stored.text_snapshot == first_text
    assert stored.source_version == sha256(first_text.encode("utf-8")).hexdigest()
    assert stored.source_content_hash == sha256(first_text.encode("utf-8")).hexdigest()


def test_same_text_excerpt_reindex_replaces_stale_canonical_provenance(
    session, project, official_outline
) -> None:
    _add_fts(session)
    workflow, step = _workflow(session, project, official_outline)
    body = "identical excerpt"
    first_canonical = _source(
        project.id, "official_chapter", "chapter-a", 1, "official", 7, body
    )
    second_canonical = _source(
        project.id, "official_chapter", "chapter-b", 1, "official", 7, body
    )
    session.add_all([first_canonical, second_canonical])
    session.commit()
    digest = sha256(body.encode("utf-8")).hexdigest()
    artifact = WorkflowArtifact(
        id=str(uuid4()),
        workflow_id=workflow.id,
        step_id=step.id,
        kind="requested_excerpt",
        ordinal=1,
        text_content=body,
        payload={
            "explicitly_requested": True,
            "canonical_source_type": "official_chapter",
            "canonical_source_id": first_canonical.source_id,
            "excerpt_start": 0,
            "excerpt_end": len(body),
        },
        visible_char_count=len(body),
        content_hash=digest,
    )
    session.add(artifact)
    session.commit()
    first_indexed = ContextIndexService(session).index_workflow_artifact(artifact.id)
    item = next(
        value
        for value in ContextBuilder(session).candidates_for_step(workflow.id, step.id)
        if value.source_id == first_indexed.id
    )
    packet = ContextService(session).build_packet(
        workflow.id,
        step.id,
        [],
        [item],
        {"input_capacity_tokens": 100, "reserved_output_tokens": 0},
    )

    artifact.payload = {
        **artifact.payload,
        "canonical_source_id": second_canonical.source_id,
    }
    session.commit()
    second_indexed = ContextIndexService(session).index_workflow_artifact(artifact.id)
    stored = session.scalar(
        select(ContextPacketItem).where(ContextPacketItem.packet_id == packet.id)
    )

    assert second_indexed.id != first_indexed.id
    assert second_indexed.canonical_source_id == second_canonical.source_id
    assert stored is not None
    assert stored.source_id is None
    assert stored.canonical_source_id == first_canonical.source_id
    assert stored.text_snapshot == body
    assert stored.source_content_hash == digest

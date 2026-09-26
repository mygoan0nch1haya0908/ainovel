from copy import deepcopy

import pytest


def plot_payload(counts=(2, 4)):
    return {
        'format': 'plot_points_v1', 'goal': '查明真相', 'start_state': '入城',
        'end_state': '破案', 'key_events': ['调查'], 'foreshadowing': [],
        'points': [dict(point_id=f'p{i}', ordinal=i, title=f'剧情{i}', goal=f'目标{i}',
                        key_events=['发现线索'], character_changes=[], foreshadowing=[],
                        chapter_count=count, dependencies=[f'p{i-1}'] if i > 1 else [])
                   for i, count in enumerate(counts, 1)],
    }


def test_plot_ranges_and_cross_boundary_slots():
    from ainovel.services.stage_planning import chapter_slots, plot_point_ranges, roadmap_chapter_count
    payload = plot_payload()
    assert roadmap_chapter_count(payload) == 6
    assert plot_point_ranges(payload) == [dict(point_id='p1', start=1, end=2), dict(point_id='p2', start=3, end=6)]
    slots = chapter_slots(payload, 1, 5)
    assert [s['stage_ordinal'] for s in slots] == [2, 3, 4, 5, 6]
    assert [s['point_ordinal'] for s in slots] == [2, 1, 2, 3, 4]
    assert [s['node_id'] for s in slots] == ['p1:2', 'p2:1', 'p2:2', 'p2:3', 'p2:4']
    assert len(chapter_slots(payload, 5, 5)) == 1
    assert chapter_slots(payload, 6, 5) == []


@pytest.mark.parametrize('counts', [(0,), (-1,), (1.5,), ('2',), (True,), (101,), (100,)*5+(1,), (1,)*31])
def test_reject_invalid_chapter_allocations(counts):
    from ainovel.services.stage_planning import parse_stage_roadmap
    with pytest.raises(ValueError):
        parse_stage_roadmap(plot_payload(counts))


@pytest.mark.parametrize('field,value', [('point_id','p1'), ('ordinal',1), ('dependencies',['p2']), ('dependencies',['p1','p1'])])
def test_reject_invalid_point_graph(field, value):
    from ainovel.services.stage_planning import parse_stage_roadmap
    payload = plot_payload()
    payload['points'][1][field] = value
    with pytest.raises(ValueError):
        parse_stage_roadmap(payload)


@pytest.mark.parametrize('confirmed,requested', [(-1,5),(0,0),(0,6),(True,1),(0,True)])
def test_reject_invalid_slot_request(confirmed, requested):
    from ainovel.services.stage_planning import chapter_slots
    with pytest.raises(ValueError):
        chapter_slots(plot_payload(), confirmed, requested)


def test_lock_whole_started_point_and_allow_unstarted_change():
    from ainovel.services.stage_planning import validate_locked_prefix
    original = plot_payload()
    changed = deepcopy(original)
    changed['points'][0]['chapter_count'] = 3
    with pytest.raises(ValueError):
        validate_locked_prefix(original, changed, 1)
    changed = deepcopy(original)
    changed['points'][1]['chapter_count'] = 6
    validate_locked_prefix(original, changed, 1)


def test_legacy_and_unknown_format():
    from ainovel.services.stage_planning import parse_stage_roadmap, chapter_slots, validate_locked_prefix
    from test_stages import roadmap_payload
    old = roadmap_payload(2)
    assert parse_stage_roadmap(old).model_dump() == old
    assert chapter_slots(old, 1, 5) == [dict(node_id='node-2', stage_ordinal=2)]
    with pytest.raises(ValueError):
        parse_stage_roadmap({**plot_payload(), 'format': 'future'})
    with pytest.raises(ValueError):
        validate_locked_prefix(old, plot_payload(), 1)
    validate_locked_prefix(old, plot_payload(), 0)

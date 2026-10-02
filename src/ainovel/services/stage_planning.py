"""Pure, version-aware stage allocation. Never rewrite historical payloads."""
from ainovel.agents.stage_contracts import StagePlotRoadmapDraft, StageRoadmapDraft

PLOT_FORMAT = 'plot_points_v1'


def parse_stage_roadmap(payload: dict) -> StageRoadmapDraft | StagePlotRoadmapDraft:
    if 'format' not in payload:
        return StageRoadmapDraft.model_validate(payload)
    if payload['format'] == PLOT_FORMAT:
        return StagePlotRoadmapDraft.model_validate(payload)
    raise ValueError('unknown roadmap format')


def roadmap_chapter_count(payload: dict) -> int:
    return parse_stage_roadmap(payload).estimated_chapters


def plot_point_ranges(payload: dict) -> list[dict]:
    draft = parse_stage_roadmap(payload)
    if not isinstance(draft, StagePlotRoadmapDraft):
        return []
    result, start = [], 1
    for point in draft.points:
        end = start + point.chapter_count - 1
        result.append(dict(point_id=point.point_id, start=start, end=end))
        start = end + 1
    return result


def chapter_slots(payload: dict, confirmed: int, requested: int) -> list[dict]:
    if type(confirmed) is not int or confirmed < 0 or type(requested) is not int or not 1 <= requested <= 5:
        raise ValueError('invalid chapter slot request')
    draft = parse_stage_roadmap(payload)
    if isinstance(draft, StageRoadmapDraft):
        return [dict(node_id=n.node_id, stage_ordinal=n.ordinal) for n in draft.nodes[confirmed:confirmed + requested]]
    result = []
    for span in plot_point_ranges(payload):
        for number in range(max(confirmed + 1, span['start']), min(confirmed + requested, span['end']) + 1):
            local = number - span['start'] + 1
            result.append(dict(node_id=f"{span['point_id']}:{local}", point_id=span['point_id'],
                               point_ordinal=local, stage_ordinal=number))
    return result


def validate_locked_prefix(previous: dict, candidate: dict, confirmed: int) -> None:
    old, new = parse_stage_roadmap(previous), parse_stage_roadmap(candidate)
    if confirmed == 0:
        return
    if type(old) is not type(new):
        raise ValueError('cannot change roadmap format after confirmed chapters')
    if isinstance(old, StageRoadmapDraft):
        if old.nodes[:confirmed] != new.nodes[:confirmed]:
            raise ValueError('revision cannot change confirmed nodes')
    else:
        locked = sum(span['start'] <= confirmed for span in plot_point_ranges(previous))
        if old.points[:locked] != new.points[:locked]:
            raise ValueError('revision cannot change started plot points / 已开始的剧情点不能修改')

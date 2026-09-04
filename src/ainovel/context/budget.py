from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Protocol


ITEM_FRAMING_TOKENS = 4


@dataclass(frozen=True)
class ContextCandidate:
    stable_key: str
    layer: int
    text: str
    required: bool
    relevance: int
    temporal_distance: int
    source_id: str | None
    source_type: str
    source_version: str
    state_scope: str
    excerpt_start: int | None
    excerpt_end: int | None


@dataclass(frozen=True)
class TrimmedContext:
    item: ContextCandidate
    reason: str


class RequiredContextOverflow(ValueError):
    def __init__(self, stable_key: str, required_tokens: int, capacity: int) -> None:
        self.stable_key = stable_key
        self.required_tokens = required_tokens
        self.capacity = capacity
        super().__init__(
            f"required context {stable_key!r} needs {required_tokens} tokens "
            f"but capacity is {capacity}"
        )


@dataclass(frozen=True)
class PackedContext:
    selected: tuple[ContextCandidate, ...]
    trimmed: tuple[TrimmedContext, ...]
    used_tokens: int
    max_input_tokens: int
    reserved_output_tokens: int


class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


class ConservativeEstimator:
    """Estimate content tokens; the budgeter adds item framing separately."""

    def estimate(self, text: str) -> int:
        return max(1, ceil(len(text.encode("utf-8")) / 3))


def effective_input_capacity(
    configured_input_tokens: int,
    provider_context_window: int,
    reserved_output_tokens: int,
    safety_tokens: int = 1024,
) -> int:
    if configured_input_tokens <= 0 or provider_context_window <= 0:
        raise ValueError("token limits must be positive")
    if reserved_output_tokens < 0 or safety_tokens < 0:
        raise ValueError("reserved and safety tokens must be nonnegative")
    capacity = min(
        configured_input_tokens,
        provider_context_window - reserved_output_tokens - safety_tokens,
    )
    if capacity <= 0:
        raise ValueError("effective input capacity is nonpositive")
    return capacity


class ContextBudgeter:
    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self.estimator = estimator or ConservativeEstimator()

    def pack(
        self,
        candidates: list[ContextCandidate] | tuple[ContextCandidate, ...],
        input_capacity_tokens: int,
        reserved_output_tokens: int,
        fixed_overhead_tokens: int = 0,
    ) -> PackedContext:
        if input_capacity_tokens <= 0:
            raise ValueError("input capacity must be positive")
        if reserved_output_tokens < 0:
            raise ValueError("reserved output tokens must be nonnegative")
        if fixed_overhead_tokens < 0 or fixed_overhead_tokens > input_capacity_tokens:
            raise ValueError("fixed overhead must fit within input capacity")

        snapshot = tuple(candidates)
        self._validate_candidates(snapshot)
        ordered = sorted(
            snapshot,
            key=lambda item: (
                not item.required,
                item.layer,
                -item.relevance,
                item.temporal_distance,
                item.stable_key,
            ),
        )
        selected: list[ContextCandidate] = []
        trimmed: list[TrimmedContext] = []
        used = fixed_overhead_tokens
        for item in ordered:
            cost = self.estimator.estimate(item.text) + ITEM_FRAMING_TOKENS
            if cost <= ITEM_FRAMING_TOKENS:
                raise ValueError("token estimator must return a positive cost")
            if item.required and used + cost > input_capacity_tokens:
                raise RequiredContextOverflow(
                    item.stable_key, used + cost, input_capacity_tokens
                )
            if used + cost <= input_capacity_tokens:
                selected.append(item)
                used += cost
            else:
                trimmed.append(TrimmedContext(item=item, reason="budget"))
        return PackedContext(
            selected=tuple(selected),
            trimmed=tuple(trimmed),
            used_tokens=used,
            max_input_tokens=input_capacity_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

    @staticmethod
    def _validate_candidates(candidates: tuple[ContextCandidate, ...]) -> None:
        keys: set[str] = set()
        for item in candidates:
            if not item.stable_key:
                raise ValueError("stable key is required")
            if item.stable_key in keys:
                raise ValueError(f"duplicate stable key: {item.stable_key}")
            keys.add(item.stable_key)
            if not 0 <= item.layer <= 7:
                raise ValueError("context layer must be between 0 and 7")
            if item.temporal_distance < 0:
                raise ValueError("temporal distance must be nonnegative")
            offsets = (item.excerpt_start, item.excerpt_end)
            if offsets == (None, None):
                continue
            if (
                item.excerpt_start is None
                or item.excerpt_end is None
                or item.excerpt_start < 0
                or item.excerpt_end <= item.excerpt_start
            ):
                raise ValueError("excerpt offsets must form a positive range")

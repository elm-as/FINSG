"""Leak-free temporal graph construction from historical entity co-movements."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphEdge:
    """Weighted undirected edge between two financial entities."""

    source: str
    target: str
    weight: float


@dataclass(frozen=True)
class TemporalGraph:
    """Immutable adjacency representation constructed strictly from past observations."""

    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]


def build_correlation_graph(
    returns_by_entity: dict[str, tuple[float, ...]],
    cutoff_index: int,
    min_correlation: float = 0.4,
) -> TemporalGraph:
    """Build weighted edges using only return series up to cutoff_index (inclusive)."""
    if cutoff_index < 2:
        raise ValueError("cutoff_index must be >= 2 to compute historical correlation.")

    entities = tuple(sorted(returns_by_entity.keys()))
    edges: list[GraphEdge] = []
    for i, src in enumerate(entities):
        for dst in entities[i + 1 :]:
            corr = _pearson_prefix(returns_by_entity[src], returns_by_entity[dst], cutoff_index)
            if corr >= min_correlation:
                edges.append(GraphEdge(source=src, target=dst, weight=corr))

    return TemporalGraph(nodes=entities, edges=tuple(edges))


def _pearson_prefix(xs: tuple[float, ...], ys: tuple[float, ...], cutoff_index: int) -> float:
    """Compute Pearson correlation over indices [0, cutoff_index]."""
    end = min(len(xs), len(ys), cutoff_index + 1)
    if end < 3:
        return 0.0

    x_slice = xs[:end]
    y_slice = ys[:end]
    mean_x = sum(x_slice) / end
    mean_y = sum(y_slice) / end
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x_slice, y_slice))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x_slice))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y_slice))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)

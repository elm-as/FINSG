"""Graph message-passing propagation for systemic volatility contagion."""

from __future__ import annotations

from src.graph.temporal_graph import TemporalGraph


def propagate_node_signals(
    graph: TemporalGraph,
    local_signals: dict[str, float],
    damping: float = 0.35,
) -> dict[str, float]:
    """Blend local narrative stress with weighted 1-hop neighbor contagion."""
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must lie in [0, 1].")

    neighbor_sums = {node: 0.0 for node in graph.nodes}
    weight_totals = {node: 0.0 for node in graph.nodes}

    for edge in graph.edges:
        src_val = local_signals.get(edge.source, 0.0)
        dst_val = local_signals.get(edge.target, 0.0)
        neighbor_sums[edge.source] += edge.weight * dst_val
        weight_totals[edge.source] += edge.weight
        neighbor_sums[edge.target] += edge.weight * src_val
        weight_totals[edge.target] += edge.weight

    propagated: dict[str, float] = {}
    for node in graph.nodes:
        own = local_signals.get(node, 0.0)
        if weight_totals[node] <= 1e-12:
            propagated[node] = own
            continue
        neighbor_avg = neighbor_sums[node] / weight_totals[node]
        propagated[node] = (1.0 - damping) * own + damping * neighbor_avg
    return propagated

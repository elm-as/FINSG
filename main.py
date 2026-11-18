"""CLI entrypoint for FINSG (Financial Narrative & Systemic Graph Intelligence)."""

from __future__ import annotations

from src.corpus.document_aligner import filter_documents_before_cutoff, generate_synthetic_corpus
from src.evaluation.walk_forward import compute_macro_f1, compute_roc_auc
from src.graph.temporal_graph import build_correlation_graph
from src.models.graph_propagation import propagate_node_signals
from src.nlp.sentiment_scorer import aggregate_entity_risk


def run_pipeline() -> dict[str, float]:
    """Execute end-to-end FINSG benchmark: NLP stress -> temporal graph -> contagion AUC."""
    corpus = generate_synthetic_corpus()
    visible_docs = filter_documents_before_cutoff(corpus, cutoff_day=5)
    local_risk = aggregate_entity_risk(visible_docs)

    returns = {
        "AAPL": (0.01, -0.02, -0.01, -0.03, -0.02),
        "ASML": (0.02, 0.01, 0.00, -0.01, 0.01),
        "INTC": (-0.01, -0.03, -0.02, -0.04, -0.03),
        "MSFT": (0.02, 0.03, 0.01, 0.02, 0.03),
        "NVDA": (0.03, 0.04, 0.02, 0.01, 0.03),
        "TSMC": (0.02, 0.02, 0.01, 0.00, 0.02),
    }
    graph = build_correlation_graph(returns, cutoff_index=4, min_correlation=0.5)
    propagated = propagate_node_signals(graph, local_risk, damping=0.35)

    ordered_nodes = graph.nodes
    labels = tuple(1 if node in {"AAPL", "INTC"} else 0 for node in ordered_nodes)
    scores = tuple(propagated[node] for node in ordered_nodes)
    preds = tuple(1 if score >= 0.45 else 0 for score in scores)

    return {
        "n_edges": float(len(graph.edges)),
        "roc_auc": round(compute_roc_auc(scores, labels), 4),
        "macro_f1": round(compute_macro_f1(preds, labels), 4),
    }


if __name__ == "__main__":
    metrics = run_pipeline()
    print("[FINSG] Pipeline completed:", metrics)

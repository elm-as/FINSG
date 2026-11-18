"""Automated test suite for FINSG: temporal cutoff, leak-free graph, propagation, metrics."""

from __future__ import annotations

import pytest

from src.corpus.document_aligner import filter_documents_before_cutoff, generate_synthetic_corpus
from src.evaluation.walk_forward import compute_macro_f1, compute_roc_auc
from src.graph.temporal_graph import build_correlation_graph
from src.models.graph_propagation import propagate_node_signals
from src.nlp.sentiment_scorer import aggregate_entity_risk, score_financial_text


def test_document_cutoff_excludes_future_publications() -> None:
    corpus = generate_synthetic_corpus()
    filtered = filter_documents_before_cutoff(corpus, cutoff_day=2)
    assert all(doc.day_index <= 2 for doc in filtered)
    assert len(filtered) == 3


def test_sentiment_scorer_distinguishes_positive_and_negative_tone() -> None:
    pos = score_financial_text("Strong guidance and resilient growth")
    neg = score_financial_text("Severe downturn and margin pressure risk")
    assert pos.polarity > 0.5
    assert neg.polarity < -0.5
    assert neg.uncertainty_ratio > 0.0


def test_temporal_graph_ignores_returns_after_cutoff() -> None:
    returns_a = {
        "A": (0.01, 0.02, 0.03, 0.04, 0.99),
        "B": (0.01, 0.02, 0.03, 0.04, -0.99),
    }
    graph_past = build_correlation_graph(returns_a, cutoff_index=3, min_correlation=0.9)
    assert len(graph_past.edges) == 1

    graph_full = build_correlation_graph(returns_a, cutoff_index=4, min_correlation=0.9)
    assert len(graph_full.edges) == 0


def test_graph_propagation_remains_bounded_and_ranks_risk() -> None:
    corpus = generate_synthetic_corpus()
    local_risk = aggregate_entity_risk(corpus)
    returns = {
        "AAPL": (0.01, -0.02, -0.01, -0.03),
        "INTC": (-0.01, -0.03, -0.02, -0.04),
        "NVDA": (0.03, 0.04, 0.02, 0.01),
    }
    graph = build_correlation_graph(returns, cutoff_index=3, min_correlation=0.5)
    propagated = propagate_node_signals(graph, local_risk, damping=0.3)

    assert all(0.0 <= val <= 1.0 for val in propagated.values())
    auc = compute_roc_auc(
        (propagated["AAPL"], propagated["INTC"], propagated["NVDA"]),
        (1, 1, 0),
    )
    assert auc == pytest.approx(1.0)
    assert compute_macro_f1((1, 1, 0), (1, 1, 0)) == pytest.approx(1.0)

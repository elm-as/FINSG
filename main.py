"""Ingestion and deterministic benchmark generation for financial disclosures & returns."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from src.corpus.document_aligner import FinancialDocument, filter_documents_before_cutoff
from src.evaluation.walk_forward import compute_macro_f1, compute_roc_auc
from src.graph.temporal_graph import build_correlation_graph
from src.models.graph_propagation import propagate_node_signals
from src.nlp.sentiment_scorer import aggregate_entity_risk

CORPUS_PATH = Path("data/financial_corpus.json")
RETURNS_PATH = Path("data/entity_returns.csv")
RESULTS_PATH = Path("artifacts/results.json")

STRESSED_ENTITIES = frozenset({"AAPL", "INTC", "BABA", "BA"})


def ensure_datasets() -> tuple[tuple[FinancialDocument, ...], dict[str, tuple[float, ...]]]:
    """Load or generate persisted financial corpus JSON and multi-entity return CSV."""
    if CORPUS_PATH.exists() and RETURNS_PATH.exists():
        return _load_corpus(CORPUS_PATH), _load_returns(RETURNS_PATH)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = _build_benchmark_corpus()
    CORPUS_PATH.write_text(
        json.dumps([{"entity_id": d.entity_id, "day_index": d.day_index, "headline": d.headline} for d in docs], indent=2),
        encoding="utf-8",
    )
    returns = _build_benchmark_returns(seed=42)
    with RETURNS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        entities = sorted(returns.keys())
        writer.writerow(["day_index", *entities])
        for day in range(len(next(iter(returns.values())))):
            writer.writerow([day, *[returns[e][day] for e in entities]])
    return docs, returns


def _load_corpus(path: Path) -> tuple[FinancialDocument, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(FinancialDocument(r["entity_id"], int(r["day_index"]), r["headline"]) for r in raw)


def _load_returns(path: Path) -> dict[str, tuple[float, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols: dict[str, list[float]] = {k: [] for k in reader.fieldnames or [] if k != "day_index"}
        for row in reader:
            for entity in cols:
                cols[entity].append(float(row[entity]))
    return {k: tuple(v) for k, v in cols.items()}


def _build_benchmark_corpus() -> tuple[FinancialDocument, ...]:
    return (
        FinancialDocument("NVDA", 10, "Strong guidance and record AI server demand"),
        FinancialDocument("MSFT", 12, "Cloud revenue beats consensus with resilient growth"),
        FinancialDocument("ASML", 15, "Lithography backlog rises with stable margin"),
        FinancialDocument("TSMC", 18, "Foundry utilization rises with strong wafer demand"),
        FinancialDocument("GOOGL", 20, "Search growth beats outlook and resilient cloud margin"),
        FinancialDocument("AMZN", 22, "Retail margin rises with record AWS demand"),
        FinancialDocument("META", 25, "Ad revenue beats consensus with strong efficiency"),
        FinancialDocument("ORCL", 28, "Cloud backlog rises despite mild uncertainty"),
        FinancialDocument("AAPL", 30, "Margin pressure and component shortage risk"),
        FinancialDocument("INTC", 32, "Restructuring charge and severe demand downturn"),
        FinancialDocument("BABA", 35, "Regulatory uncertainty and margin pressure downturn"),
        FinancialDocument("BA", 38, "Supply bottleneck and severe delivery charge risk"),
    )


def _build_benchmark_returns(seed: int) -> dict[str, tuple[float, ...]]:
    rng = random.Random(seed)
    entities = ("NVDA", "MSFT", "ASML", "TSMC", "GOOGL", "AMZN", "META", "ORCL", "AAPL", "INTC", "BABA", "BA")
    series: dict[str, tuple[float, ...]] = {}
    market = [rng.gauss(0.002, 0.012) for _ in range(60)]
    stress_factor = [rng.gauss(-0.004, 0.015) for _ in range(60)]
    for entity in entities:
        is_stressed = entity in STRESSED_ENTITIES
        vals = [
            round((0.8 * stress_factor[t] if is_stressed else 0.8 * market[t]) + rng.gauss(0.0, 0.005), 5)
            for t in range(60)
        ]
        series[entity] = tuple(vals)
    return series


def run_pipeline() -> dict[str, float]:
    """Execute end-to-end FINSG benchmark: NLP stress -> temporal graph -> contagion AUC."""
    corpus, returns = ensure_datasets()
    visible_docs = filter_documents_before_cutoff(corpus, cutoff_day=45)
    local_risk = aggregate_entity_risk(visible_docs)

    graph = build_correlation_graph(returns, cutoff_index=45, min_correlation=0.5)
    propagated = propagate_node_signals(graph, local_risk, damping=0.35)

    ordered_nodes = graph.nodes
    labels = tuple(1 if node in STRESSED_ENTITIES else 0 for node in ordered_nodes)
    local_scores = tuple(local_risk.get(node, 0.0) for node in ordered_nodes)
    prop_scores = tuple(propagated[node] for node in ordered_nodes)
    preds = tuple(1 if score >= 0.45 else 0 for score in prop_scores)

    metrics = {
        "n_entities": float(len(ordered_nodes)),
        "n_documents": float(len(visible_docs)),
        "n_edges": float(len(graph.edges)),
        "local_roc_auc": round(compute_roc_auc(local_scores, labels), 4),
        "graph_roc_auc": round(compute_roc_auc(prop_scores, labels), 4),
        "macro_f1": round(compute_macro_f1(preds, labels), 4),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    output = run_pipeline()
    print("[FINSG] Pipeline completed:", output)

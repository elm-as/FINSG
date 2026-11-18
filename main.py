"""CLI entrypoint for FINSG using real yfinance returns and Google News RSS headlines."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.corpus.document_aligner import FinancialDocument
from src.corpus.download_market_news import TICKERS, download_financial_news, download_market_returns
from src.evaluation.walk_forward import compute_macro_f1, compute_roc_auc
from src.graph.temporal_graph import build_correlation_graph
from src.models.graph_propagation import propagate_node_signals
from src.nlp.sentiment_scorer import aggregate_entity_risk
from src.visualization.finsg_plots import generate_finsg_visuals

RETURNS_PATH = Path("data/market_returns.csv")
NEWS_PATH = Path("data/financial_news.json")
RESULTS_PATH = Path("artifacts/results.json")
DASHBOARD_PATH = Path("artifacts/dashboard.html")
IMAGES_DIR = Path("images")


def run_pipeline() -> dict[str, float]:
    """Download real returns & news, build leak-free graph, evaluate walk-forward volatility risk."""
    returns_df = download_market_returns(RETURNS_PATH, period="2y")
    news_raw = download_financial_news(NEWS_PATH)

    docs = tuple(
        FinancialDocument(entity_id=item["entity_id"], day_index=idx, headline=item["headline"])
        for idx, item in enumerate(news_raw)
    )
    local_risk = aggregate_entity_risk(docs)

    returns_map = {ticker: tuple(returns_df[ticker].astype(float).tolist()) for ticker in TICKERS}
    cutoff = int(len(returns_df) * 0.75)

    # Labels: top-tercile realized annualized volatility in the forward 25% out-of-sample window.
    fwd_vols = {
        t: _std(returns_map[t][cutoff:]) * math.sqrt(252.0)
        for t in TICKERS
    }
    vol_threshold = sorted(fwd_vols.values(), reverse=True)[3]
    labels_map = {t: (1 if fwd_vols[t] >= vol_threshold else 0) for t in TICKERS}

    graph = build_correlation_graph(returns_map, cutoff_index=cutoff, min_correlation=0.35)
    propagated = propagate_node_signals(graph, local_risk, damping=0.40)

    ordered = graph.nodes
    y_true = tuple(labels_map[n] for n in ordered)
    s_local = tuple(local_risk.get(n, 0.5) for n in ordered)
    s_prop = tuple(propagated[n] for n in ordered)
    median_prop = sorted(s_prop)[len(s_prop) // 2]
    preds = tuple(1 if s >= median_prop else 0 for s in s_prop)

    generate_finsg_visuals(graph, returns_df, local_risk, propagated, labels_map, IMAGES_DIR, DASHBOARD_PATH)

    metrics = {
        "n_trading_days": float(len(returns_df)),
        "n_entities": float(len(ordered)),
        "n_headlines": float(len(docs)),
        "n_edges": float(len(graph.edges)),
        "local_roc_auc": round(compute_roc_auc(s_local, y_true), 4),
        "graph_roc_auc": round(compute_roc_auc(s_prop, y_true), 4),
        "macro_f1": round(compute_macro_f1(preds, y_true), 4),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _std(values: tuple[float, ...]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1))


if __name__ == "__main__":
    print("[FINSG] Pipeline completed:", run_pipeline())

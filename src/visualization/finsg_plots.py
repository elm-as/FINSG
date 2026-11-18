"""Publication-grade 300 DPI matplotlib charts and interactive Plotly dashboard for FINSG."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from src.graph.temporal_graph import TemporalGraph


def generate_finsg_visuals(
    graph: TemporalGraph,
    returns_df: pd.DataFrame,
    local_scores: dict[str, float],
    prop_scores: dict[str, float],
    labels: dict[str, int],
    images_dir: Path,
    dashboard_path: Path,
) -> None:
    """Render 3 PNG figures (300 DPI) and a standalone Plotly HTML5 dashboard."""
    images_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

    _plot_systemic_graph(graph, prop_scores, images_dir / "finsg_systemic_graph.png")
    _plot_roc_pr_ablation(graph.nodes, local_scores, prop_scores, labels, images_dir / "finsg_ablation_roc_pr.png")
    _plot_volatility_timeline(returns_df, prop_scores, images_dir / "finsg_narrative_volatility_timeline.png")
    _build_plotly_dashboard(graph, returns_df, prop_scores, dashboard_path)


def _plot_systemic_graph(graph: TemporalGraph, prop_scores: dict[str, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    nodes = graph.nodes
    n = len(nodes)
    pos = {
        node: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, node in enumerate(nodes)
    }
    for edge in graph.edges:
        x0, y0 = pos[edge.source]
        x1, y1 = pos[edge.target]
        ax.plot([x0, x1], [y0, y1], color="#94A3B8", alpha=min(0.85, edge.weight), linewidth=1.2 + 2.0 * (edge.weight - 0.5))

    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    colors = [prop_scores.get(node, 0.5) for node in nodes]
    sc = ax.scatter(xs, ys, c=colors, cmap="YlOrRd", s=650, edgecolors="#1E293B", linewidth=1.5, zorder=3)
    for node in nodes:
        x, y = pos[node]
        ax.text(x, y, node, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=4)

    plt.colorbar(sc, ax=ax, label="Propagated Systemic Stress Score")
    ax.set_title("Temporal Correlation Graph & Propagated Narrative Stress (12 Equities)", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_roc_pr_ablation(
    nodes: tuple[str, ...],
    local_scores: dict[str, float],
    prop_scores: dict[str, float],
    labels: dict[str, int],
    out_path: Path,
) -> None:
    y_true = np.array([labels[n] for n in nodes])
    s_loc = np.array([local_scores[n] for n in nodes])
    s_prop = np.array([prop_scores[n] for n in nodes])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for name, scores, color in [("Local Narrative Only", s_loc, "#64748B"), ("Graph-Propagated (1-Hop)", s_prop, "#2563EB")]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ax1.plot(fpr, tpr, label=name, color=color, linewidth=2.0)
        ax2.plot(rec, prec, label=name, color=color, linewidth=2.0)

    ax1.plot([0, 1], [0, 1], "--", color="#CBD5E1", linewidth=1.0)
    ax1.set_title("ROC Curve: High-Volatility Regime Detection", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(frameon=False)

    ax2.set_title("Precision-Recall Ablation", fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_volatility_timeline(returns_df: pd.DataFrame, prop_scores: dict[str, float], out_path: Path) -> None:
    tickers = [c for c in returns_df.columns if c != "Date"]
    top_risk = sorted(tickers, key=lambda t: prop_scores.get(t, 0.0), reverse=True)[:3]
    low_risk = sorted(tickers, key=lambda t: prop_scores.get(t, 0.0))[:3]

    dates = pd.to_datetime(returns_df["Date"])
    high_vol = returns_df[top_risk].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100
    low_vol = returns_df[low_risk].rolling(20).std().mean(axis=1) * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(dates, high_vol, color="#DC2626", linewidth=1.8, label=f"Top-3 Graph-Risk Basket ({', '.join(top_risk)})")
    ax.plot(dates, low_vol, color="#059669", linewidth=1.8, label=f"Low-Risk Basket ({', '.join(low_risk)})")
    ax.set_title("20-Day Rolling Annualized Volatility by Graph-Propagated Stress Tier", fontweight="bold")
    ax.set_xlabel("Trading Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_plotly_dashboard(
    graph: TemporalGraph,
    returns_df: pd.DataFrame,
    prop_scores: dict[str, float],
    dashboard_path: Path,
) -> None:
    nodes = list(graph.nodes)
    scores = [prop_scores.get(n, 0.0) for n in nodes]
    fig = go.Figure(data=[go.Bar(x=nodes, y=scores, marker_color="#2563EB", name="Propagated Stress")])
    fig.update_layout(
        title="FINSG — Propagated Systemic Stress Score Across 12 Semiconductor & Big Tech Equities",
        xaxis_title="Ticker",
        yaxis_title="Systemic Stress Score [0, 1]",
        template="plotly_white",
    )
    fig.write_html(str(dashboard_path), include_plotlyjs="cdn")

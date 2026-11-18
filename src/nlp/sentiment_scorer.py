"""Deterministic financial sentiment and uncertainty scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.corpus.document_aligner import FinancialDocument

POSITIVE_TERMS = frozenset({"strong", "record", "beats", "resilient", "growth", "rises", "stable"})
NEGATIVE_TERMS = frozenset({"pressure", "shortage", "risk", "downturn", "severe", "charge", "bottleneck"})
UNCERTAINTY_TERMS = frozenset({"uncertainty", "risk", "despite", "restructuring"})


@dataclass(frozen=True)
class NarrativeSignal:
    """Extracted polarity and uncertainty ratios for a financial text."""

    polarity: float
    uncertainty_ratio: float


def score_financial_text(text: str) -> NarrativeSignal:
    """Compute normalized polarity in [-1, 1] and uncertainty ratio in [0, 1]."""
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return NarrativeSignal(polarity=0.0, uncertainty_ratio=0.0)

    pos_hits = sum(1 for tok in tokens if tok in POSITIVE_TERMS)
    neg_hits = sum(1 for tok in tokens if tok in NEGATIVE_TERMS)
    unc_hits = sum(1 for tok in tokens if tok in UNCERTAINTY_TERMS)

    total_sentiment = pos_hits + neg_hits
    polarity = (pos_hits - neg_hits) / total_sentiment if total_sentiment > 0 else 0.0
    return NarrativeSignal(
        polarity=polarity,
        uncertainty_ratio=unc_hits / len(tokens),
    )


def aggregate_entity_risk(
    documents: tuple[FinancialDocument, ...],
) -> dict[str, float]:
    """Map each entity to a mean narrative stress score in [0, 1] (higher = more stress)."""
    grouped: dict[str, list[float]] = {}
    for doc in documents:
        sig = score_financial_text(doc.headline)
        stress = min(1.0, max(0.0, 0.5 * (1.0 - sig.polarity) + 0.5 * sig.uncertainty_ratio))
        grouped.setdefault(doc.entity_id, []).append(stress)

    return {entity: sum(vals) / len(vals) for entity, vals in grouped.items()}

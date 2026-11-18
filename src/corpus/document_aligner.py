"""Temporal alignment and validation of financial news and earnings disclosures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinancialDocument:
    """Immutable financial disclosure or news item bound to an entity and timestamp."""

    entity_id: str
    day_index: int
    headline: str


def filter_documents_before_cutoff(
    documents: tuple[FinancialDocument, ...],
    cutoff_day: int,
) -> tuple[FinancialDocument, ...]:
    """Return only documents published strictly at or before cutoff_day (zero look-ahead)."""
    if cutoff_day < 0:
        raise ValueError("cutoff_day cannot be negative.")
    return tuple(doc for doc in documents if doc.day_index <= cutoff_day)


def generate_synthetic_corpus() -> tuple[FinancialDocument, ...]:
    """Generate a deterministic benchmark corpus across 6 interconnected entities."""
    return (
        FinancialDocument("NVDA", 1, "Strong guidance and record AI server demand"),
        FinancialDocument("TSMC", 2, "Foundry utilization rises despite supply bottleneck"),
        FinancialDocument("ASML", 2, "Lithography backlog stable with mild uncertainty"),
        FinancialDocument("AAPL", 3, "Margin pressure and component shortage risk"),
        FinancialDocument("MSFT", 4, "Cloud revenue beats consensus with resilient growth"),
        FinancialDocument("INTC", 5, "Restructuring charge and severe demand downturn"),
    )

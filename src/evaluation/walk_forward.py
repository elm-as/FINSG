"""Walk-forward ranking and classification metrics for volatility shock detection."""

from __future__ import annotations


def compute_roc_auc(scores: tuple[float, ...], labels: tuple[int, ...]) -> float:
    """Compute exact ROC-AUC via Mann-Whitney U statistic with tie averaging."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and equal length.")

    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires at least one positive and one negative label.")

    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def compute_macro_f1(
    predictions: tuple[int, ...],
    labels: tuple[int, ...],
) -> float:
    """Compute binary Macro-F1 score across classes 0 and 1."""
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must be non-empty and equal length.")

    f1_scores = [_binary_f1(predictions, labels, cls) for cls in (0, 1)]
    return sum(f1_scores) / 2.0


def _binary_f1(predictions: tuple[int, ...], labels: tuple[int, ...], target_cls: int) -> float:
    """Compute F1 for a single class."""
    tp = sum(1 for p, y in zip(predictions, labels) if p == target_cls and y == target_cls)
    fp = sum(1 for p, y in zip(predictions, labels) if p == target_cls and y != target_cls)
    fn = sum(1 for p, y in zip(predictions, labels) if p != target_cls and y == target_cls)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0

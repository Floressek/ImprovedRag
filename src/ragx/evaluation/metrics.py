"""EM, F1, BLEU, custom metrics (placeholder)."""
from __future__ import annotations


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def f1(pred: str, gold: str) -> float:
    ps = set(pred.split())
    gs = set(gold.split())
    if not ps or not gs:
        return 0.0
    tp = len(ps & gs)
    precision = tp / len(ps)
    recall = tp / len(gs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

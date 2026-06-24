"""Pure classification metrics — no I/O, no model. Given parallel lists of
true + predicted string labels, compute a confusion matrix, per-label
precision/recall/F1, and a binary (theft-vs-benign) summary.

Kept dependency-free (stdlib only) so it runs anywhere and is trivially testable.
"""

from __future__ import annotations

from collections.abc import Sequence

# VLM categories that count as a theft signal for the binary view. browsing /
# cart_pickup (own cart) / other are benign. A ground-truth label of "theft" or
# "benign" is honoured directly so a human can label clips simply.
THEFT_CATEGORIES: frozenset[str] = frozenset({"pocket_conceal", "bag_conceal", "theft"})


def to_binary(label: str) -> str:
    """Collapse a category (or 'theft'/'benign') to 'theft' | 'benign'."""
    return "theft" if label.strip().lower() in THEFT_CATEGORIES else "benign"


def _safe_div(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    """Nested dict cm[true][pred] = count, over the union of observed labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    labels = sorted(set(y_true) | set(y_pred))
    cm = {t: dict.fromkeys(labels, 0) for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t][p] += 1
    return cm


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    """Multi-class per-label precision/recall/F1/support + accuracy + macro-F1.

    precision[L] = TP / (predicted L); recall[L] = TP / (actual L). None when the
    denominator is 0 (so 'no data' is distinct from 0.0)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    labels = sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, object]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        pred_pos = sum(1 for p in y_pred if p == label)
        actual_pos = sum(1 for t in y_true if t == label)
        precision = _safe_div(tp, pred_pos)
        recall = _safe_div(tp, actual_pos)
        f1 = (
            _safe_div(2 * precision * recall, precision + recall)
            if precision and recall
            else (0.0 if (precision is not None and recall is not None) else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_pos,
        }
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return {
        "n": n,
        "accuracy": _safe_div(correct, n),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "labels": per_label,
    }


def binary_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], *, positive: str = "theft"
) -> dict[str, object]:
    """Binary theft-vs-benign summary after collapsing labels with `to_binary`.

    Reports tp/fp/fn/tn + precision/recall/F1/accuracy. Recall is only meaningful
    when the labelled set contains real positives that the system COULD have
    missed (i.e. a curated clip set, not just alerts) — that's the whole point of
    measuring against ground truth rather than only alerted clips."""
    bt = [to_binary(x) for x in y_true]
    bp = [to_binary(x) for x in y_pred]
    tp = sum(1 for t, p in zip(bt, bp, strict=True) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(bt, bp, strict=True) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(bt, bp, strict=True) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(bt, bp, strict=True) if t != positive and p != positive)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision and recall else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
    }

"""Unit tests for the pure eval metrics (confusion / classification / binary)."""

from __future__ import annotations

from sentry_ai.eval.metrics import (
    binary_metrics,
    classification_report,
    confusion_matrix,
    to_binary,
)


def test_to_binary_maps_concealment_to_theft() -> None:
    assert to_binary("pocket_conceal") == "theft"
    assert to_binary("bag_conceal") == "theft"
    assert to_binary("theft") == "theft"
    assert to_binary("browsing") == "benign"
    assert to_binary("cart_pickup") == "benign"
    assert to_binary("other") == "benign"
    assert to_binary("BENIGN") == "benign"


def test_confusion_matrix_counts() -> None:
    y_true = ["theft", "theft", "benign", "benign"]
    y_pred = ["theft", "benign", "benign", "benign"]
    cm = confusion_matrix(y_true, y_pred)
    assert cm["theft"]["theft"] == 1
    assert cm["theft"]["benign"] == 1
    assert cm["benign"]["benign"] == 2
    assert cm["benign"]["theft"] == 0


def test_confusion_matrix_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="same length"):
        confusion_matrix(["a"], ["a", "b"])


def test_classification_report_perfect() -> None:
    y = ["pocket_conceal", "browsing", "bag_conceal"]
    rep = classification_report(y, y)
    assert rep["accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0
    labels = rep["labels"]  # type: ignore[index]
    assert labels["pocket_conceal"]["precision"] == 1.0
    assert labels["pocket_conceal"]["recall"] == 1.0


def test_classification_report_mixed() -> None:
    # 2 pocket_conceal (1 right, 1 called browsing), 1 browsing (right)
    y_true = ["pocket_conceal", "pocket_conceal", "browsing"]
    y_pred = ["pocket_conceal", "browsing", "browsing"]
    rep = classification_report(y_true, y_pred)
    labels = rep["labels"]  # type: ignore[index]
    # pocket_conceal: tp=1, predicted=1 → precision 1.0; actual=2 → recall 0.5
    assert labels["pocket_conceal"]["precision"] == 1.0
    assert labels["pocket_conceal"]["recall"] == 0.5
    # browsing: tp=1, predicted=2 → precision 0.5; actual=1 → recall 1.0
    assert labels["browsing"]["precision"] == 0.5
    assert labels["browsing"]["recall"] == 1.0
    assert rep["accuracy"] == round(2 / 3, 4)


def test_binary_metrics_precision_recall() -> None:
    # ground truth: 2 theft, 2 benign. Pred: catches 1 theft, 1 false alarm.
    y_true = ["pocket_conceal", "bag_conceal", "browsing", "other"]
    y_pred = ["pocket_conceal", "browsing", "pocket_conceal", "other"]
    m = binary_metrics(y_true, y_pred)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["precision"] == 0.5  # 1 / (1 + 1)
    assert m["recall"] == 0.5  # 1 / (1 + 1)
    assert m["accuracy"] == 0.5


def test_binary_metrics_empty_is_none() -> None:
    m = binary_metrics([], [])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["tp"] == 0

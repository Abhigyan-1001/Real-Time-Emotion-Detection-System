"""Classification metrics and uncertainty-threshold analysis.

Pure functions on NumPy arrays: no model, camera or file access, so they are easy to test.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_summary(y_true, y_pred, class_names) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),  # plain average over classes: every class counts equally
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),  # rows = true class, columns = predicted class
        "class_names": list(class_names),
    }


def threshold_sweep(scores, y_true, class_names, thresholds) -> list[dict]:
    """For each threshold: how many images get a label, and how accurate are those labels."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    top = scores.max(axis=1)
    correct = scores.argmax(axis=1) == y_true
    rows = []
    for t in thresholds:
        accepted = top >= t  # same rule as the app: below the threshold -> "uncertain"
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(accepted.mean()),
                "accuracy_on_accepted": float(correct[accepted].mean()) if accepted.any() else math.nan,
                "per_class_coverage": {
                    name: float(accepted[y_true == i].mean()) if (y_true == i).any() else math.nan
                    for i, name in enumerate(class_names)
                },
            }
        )
    return rows


def reliability_table(scores, y_true, edges) -> list[dict]:
    """Group predictions by their top score and report the actual accuracy in each group."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    top = scores.max(axis=1)
    correct = scores.argmax(axis=1) == y_true
    rows = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        is_last = i == len(edges) - 2
        in_bin = (top >= low) & ((top <= high) if is_last else (top < high))
        n = int(in_bin.sum())
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "count": n,
                "mean_confidence": float(top[in_bin].mean()) if n else math.nan,
                "accuracy": float(correct[in_bin].mean()) if n else math.nan,
            }
        )
    return rows
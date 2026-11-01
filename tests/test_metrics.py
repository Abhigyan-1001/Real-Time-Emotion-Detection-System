import math

import numpy as np
import pytest

from src.metrics import classification_summary, reliability_table, threshold_sweep

NAMES = ["a", "b", "c"]


def test_summary_matches_hand_calculation():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    s = classification_summary(y_true, y_pred, NAMES)

    # class a: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=0.5
    # class b: TP=2, FP=1, FN=0 -> P=2/3, R=1,   F1=0.8
    # class c: TP=1, FP=0, FN=1 -> P=1,   R=1/2, F1=2/3
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["per_class"]["a"]["precision"] == pytest.approx(0.5)
    assert s["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert s["per_class"]["b"]["recall"] == pytest.approx(1.0)
    assert s["per_class"]["c"]["recall"] == pytest.approx(0.5)
    assert s["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert s["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert [s["per_class"][n]["support"] for n in NAMES] == [2, 2, 2]


def test_class_that_is_never_predicted_scores_zero_without_crashing():
    s = classification_summary([0, 1, 2], [0, 1, 1], NAMES)
    assert s["per_class"]["c"]["precision"] == 0.0
    assert s["per_class"]["c"]["recall"] == 0.0


SCORES = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.3, 0.7]])
Y = np.array([0, 1, 0, 1])  # predictions are [0, 0, 0, 1] -> correct [T, F, T, T]
TWO = ["x", "y"]


def test_sweep_with_everything_accepted():
    row = threshold_sweep(SCORES, Y, TWO, [0.5])[0]
    assert row["coverage"] == pytest.approx(1.0)
    assert row["accuracy_on_accepted"] == pytest.approx(0.75)


def test_sweep_keeps_only_confident_predictions():
    row = threshold_sweep(SCORES, Y, TWO, [0.65])[0]  # accepted: 0.9 and 0.7
    assert row["coverage"] == pytest.approx(0.5)
    assert row["accuracy_on_accepted"] == pytest.approx(1.0)
    assert row["per_class_coverage"] == {"x": pytest.approx(0.5), "y": pytest.approx(0.5)}


def test_score_exactly_at_the_threshold_is_accepted():
    row = threshold_sweep(SCORES, Y, TWO, [0.6])[0]  # accepted: 0.9, 0.6, 0.7
    assert row["coverage"] == pytest.approx(0.75)
    assert row["accuracy_on_accepted"] == pytest.approx(2 / 3)


def test_sweep_with_nothing_accepted():
    row = threshold_sweep(SCORES, Y, TWO, [0.95])[0]
    assert row["coverage"] == 0.0
    assert math.isnan(row["accuracy_on_accepted"])


def test_reliability_bins():
    rows = reliability_table(SCORES, Y, [0.5, 0.75, 1.0])
    assert rows[0]["count"] == 3  # top scores 0.6, 0.55, 0.7
    assert rows[0]["accuracy"] == pytest.approx(2 / 3)
    assert rows[0]["mean_confidence"] == pytest.approx((0.6 + 0.55 + 0.7) / 3)
    assert rows[1]["count"] == 1  # top score 0.9
    assert rows[1]["accuracy"] == pytest.approx(1.0)


def test_empty_bin_reports_nan():
    rows = reliability_table(SCORES, Y, [0.0, 0.2, 1.0])
    assert rows[0]["count"] == 0
    assert math.isnan(rows[0]["accuracy"])
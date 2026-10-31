import json

import numpy as np
import pytest

from src.predictor import EmotionPredictor

EXPECTED_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def make_batch(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 48, 48, 1)).astype(np.float32)


def test_labels_are_in_training_order(predictor):
    assert predictor.labels == EXPECTED_LABELS


def test_one_row_per_input_and_one_column_per_class(predictor):
    assert predictor.predict_scores(make_batch(3)).shape == (3, len(EXPECTED_LABELS))


def test_scores_are_probabilities(predictor):
    scores = predictor.predict_scores(make_batch(4))
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0, atol=1e-4)


def test_inference_is_deterministic(predictor):
    batch = make_batch(2)
    assert np.allclose(predictor.predict_scores(batch), predictor.predict_scores(batch))


def test_top_label_uses_the_index_to_label_mapping(predictor):
    label, score = predictor.top_label(np.array([0.05, 0.02, 0.08, 0.70, 0.05, 0.07, 0.03]))
    assert label == "happy"
    assert score == pytest.approx(0.70)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((48, 48, 1), dtype=np.float32),      # missing batch axis
        np.zeros((1, 48, 48), dtype=np.float32),      # missing channel axis
        np.zeros((1, 64, 64, 1), dtype=np.float32),   # wrong size
        None,
        "not an array",
    ],
)
def test_wrong_input_raises_value_error(predictor, bad):
    with pytest.raises(ValueError):
        predictor.predict_scores(bad)


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmotionPredictor(tmp_path / "missing.keras", tmp_path / "missing.json")


def test_label_count_mismatch_is_rejected(model_paths, tmp_path):
    model_path, _ = model_paths
    bad_labels = tmp_path / "labels.json"
    bad_labels.write_text(json.dumps(["a", "b", "c"]))
    with pytest.raises(ValueError):
        EmotionPredictor(model_path, bad_labels)
        
import numpy as np
import pytest

from src.face_detector import Detection
from src.pipeline import FaceResult
from src.postprocessing import (
    UNCERTAIN_LABEL,
    FaceTracker,
    ScoreSmoother,
    decide_label,
    iou,
)

LABELS = ["a", "b", "c"]


def make_result(box, scores):
    scores = np.array(scores, dtype=np.float64)
    detection = Detection(
        x=box[0], y=box[1], w=box[2], h=box[3], score=0.99, landmarks=np.zeros((5, 2))
    )
    return FaceResult(detection=detection, label="x", score=float(scores.max()), scores=scores)


# ---- IoU ----

def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0


def test_iou_partial_overlap():
    # intersection 5*10 = 50, union 100 + 100 - 50 = 150
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


# ---- Smoothing ----

def test_first_update_returns_input_unchanged():
    s = ScoreSmoother(alpha=0.3)
    assert np.allclose(s.update([0.7, 0.3]), [0.7, 0.3])


def test_ema_matches_the_formula():
    s = ScoreSmoother(alpha=0.3)
    s.update([0.7, 0.3])
    out = s.update([0.1, 0.9])
    # 0.3*0.1 + 0.7*0.7 = 0.52 ; 0.3*0.9 + 0.7*0.3 = 0.48
    assert np.allclose(out, [0.52, 0.48])
    assert out.sum() == pytest.approx(1.0)


def test_alpha_one_means_no_smoothing():
    s = ScoreSmoother(alpha=1.0)
    s.update([0.7, 0.3])
    assert np.allclose(s.update([0.1, 0.9]), [0.1, 0.9])


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        ScoreSmoother(alpha=alpha)


def test_smoothing_reduces_label_flicker():
    frames = [[0.6, 0.4] if i % 2 == 0 else [0.4, 0.6] for i in range(20)]
    smoother = ScoreSmoother(alpha=0.3)
    raw = [int(np.argmax(f)) for f in frames]
    smoothed = [int(np.argmax(smoother.update(f))) for f in frames]

    def changes(seq):
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    assert changes(smoothed) < changes(raw)


def test_reset_forgets_history():
    s = ScoreSmoother(alpha=0.3)
    s.update([0.9, 0.1])
    s.reset()
    assert np.allclose(s.update([0.2, 0.8]), [0.2, 0.8])


# ---- Uncertainty ----

def test_confident_prediction_keeps_its_label():
    label, score, uncertain = decide_label(np.array([0.1, 0.7, 0.2]), LABELS, 0.4)
    assert (label, uncertain) == ("b", False)
    assert score == pytest.approx(0.7)


def test_low_confidence_becomes_uncertain():
    label, _, uncertain = decide_label(np.array([0.34, 0.33, 0.33]), LABELS, 0.4)
    assert label == UNCERTAIN_LABEL
    assert uncertain is True


def test_score_exactly_at_threshold_is_accepted():
    _, _, uncertain = decide_label(np.array([0.4, 0.3, 0.3]), LABELS, 0.4)
    assert uncertain is False


# ---- Tracking ----

def test_same_face_keeps_its_id_when_it_moves_slightly():
    tracker = FaceTracker(LABELS)
    first = tracker.update([make_result((100, 100, 80, 80), [0.8, 0.1, 0.1])])
    second = tracker.update([make_result((108, 104, 80, 80), [0.8, 0.1, 0.1])])
    assert first[0].track_id == second[0].track_id


def test_two_faces_get_different_ids_regardless_of_list_order():
    tracker = FaceTracker(LABELS)
    face_a = make_result((0, 0, 100, 100), [0.9, 0.05, 0.05])
    face_b = make_result((300, 0, 100, 100), [0.05, 0.9, 0.05])
    first = tracker.update([face_a, face_b])
    second = tracker.update([face_b, face_a])  # same faces, reversed order
    assert first[0].track_id != first[1].track_id
    assert second[0].track_id == first[1].track_id
    assert second[1].track_id == first[0].track_id
    assert second[0].label == "b"


def test_histories_of_two_faces_are_not_mixed():
    tracker = FaceTracker(LABELS, alpha=0.3)
    for _ in range(5):
        out = tracker.update(
            [
                make_result((0, 0, 100, 100), [0.9, 0.05, 0.05]),
                make_result((300, 0, 100, 100), [0.05, 0.9, 0.05]),
            ]
        )
    assert out[0].label == "a" and out[0].score == pytest.approx(0.9)
    assert out[1].label == "b" and out[1].score == pytest.approx(0.9)


def test_face_missing_briefly_keeps_its_id():
    tracker = FaceTracker(LABELS, max_missed=5)
    box = (100, 100, 80, 80)
    first = tracker.update([make_result(box, [0.8, 0.1, 0.1])])
    tracker.update([])  # face lost for one frame
    again = tracker.update([make_result(box, [0.8, 0.1, 0.1])])
    assert again[0].track_id == first[0].track_id


def test_face_missing_too_long_gets_a_new_id():
    tracker = FaceTracker(LABELS, max_missed=2)
    box = (100, 100, 80, 80)
    first = tracker.update([make_result(box, [0.8, 0.1, 0.1])])
    for _ in range(3):
        tracker.update([])
    again = tracker.update([make_result(box, [0.8, 0.1, 0.1])])
    assert again[0].track_id != first[0].track_id


def test_far_away_face_is_a_new_person():
    tracker = FaceTracker(LABELS)
    first = tracker.update([make_result((0, 0, 80, 80), [0.8, 0.1, 0.1])])
    second = tracker.update([make_result((400, 300, 80, 80), [0.8, 0.1, 0.1])])
    assert second[0].track_id != first[0].track_id


def test_uncertain_face_keeps_raw_prediction_for_comparison():
    tracker = FaceTracker(LABELS, threshold=0.4)
    out = tracker.update([make_result((0, 0, 80, 80), [0.34, 0.33, 0.33])])
    assert out[0].uncertain is True
    assert out[0].label == UNCERTAIN_LABEL
    assert out[0].raw_label == "a"


def test_reset_restarts_track_ids():
    tracker = FaceTracker(LABELS)
    tracker.update([make_result((0, 0, 80, 80), [0.8, 0.1, 0.1])])
    tracker.reset()
    out = tracker.update([make_result((0, 0, 80, 80), [0.8, 0.1, 0.1])])
    assert out[0].track_id == 1

def test_configure_changes_alpha_for_existing_faces():
    tracker = FaceTracker(LABELS, alpha=1.0)
    box = (0, 0, 80, 80)
    tracker.update([make_result(box, [0.9, 0.05, 0.05])])
    tracker.configure(alpha=0.5)
    out = tracker.update([make_result(box, [0.1, 0.8, 0.1])])
    # a: 0.5*0.1 + 0.5*0.9 = 0.50   b: 0.5*0.8 + 0.5*0.05 = 0.425
    assert out[0].label == "a"
    assert out[0].score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0.0}, {"alpha": 1.5}, {"threshold": -0.1}, {"threshold": 1.2}]
)
def test_configure_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FaceTracker(LABELS).configure(**kwargs)  
import threading

import numpy as np
import pytest

from src.face_detector import Detection
from src.session import LiveAnalyzer

LABELS = ["a", "b", "c"]


class FakeDetector:
    def __init__(self):
        self.boxes = []

    def detect(self, frame):
        return [
            Detection(x=x, y=y, w=w, h=h, score=0.99, landmarks=np.zeros((5, 2)))
            for x, y, w, h in self.boxes
        ]


class RaisingDetector:
    def detect(self, frame):
        raise RuntimeError("boom")


class FakePredictor:
    labels = LABELS

    def __init__(self):
        self.scores = np.array([0.8, 0.1, 0.1])

    def predict_scores(self, model_input):
        return np.tile(self.scores, (model_input.shape[0], 1))

    def top_label(self, scores):
        i = int(np.argmax(scores))
        return LABELS[i], float(scores[i])


def blank():
    return np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.fixture
def setup():
    detector, predictor = FakeDetector(), FakePredictor()
    return LiveAnalyzer(detector, predictor, alpha=1.0, threshold=0.4), detector, predictor


def test_frame_without_faces(setup):
    analyzer, _, _ = setup
    out = analyzer.process(blank())
    snap = analyzer.snapshot()
    assert out.shape == (200, 300, 3)
    assert (snap.frames_total, snap.frames_with_face) == (1, 0)
    assert snap.current_label is None
    assert snap.prediction_counts == {}


def test_one_face_is_counted_and_reported(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(50, 50, 80, 80)]
    analyzer.process(blank())
    snap = analyzer.snapshot()
    assert snap.current_label == "a"
    assert snap.current_score == pytest.approx(0.8)
    assert snap.current_uncertain is False
    assert snap.prediction_counts == {"a": 1}
    assert snap.faces_in_frame == 1


def test_overlay_is_drawn_on_the_frame(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(50, 50, 80, 80)]
    out = analyzer.process(blank())
    assert out.any()  # a blank frame now contains drawn pixels


def test_uncertain_predictions_are_counted_as_uncertain(setup):
    analyzer, detector, predictor = setup
    detector.boxes = [(50, 50, 80, 80)]
    predictor.scores = np.array([0.34, 0.33, 0.33])
    analyzer.process(blank())
    snap = analyzer.snapshot()
    assert snap.prediction_counts == {"uncertain": 1}
    assert snap.current_uncertain is True


def test_every_face_is_counted(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(10, 10, 40, 40), (100, 50, 80, 80)]
    analyzer.process(blank())
    snap = analyzer.snapshot()
    assert snap.prediction_counts == {"a": 2}
    assert snap.faces_in_frame == 2
    assert snap.frames_with_face == 1  # one frame, two faces


def test_reset_clears_statistics(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(50, 50, 80, 80)]
    analyzer.process(blank())
    analyzer.reset()
    snap = analyzer.snapshot()
    assert snap.frames_total == 0
    assert snap.prediction_counts == {}
    assert snap.current_label is None


def test_threshold_can_be_changed_while_running(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(50, 50, 80, 80)]
    analyzer.configure(threshold=0.9)  # the fake model's top score is 0.8
    analyzer.process(blank())
    assert analyzer.snapshot().prediction_counts == {"uncertain": 1}


def test_invalid_settings_are_rejected(setup):
    analyzer, _, _ = setup
    with pytest.raises(ValueError):
        analyzer.configure(alpha=0.0)


def test_pipeline_errors_do_not_escape():
    analyzer = LiveAnalyzer(RaisingDetector(), FakePredictor())
    frame = blank()
    out = analyzer.process(frame)  # must not raise
    snap = analyzer.snapshot()
    assert out is frame
    assert "RuntimeError" in snap.last_error
    assert snap.frames_total == 1


def test_non_array_input_is_ignored(setup):
    analyzer, _, _ = setup
    assert analyzer.process(None) is None
    assert analyzer.snapshot().frames_total == 0


def test_timing_statistics_are_collected(setup):
    analyzer, _, _ = setup
    for _ in range(5):
        analyzer.process(blank())
    snap = analyzer.snapshot()
    assert snap.latency_ms_mean > 0
    assert snap.latency_ms_p95 >= 0
    assert snap.fps > 0


def test_snapshot_while_another_thread_processes_frames(setup):
    analyzer, detector, _ = setup
    detector.boxes = [(50, 50, 80, 80)]

    def worker():
        for _ in range(50):
            analyzer.process(blank())

    thread = threading.Thread(target=worker)
    thread.start()
    while thread.is_alive():
        analyzer.snapshot()  # reading concurrently must never raise
    thread.join()
    assert analyzer.snapshot().frames_total == 50
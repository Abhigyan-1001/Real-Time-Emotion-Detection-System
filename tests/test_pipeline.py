import numpy as np

from src.pipeline import analyze_frame
from tests.helpers import face_row


def test_blank_frame_gives_no_results(detector, predictor):
    assert analyze_frame(np.zeros((480, 640, 3), dtype=np.uint8), detector, predictor) == []


def test_invalid_frame_gives_no_results(detector, predictor):
    assert analyze_frame(None, detector, predictor) == []


def test_one_result_per_face_largest_first(fake_detector_factory, predictor):
    rows = np.stack([face_row(10, 10, 40, 40), face_row(100, 50, 80, 80)])
    det, _ = fake_detector_factory(rows)
    frame = np.random.default_rng(0).integers(0, 256, size=(200, 300, 3), dtype=np.uint8)

    results = analyze_frame(frame, det, predictor)

    assert len(results) == 2
    assert results[0].detection.w == 80  # the larger face comes first
    for r in results:
        assert r.scores.shape == (len(predictor.labels),)
        assert r.label in predictor.labels


def test_max_faces_limits_the_work(fake_detector_factory, predictor):
    rows = np.stack([face_row(10 * i, 10, 30, 30) for i in range(1, 5)])
    det, _ = fake_detector_factory(rows)
    frame = np.random.default_rng(1).integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    assert len(analyze_frame(frame, det, predictor, max_faces=2)) == 2
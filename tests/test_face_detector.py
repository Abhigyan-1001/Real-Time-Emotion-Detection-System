import numpy as np
import pytest

from src.face_detector import FaceDetector
from tests.helpers import face_row


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaceDetector(tmp_path / "missing.onnx")


def test_blank_frame_has_no_faces(detector):
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "not a frame",
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((100, 100), dtype=np.uint8),         # grayscale
        np.zeros((100, 100, 3), dtype=np.float32),    # wrong dtype
        np.zeros((100, 100, 4), dtype=np.uint8),      # 4 channels
    ],
)
def test_invalid_frames_return_empty_list(detector, bad):
    assert detector.detect(bad) == []


def test_none_from_opencv_means_no_faces(fake_detector_factory):
    det, _ = fake_detector_factory(None)  # OpenCV returns None when nothing is found
    assert det.detect(np.zeros((100, 200, 3), dtype=np.uint8)) == []


def test_detection_fields_are_converted(fake_detector_factory):
    det, _ = fake_detector_factory(np.stack([face_row(10, 20, 50, 60, score=0.9)]))
    faces = det.detect(np.zeros((200, 300, 3), dtype=np.uint8))
    assert len(faces) == 1
    assert faces[0].box == (10, 20, 50, 60)
    assert faces[0].score == pytest.approx(0.9)
    assert faces[0].landmarks.shape == (5, 2)


def test_box_is_clipped_to_the_frame(fake_detector_factory):
    det, _ = fake_detector_factory(np.stack([face_row(-20, -10, 60, 50)]))
    faces = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))  # height 100, width 200
    assert faces[0].box == (0, 0, 40, 40)


def test_box_outside_the_frame_is_dropped(fake_detector_factory):
    det, _ = fake_detector_factory(np.stack([face_row(500, 500, 50, 50)]))
    assert det.detect(np.zeros((100, 200, 3), dtype=np.uint8)) == []


def test_input_size_is_updated_only_when_the_frame_size_changes(fake_detector_factory):
    det, fake = fake_detector_factory(None)
    det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    det.detect(np.zeros((50, 80, 3), dtype=np.uint8))
    assert fake.sizes == [(200, 100), (80, 50)]

    
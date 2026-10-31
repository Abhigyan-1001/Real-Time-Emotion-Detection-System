"""Shared fixtures. Model files are loaded once per test session."""

from pathlib import Path

import pytest

from src.face_detector import FaceDetector
from src.predictor import EmotionPredictor

ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_PATH = ROOT / "models" / "emotion_cnn_v1.keras"
LABELS_PATH = ROOT / "models" / "labels.json"


@pytest.fixture(scope="session")
def yunet_path():
    if not YUNET_PATH.is_file():
        pytest.skip("YuNet model file not present in models/")
    return YUNET_PATH


@pytest.fixture(scope="session")
def model_paths():
    if not (MODEL_PATH.is_file() and LABELS_PATH.is_file()):
        pytest.skip("Trained model files not present in models/")
    return MODEL_PATH, LABELS_PATH


@pytest.fixture(scope="session")
def detector(yunet_path):
    return FaceDetector(yunet_path)


@pytest.fixture(scope="session")
def predictor(model_paths):
    return EmotionPredictor(*model_paths)


@pytest.fixture
def fake_detector_factory(yunet_path):
    """Build a FaceDetector whose underlying YuNet returns the rows we choose."""
    from tests.helpers import FakeYuNet

    def factory(rows):
        det = FaceDetector(yunet_path)
        fake = FakeYuNet(rows)
        det._detector = fake  # test-only: replace YuNet with a stand-in
        return det, fake

    return factory
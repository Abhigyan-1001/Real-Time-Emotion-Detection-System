import numpy as np


class FakeYuNet:
    """Stands in for cv2.FaceDetectorYN and returns preset rows."""

    def __init__(self, rows):
        self.rows = rows
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, frame):
        return 1, self.rows


def face_row(x, y, w, h, score=0.95):
    """Build one YuNet-format row: x, y, w, h, five (x, y) landmarks, score."""
    landmarks = np.arange(10, dtype=np.float32)
    return np.array([x, y, w, h, *landmarks, score], dtype=np.float32)
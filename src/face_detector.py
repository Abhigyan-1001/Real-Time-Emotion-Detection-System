"""Face detection with OpenCV's YuNet model.

Stage 1 of the pipeline: find where faces are in a frame.
This module does NOT classify expressions or recognise identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected face, with the box clipped to the frame."""

    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray  # shape (5, 2): two eyes, nose tip, two mouth corners

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class FaceDetector:
    """Wraps cv2.FaceDetectorYN so the rest of the app never touches it directly."""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model not found at: {path}")

        self._input_size = (320, 320)  # (width, height); updated per frame size
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", self._input_size, score_threshold, nms_threshold, top_k
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return all faces found in a BGR uint8 frame (empty list if none/invalid)."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            return []

        height, width = frame.shape[:2]
        if (width, height) != self._input_size:
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, faces = self._detector.detect(frame)
        if faces is None:  # OpenCV returns None when there are no faces
            return []

        detections: list[Detection] = []
        for row in faces:
            x1 = max(int(row[0]), 0)
            y1 = max(int(row[1]), 0)
            x2 = min(int(row[0] + row[2]), width)
            y2 = min(int(row[1] + row[3]), height)
            if x2 - x1 <= 0 or y2 - y1 <= 0:  # box fell outside the frame
                continue
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    score=float(row[14]),
                    landmarks=row[4:14].reshape(5, 2),
                )
            )
        return detections
    
"""One function that runs the full pipeline on a single frame.

frame -> detect faces -> crop -> preprocess -> CNN scores -> results

Used by the live demo, the Streamlit app, the benchmark and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_detector import Detection, FaceDetector
from .predictor import EmotionPredictor
from .preprocessing import crop_face, to_model_input


@dataclass(frozen=True)
class FaceResult:
    detection: Detection
    label: str  # highest-scoring class
    score: float  # that class's softmax score
    scores: np.ndarray  # all class scores, shape (num_classes,)


def analyze_frame(
    frame: np.ndarray,
    detector: FaceDetector,
    predictor: EmotionPredictor,
    max_faces: int = 5,
) -> list[FaceResult]:
    """Return one FaceResult per usable face (largest faces first); [] if none."""
    detections = detector.detect(frame)
    detections = sorted(detections, key=lambda d: d.w * d.h, reverse=True)[:max_faces]

    kept: list[Detection] = []
    inputs: list[np.ndarray] = []
    for det in detections:
        crop = crop_face(frame, det.box)
        model_input = None if crop is None else to_model_input(crop)
        if model_input is not None:
            kept.append(det)
            inputs.append(model_input)

    if not kept:
        return []

    # One model call for all faces in the frame (batch of N), not N separate calls.
    scores = predictor.predict_scores(np.concatenate(inputs, axis=0))

    results = []
    for det, row in zip(kept, scores):
        label, score = predictor.top_label(row)
        results.append(FaceResult(detection=det, label=label, score=score, scores=row))
    return results
"""Live-session logic for the dashboard.

Runs the pipeline on each frame, draws overlays and keeps session statistics.
Contains no Streamlit code, so it can be tested with pytest.

The video callback runs in a worker thread while the dashboard reads statistics
from the main thread, so every shared value is guarded by a lock.
No frames are stored; only counts and timings are kept.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .face_detector import FaceDetector
from .pipeline import analyze_frame
from .postprocessing import FaceTracker, TrackedFace
from .predictor import EmotionPredictor

GREEN = (0, 255, 0)
YELLOW = (0, 215, 255)  # BGR: shown for "uncertain" predictions


@dataclass(frozen=True)
class SessionSnapshot:
    """A consistent copy of the statistics, safe to read from the dashboard thread."""

    frames_total: int
    frames_with_face: int
    prediction_counts: dict[str, int]  # smoothed label -> count (includes "uncertain")
    current_label: str | None  # largest face in the latest frame
    current_score: float | None
    current_uncertain: bool
    faces_in_frame: int
    fps: float  # over the most recent frames
    latency_ms_mean: float  # time inside our callback, per frame
    latency_ms_p95: float
    duration_s: float  # first processed frame to latest processed frame
    last_error: str | None


def draw_faces(frame: np.ndarray, faces: list[TrackedFace]) -> None:
    """Draw boxes and labels onto the frame in place."""
    for f in faces:
        d = f.detection
        color = YELLOW if f.uncertain else GREEN
        cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        text = f"#{f.track_id} {f.label} {f.score:.0%}"
        cv2.putText(frame, text, (d.x, max(d.y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class LiveAnalyzer:
    def __init__(
        self,
        detector: FaceDetector,
        predictor: EmotionPredictor,
        alpha: float = 0.3,
        threshold: float = 0.40,
        max_faces: int = 5,
        history: int = 120,
    ) -> None:
        self._detector = detector
        self._predictor = predictor
        self._max_faces = max_faces
        self._history = history
        self._tracker = FaceTracker(predictor.labels, alpha=alpha, threshold=threshold)
        self._lock = threading.Lock()
        self._reset_stats()

    # ---- called from the video thread ----

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Analyze one BGR frame and draw overlays on it in place. Never raises."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return frame

        start = time.perf_counter()
        faces: list[TrackedFace] = []
        error: str | None = None
        try:
            results = analyze_frame(frame, self._detector, self._predictor, self._max_faces)
            with self._lock:
                faces = self._tracker.update(results)
            draw_faces(frame, faces)
        except Exception as exc:  # keep the stream alive whatever happens
            error = f"{type(exc).__name__}: {exc}"
        self._record(faces, time.perf_counter() - start, error)
        return frame

    def _record(self, faces: list[TrackedFace], elapsed: float, error: str | None) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._started_at is None:
                self._started_at = now
            self._last_at = now
            self._frames_total += 1
            self._stamps.append(now)
            self._latencies.append(elapsed * 1000.0)
            self._last_error = error
            self._faces_in_frame = len(faces)
            if faces:
                self._frames_with_face += 1
                for f in faces:
                    self._counts[f.label] += 1
                main = max(faces, key=lambda f: f.detection.w * f.detection.h)
                self._current = (main.label, main.score, main.uncertain)
            else:
                self._current = None

    # ---- called from the dashboard thread ----

    def configure(self, alpha: float | None = None, threshold: float | None = None) -> None:
        with self._lock:
            self._tracker.configure(alpha=alpha, threshold=threshold)

    def reset(self) -> None:
        with self._lock:
            self._tracker.reset()
            self._reset_stats()

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            latencies = np.array(self._latencies, dtype=float)
            stamps = list(self._stamps)
            fps = 0.0
            if len(stamps) >= 2 and stamps[-1] > stamps[0]:
                fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            duration = 0.0
            if self._started_at is not None and self._last_at is not None:
                duration = self._last_at - self._started_at
            label, score, uncertain = self._current if self._current else (None, None, False)
            return SessionSnapshot(
                frames_total=self._frames_total,
                frames_with_face=self._frames_with_face,
                prediction_counts=dict(self._counts),
                current_label=label,
                current_score=score,
                current_uncertain=uncertain,
                faces_in_frame=self._faces_in_frame,
                fps=fps,
                latency_ms_mean=float(latencies.mean()) if latencies.size else 0.0,
                latency_ms_p95=float(np.percentile(latencies, 95)) if latencies.size else 0.0,
                duration_s=duration,
                last_error=self._last_error,
            )

    # ---- internals (callers hold the lock, or no other thread exists yet) ----

    def _reset_stats(self) -> None:
        self._frames_total = 0
        self._frames_with_face = 0
        self._counts: Counter = Counter()
        self._latencies: deque = deque(maxlen=self._history)
        self._stamps: deque = deque(maxlen=self._history)
        self._started_at: float | None = None
        self._last_at: float | None = None
        self._current: tuple[str, float, bool] | None = None
        self._faces_in_frame = 0
        self._last_error: str | None = None
"""Smoothing, uncertainty handling and lightweight face tracking.

This module makes the *displayed* prediction steadier. It does not make the
underlying model more accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face_detector import Detection
from .pipeline import FaceResult

UNCERTAIN_LABEL = "uncertain"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes: 0 = no overlap, 1 = identical."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class ScoreSmoother:
    """Exponential moving average over a class-score vector: s = a*p + (1-a)*s_prev."""

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def update(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self._state is None or self._state.shape != scores.shape:
            self._state = scores.copy()  # first frame: nothing to blend with
        else:
            self._state = self.alpha * scores + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


def decide_label(scores: np.ndarray, labels: list[str], threshold: float) -> tuple[str, float, bool]:
    """Return (label, top_score, is_uncertain). Low top score -> 'uncertain'."""
    index = int(np.argmax(scores))
    top_score = float(scores[index])
    if top_score < threshold:
        return UNCERTAIN_LABEL, top_score, True
    return labels[index], top_score, False


@dataclass(frozen=True)
class TrackedFace:
    track_id: int
    detection: Detection
    label: str  # smoothed label, or "uncertain"
    score: float  # smoothed top score
    uncertain: bool
    raw_label: str  # this frame's unsmoothed prediction
    raw_score: float


@dataclass
class _Track:
    track_id: int
    box: tuple[int, int, int, int]
    smoother: ScoreSmoother
    missed: int = 0


class FaceTracker:
    """Keeps a separate smoothing history per face, matched across frames by IoU."""

    def __init__(
        self,
        labels: list[str],
        alpha: float = 0.3,
        threshold: float = 0.40,
        iou_threshold: float = 0.3,
        max_missed: int = 10,
    ) -> None:
        self.labels = labels
        self.alpha = alpha
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, results: list[FaceResult]) -> list[TrackedFace]:
        # 1. Score every (existing track, new detection) pair by box overlap.
        pairs = []
        for ti, track in enumerate(self._tracks):
            for ri, result in enumerate(results):
                overlap = iou(track.box, result.detection.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, ti, ri))

        # 2. Greedy matching: best overlaps first, each track/detection used once.
        assigned: dict[int, int] = {}  # result index -> track index
        used_tracks: set[int] = set()
        for _, ti, ri in sorted(pairs, reverse=True):
            if ri in assigned or ti in used_tracks:
                continue
            assigned[ri] = ti
            used_tracks.add(ti)

        # 3. Tracks with no match this frame age by one.
        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.missed += 1

        # 4. Update matched tracks, start new ones, then smooth and decide labels.
        tracked: list[TrackedFace] = []
        for ri, result in enumerate(results):
            if ri in assigned:
                track = self._tracks[assigned[ri]]
                track.box = result.detection.box
                track.missed = 0
            else:
                track = _Track(self._next_id, result.detection.box, ScoreSmoother(self.alpha))
                self._next_id += 1
                self._tracks.append(track)

            smoothed = track.smoother.update(result.scores)
            label, score, uncertain = decide_label(smoothed, self.labels, self.threshold)
            raw_index = int(np.argmax(result.scores))
            tracked.append(
                TrackedFace(
                    track_id=track.track_id,
                    detection=result.detection,
                    label=label,
                    score=score,
                    uncertain=uncertain,
                    raw_label=self.labels[raw_index],
                    raw_score=float(result.scores[raw_index]),
                )
            )

        # 5. Forget faces that have been gone for too long.
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return tracked
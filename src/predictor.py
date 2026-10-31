"""Load the trained Keras model and turn model inputs into class scores."""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np


class EmotionPredictor:
    """Wraps the CNN and its label list so the two can never be mismatched silently."""

    def __init__(self, model_path: str | Path, labels_path: str | Path) -> None:
        model_path, labels_path = Path(model_path), Path(labels_path)
        for path in (model_path, labels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required file not found: {path}")

        # compile=False: we only run inference, so no optimizer or loss is needed.
        self._model = keras.models.load_model(model_path, compile=False)
        self.labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))

        n_outputs = self._model.output_shape[-1]
        if n_outputs != len(self.labels):
            raise ValueError(
                f"Model has {n_outputs} outputs but labels.json has {len(self.labels)} labels."
            )

        # Warm-up: the first call is slower because the model builds its execution graph.
        self._model(np.zeros((1, 48, 48, 1), dtype="float32"), training=False)

    def predict_scores(self, model_input: np.ndarray) -> np.ndarray:
        """Return class scores of shape (N, num_classes) for input of shape (N, 48, 48, 1)."""
        if (
            not isinstance(model_input, np.ndarray)
            or model_input.ndim != 4
            or model_input.shape[1:] != (48, 48, 1)
        ):
            raise ValueError("Expected an array of shape (N, 48, 48, 1).")
        scores = self._model(model_input.astype(np.float32, copy=False), training=False)
        return scores.numpy()

    def top_label(self, scores: np.ndarray) -> tuple[str, float]:
        """Return (label, score) for the highest-scoring class of one score vector."""
        index = int(np.argmax(scores))
        return self.labels[index], float(scores[index])
    
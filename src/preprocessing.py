"""Turn a detected face region into the tensor the CNN expects.

The model was trained on 48x48 grayscale images and contains its own
Rescaling(1/255) layer, so this module must NOT divide by 255.
"""

from __future__ import annotations

import cv2
import numpy as np

MODEL_INPUT_SIZE = 48


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Return the face region of a frame, or None if the box is invalid."""
    if not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x, y, w, h = box
    x1, y1 = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x + w), width), min(int(y + h), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def to_model_input(image: np.ndarray) -> np.ndarray | None:
    """Convert a BGR or grayscale uint8 image to shape (1, 48, 48, 1), float32, 0-255.

    Returns None for invalid input instead of raising, so a live loop can skip the frame.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        return None
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return None

    # INTER_AREA is the right choice when shrinking; INTER_LINEAR when enlarging.
    shrinking = gray.shape[0] * gray.shape[1] > MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=interpolation)

    return resized.astype(np.float32).reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)
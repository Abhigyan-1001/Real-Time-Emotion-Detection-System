import numpy as np
import pytest

from src.preprocessing import crop_face, to_model_input


def make_bgr(height=100, width=120, value=None):
    if value is not None:
        return np.full((height, width, 3), value, dtype=np.uint8)
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_output_shape_and_dtype():
    out = to_model_input(make_bgr())
    assert out.shape == (1, 48, 48, 1)
    assert out.dtype == np.float32


def test_values_stay_in_0_255_range_and_are_not_rescaled():
    # The model has its own Rescaling(1/255) layer, so we must NOT divide by 255 here.
    out = to_model_input(make_bgr(value=255))
    assert out.min() >= 0.0
    assert out.max() == pytest.approx(255.0)


def test_grayscale_input_is_accepted():
    gray = np.random.default_rng(1).integers(0, 256, size=(64, 64), dtype=np.uint8)
    assert to_model_input(gray).shape == (1, 48, 48, 1)


def test_small_image_is_enlarged():
    assert to_model_input(make_bgr(height=20, width=20)).shape == (1, 48, 48, 1)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "not an image",
        np.zeros((0, 0, 3), dtype=np.uint8),       # empty
        np.zeros((10, 10, 3), dtype=np.float32),   # wrong dtype
        np.zeros((10, 10, 4), dtype=np.uint8),     # 4 channels
        np.zeros((4, 10, 10, 3), dtype=np.uint8),  # too many dimensions
    ],
)
def test_invalid_inputs_return_none(bad):
    assert to_model_input(bad) is None


def test_crop_returns_the_requested_region():
    frame = make_bgr(100, 200)
    crop = crop_face(frame, (20, 10, 40, 30))
    assert np.array_equal(crop, frame[10:40, 20:60])


def test_crop_is_clipped_to_the_frame():
    frame = make_bgr(100, 200)
    crop = crop_face(frame, (-20, -10, 60, 50))  # starts outside the frame
    assert crop.shape == (40, 40, 3)


def test_crop_fully_outside_returns_none():
    assert crop_face(make_bgr(100, 200), (300, 300, 50, 50)) is None


def test_crop_with_zero_size_returns_none():
    assert crop_face(make_bgr(100, 200), (10, 10, 0, 20)) is None


def test_crop_of_non_array_returns_none():
    assert crop_face(None, (0, 0, 10, 10)) is None
    
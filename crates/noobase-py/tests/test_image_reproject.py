"""Tests for noobase.image.reproject_exact."""

import numpy as np
import pytest

import noobase


TOLERANCE = 1e-12


def _identity_corners(height_out: int, width_out: int) -> np.ndarray:
    """Corner field that makes the output grid coincide with the input.

    Output pixel (i, j) should sample input pixel (i, j). In the
    centers-at-integer convention used by the binding, corner node
    (i_node, j_node) sits at (j_node - 0.5, i_node - 0.5).
    """
    corners = np.zeros((height_out + 1, width_out + 1, 2), dtype=np.float64)
    for i_node in range(height_out + 1):
        for j_node in range(width_out + 1):
            corners[i_node, j_node, 0] = j_node - 0.5
            corners[i_node, j_node, 1] = i_node - 0.5
    return corners


# ---------------------------------------------------------------------------
# Algorithmic contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_identity_reprojection_preserves_image_footprint_and_weight(dtype):
    image_in = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=dtype,
    )
    corners = _identity_corners(3, 3)
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    assert image.dtype == np.float64
    assert footprint.dtype == np.float64
    assert weight.dtype == np.float64
    assert image.shape == (3, 3)
    assert footprint.shape == (3, 3)
    assert weight.shape == (3, 3)
    np.testing.assert_allclose(image, image_in.astype(np.float64), atol=TOLERANCE)
    np.testing.assert_allclose(footprint, 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight, 1.0, atol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_half_pixel_shift_linear_average_of_two_columns(dtype):
    image_in = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
    corners = _identity_corners(1, 4)
    corners[..., 0] += 0.5
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    # Output pixel j ends up averaging input pixels j and j+1 equally, except
    # the trailing column which only sees input pixel 3 (the rest falls off
    # the right edge of the input).
    np.testing.assert_allclose(image[0, 0], 1.5, atol=TOLERANCE)
    np.testing.assert_allclose(image[0, 1], 2.5, atol=TOLERANCE)
    np.testing.assert_allclose(image[0, 2], 3.5, atol=TOLERANCE)
    np.testing.assert_allclose(image[0, 3], 4.0, atol=TOLERANCE)
    np.testing.assert_allclose(footprint[0, :3], 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(footprint[0, 3], 0.5, atol=TOLERANCE)
    np.testing.assert_allclose(weight[0, :3], 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight[0, 3], 0.5, atol=TOLERANCE)


def test_nan_input_separates_footprint_from_weight():
    image_in = np.array([[1.0, np.nan, 3.0]], dtype=np.float64)
    corners = _identity_corners(1, 3)
    corners[..., 0] += 0.5
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    # Output pixel 0: input 0 (1.0) + input 1 (NaN), 50/50.
    # image = 1.0; footprint = 1.0 (fully covered geometrically);
    # weight = 0.5 (only non-NaN half contributes).
    np.testing.assert_allclose(image[0, 0], 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(footprint[0, 0], 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight[0, 0], 0.5, atol=TOLERANCE)
    np.testing.assert_allclose(image[0, 1], 3.0, atol=TOLERANCE)
    np.testing.assert_allclose(footprint[0, 1], 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight[0, 1], 0.5, atol=TOLERANCE)
    # Output pixel 2 hits input pixel 2 + off-grid (out of bounds, no
    # input). footprint = weight = 0.5.
    np.testing.assert_allclose(image[0, 2], 3.0, atol=TOLERANCE)
    np.testing.assert_allclose(footprint[0, 2], 0.5, atol=TOLERANCE)
    np.testing.assert_allclose(weight[0, 2], 0.5, atol=TOLERANCE)


def test_nan_in_corners_zeroes_both_footprint_and_weight():
    image_in = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    corners = _identity_corners(2, 2)
    corners[1, 1, 0] = np.nan
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    # The NaN corner is shared by all 4 output pixels.
    assert np.all(np.isnan(image))
    np.testing.assert_array_equal(footprint, 0.0)
    np.testing.assert_array_equal(weight, 0.0)


def test_output_pixels_outside_input_footprint_are_nan_zero():
    image_in = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    # Place the entire output grid far from the input image.
    corners = np.zeros((3, 3, 2), dtype=np.float64)
    for i_node in range(3):
        for j_node in range(3):
            corners[i_node, j_node, 0] = 100.0 + j_node
            corners[i_node, j_node, 1] = 100.0 + i_node
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    assert np.all(np.isnan(image))
    np.testing.assert_array_equal(footprint, 0.0)
    np.testing.assert_array_equal(weight, 0.0)


# ---------------------------------------------------------------------------
# dtype dispatch and validation
# ---------------------------------------------------------------------------


def test_f32_and_f64_inputs_produce_identical_geometry_outputs():
    height, width = 8, 10
    image_f64 = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            image_f64[i, j] = i * 0.5 + j * 0.25 + 1.0
    image_f32 = image_f64.astype(np.float32)
    corners = _identity_corners(height, width)
    corners[..., 0] += 0.3
    corners[..., 1] -= 0.2

    image_a, footprint_a, weight_a = noobase.image.reproject_exact(image_f64, corners)
    image_b, footprint_b, weight_b = noobase.image.reproject_exact(image_f32, corners)
    assert image_a.dtype == np.float64
    assert image_b.dtype == np.float64
    # Geometry-only outputs must match bit-exactly because they do not
    # depend on the image dtype.
    np.testing.assert_array_equal(footprint_a, footprint_b)
    np.testing.assert_array_equal(weight_a, weight_b)
    # Image values agree up to f32 precision.
    np.testing.assert_allclose(image_a, image_b, atol=1e-5, rtol=1e-5)


def test_pixel_corners_non_float64_raises():
    image_in = np.zeros((4, 4), dtype=np.float64)
    corners = _identity_corners(4, 4).astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        noobase.image.reproject_exact(image_in, corners)


def test_image_in_wrong_dtype_raises():
    image_in = np.zeros((4, 4), dtype=np.int32)
    corners = _identity_corners(4, 4)
    with pytest.raises(ValueError, match="float32 or float64"):
        noobase.image.reproject_exact(image_in, corners)


def test_pixel_corners_wrong_last_dim_raises():
    image_in = np.zeros((4, 4), dtype=np.float64)
    corners = np.zeros((5, 5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="last dim"):
        noobase.image.reproject_exact(image_in, corners)


def test_pixel_corners_too_small_front_dims_raises():
    image_in = np.zeros((4, 4), dtype=np.float64)
    corners = np.zeros((1, 5, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=">= 2"):
        noobase.image.reproject_exact(image_in, corners)


# ---------------------------------------------------------------------------
# make_pixel_corners helper
# ---------------------------------------------------------------------------


def _identity_pixel_to_world(x, y):
    return (x, y)


def _identity_world_to_pixel(x, y):
    return (x, y)


def test_make_pixel_corners_identity_matches_manual_construction():
    corners = noobase.image.make_pixel_corners(
        (3, 4),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
    )
    expected = _identity_corners(3, 4)
    assert corners.dtype == np.float64
    assert corners.shape == (4, 5, 2)
    np.testing.assert_allclose(corners, expected, atol=TOLERANCE)


def test_make_pixel_corners_applies_half_pixel_offset_to_callables():
    seen = {}

    def capturing_pixel_to_world(x, y):
        seen["x"] = x.copy()
        seen["y"] = y.copy()
        return (x, y)

    noobase.image.make_pixel_corners(
        (2, 2),
        target_pixel_to_world=capturing_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
    )
    # Corner nodes must arrive at the user's callable already shifted by
    # -0.5 so the caller does not have to remember the convention.
    expected_x = np.array(
        [
            [-0.5, 0.5, 1.5],
            [-0.5, 0.5, 1.5],
            [-0.5, 0.5, 1.5],
        ],
        dtype=np.float64,
    )
    expected_y = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(seen["x"], expected_x)
    np.testing.assert_array_equal(seen["y"], expected_y)


def test_make_pixel_corners_translation_through_chain():
    # World coords carry a constant offset; the inverse undoes it. The
    # composed chain is the identity, so the corners must equal the
    # identity corner field.
    def target_pixel_to_world(x, y):
        return (x + 5.0, y - 3.0)

    def source_world_to_pixel(ra, dec):
        return (ra - 5.0, dec + 3.0)

    corners = noobase.image.make_pixel_corners(
        (3, 3),
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
    )
    np.testing.assert_allclose(corners, _identity_corners(3, 3), atol=TOLERANCE)


def test_make_pixel_corners_end_to_end_with_reproject_exact():
    # Identity chain -> corners produce an identity reprojection.
    image_in = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    corners = noobase.image.make_pixel_corners(
        image_in.shape,
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
    )
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    np.testing.assert_allclose(image, image_in, atol=TOLERANCE)
    np.testing.assert_allclose(footprint, 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight, 1.0, atol=TOLERANCE)

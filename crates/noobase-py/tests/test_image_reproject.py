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


# ---------------------------------------------------------------------------
# make_pixel_corners with coarse_step
# ---------------------------------------------------------------------------


def test_coarse_step_identity_matches_dense_path():
    # Identity is linear, and Catmull-Rom with linear-extrapolation
    # boundaries is exact on linear corner fields. The dense and coarse
    # paths must therefore agree up to the rounding noise of the bicubic
    # accumulation (a handful of f64 ulps, well under 1e-12).
    dense = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
    )
    coarse = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(4, 6),
    )
    np.testing.assert_allclose(coarse, dense, atol=1e-12)


def test_coarse_step_affine_chain_matches_dense_path():
    # Affine chain (rotation + translation in world space) is also
    # exactly representable by a linear corner field, so coarse and
    # dense paths must agree to within numerical noise.
    def target_pixel_to_world(x, y):
        # Rotate by 30 degrees and translate.
        cos_theta = np.cos(np.deg2rad(30.0))
        sin_theta = np.sin(np.deg2rad(30.0))
        return (cos_theta * x - sin_theta * y + 17.0, sin_theta * x + cos_theta * y - 5.0)

    def source_world_to_pixel(ra, dec):
        cos_theta = np.cos(np.deg2rad(30.0))
        sin_theta = np.sin(np.deg2rad(30.0))
        x_world = ra - 17.0
        y_world = dec + 5.0
        return (cos_theta * x_world + sin_theta * y_world, -sin_theta * x_world + cos_theta * y_world)

    dense = noobase.image.make_pixel_corners(
        (32, 32),
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
    )
    coarse = noobase.image.make_pixel_corners(
        (32, 32),
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
        coarse_step=(8, 8),
    )
    np.testing.assert_allclose(coarse, dense, atol=1e-12)


def test_coarse_step_invokes_callables_only_on_coarse_grid():
    call_count = {"target": 0, "source": 0}
    captured_shape = {}

    def target_pixel_to_world(x, y):
        call_count["target"] += 1
        captured_shape["target"] = x.shape
        return (x, y)

    def source_world_to_pixel(ra, dec):
        call_count["source"] += 1
        captured_shape["source"] = ra.shape
        return (ra, dec)

    noobase.image.make_pixel_corners(
        (64, 64),
        target_pixel_to_world=target_pixel_to_world,
        source_world_to_pixel=source_world_to_pixel,
        coarse_step=(16, 32),
    )
    # Each callable evaluated exactly once.
    assert call_count == {"target": 1, "source": 1}
    # Coarse corner grid has shape (H/step_h + 1, W/step_w + 1).
    assert captured_shape["target"] == (5, 3)
    assert captured_shape["source"] == (5, 3)


def test_coarse_step_non_divisible_raises():
    with pytest.raises(ValueError, match="must divide target_shape"):
        noobase.image.make_pixel_corners(
            (7, 8),
            target_pixel_to_world=_identity_pixel_to_world,
            source_world_to_pixel=_identity_world_to_pixel,
            coarse_step=(3, 2),
        )


def test_coarse_step_non_positive_raises():
    with pytest.raises(ValueError, match="must be positive"):
        noobase.image.make_pixel_corners(
            (8, 8),
            target_pixel_to_world=_identity_pixel_to_world,
            source_world_to_pixel=_identity_world_to_pixel,
            coarse_step=(0, 4),
        )


def test_coarse_step_non_sequence_raises():
    with pytest.raises(ValueError, match="length-2 sequence"):
        noobase.image.make_pixel_corners(
            (8, 8),
            target_pixel_to_world=_identity_pixel_to_world,
            source_world_to_pixel=_identity_world_to_pixel,
            coarse_step=4,
        )


def test_coarse_step_wrong_length_raises():
    with pytest.raises(ValueError, match="length-2 sequence"):
        noobase.image.make_pixel_corners(
            (8, 8),
            target_pixel_to_world=_identity_pixel_to_world,
            source_world_to_pixel=_identity_world_to_pixel,
            coarse_step=(4, 4, 4),
        )


def test_coarse_step_non_integer_components_raises():
    with pytest.raises(ValueError, match="exact integers"):
        noobase.image.make_pixel_corners(
            (8, 8),
            target_pixel_to_world=_identity_pixel_to_world,
            source_world_to_pixel=_identity_world_to_pixel,
            coarse_step=(2.5, 4),
        )


def test_coarse_step_accepts_list():
    reference = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(4, 6),
    )
    from_list = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=[4, 6],
    )
    np.testing.assert_array_equal(from_list, reference)


def test_coarse_step_accepts_ndarray():
    reference = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(4, 6),
    )
    from_ndarray = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=np.array([4, 6]),
    )
    np.testing.assert_array_equal(from_ndarray, reference)


def test_coarse_step_accepts_integer_valued_floats():
    # 64.0 is an exact integer; accept it (matches numpy's general
    # convention for shape-like parameters).
    reference = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(4, 6),
    )
    from_floats = noobase.image.make_pixel_corners(
        (16, 24),
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(4.0, 6.0),
    )
    np.testing.assert_array_equal(from_floats, reference)


def test_coarse_step_end_to_end_with_reproject_exact():
    image_in = np.arange(64, dtype=np.float64).reshape((8, 8))
    corners = noobase.image.make_pixel_corners(
        image_in.shape,
        target_pixel_to_world=_identity_pixel_to_world,
        source_world_to_pixel=_identity_world_to_pixel,
        coarse_step=(2, 4),
    )
    image, footprint, weight = noobase.image.reproject_exact(image_in, corners)
    np.testing.assert_allclose(image, image_in, atol=TOLERANCE)
    np.testing.assert_allclose(footprint, 1.0, atol=TOLERANCE)
    np.testing.assert_allclose(weight, 1.0, atol=TOLERANCE)

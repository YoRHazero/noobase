"""Tests for the noobase.Grid binding and the noobase.overlap submodule."""

import numpy as np
import pytest

import noobase


TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_grid_construct_linear_centers(dtype):
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    grid = noobase.Grid(values, spacing="linear", kind="centers")
    assert len(grid) == 4
    assert grid.spacing == "linear"
    assert grid.kind == "centers"
    assert grid.dtype == np.dtype(dtype)
    np.testing.assert_allclose(grid.values, values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_grid_construct_log(dtype):
    values = np.array([1.0, 10.0, 100.0, 1000.0], dtype=dtype)
    grid = noobase.Grid(values, spacing="log", kind="centers")
    assert grid.spacing == "log"
    assert grid.dtype == np.dtype(dtype)


def test_grid_construct_default_spacing_and_kind():
    grid = noobase.Grid(np.array([1.0, 2.0, 3.0]))
    assert grid.spacing == "linear"
    assert grid.kind == "centers"


def test_grid_construct_returns_value_copy_not_view():
    source = np.array([1.0, 2.0, 3.0])
    grid = noobase.Grid(source)
    out = grid.values
    out[0] = 999.0
    # Subsequent calls return a fresh copy unaffected by mutation of the
    # previously returned array.
    assert grid.values[0] == pytest.approx(1.0)


def test_grid_rejects_float16():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    with pytest.raises(ValueError, match="float32 or float64"):
        noobase.Grid(values)


def test_grid_rejects_int():
    values = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="float32 or float64"):
        noobase.Grid(values)


def test_grid_rejects_invalid_spacing():
    with pytest.raises(ValueError, match="invalid spacing"):
        noobase.Grid(np.array([1.0, 2.0, 3.0]), spacing="bogus")


def test_grid_rejects_invalid_kind():
    with pytest.raises(ValueError, match="invalid kind"):
        noobase.Grid(np.array([1.0, 2.0, 3.0]), kind="weird")


def test_grid_rejects_non_monotonic():
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        noobase.Grid(np.array([1.0, 3.0, 2.0]))


def test_grid_rejects_log_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        noobase.Grid(np.array([-1.0, 1.0, 2.0]), spacing="log")


def test_grid_rejects_too_few_values():
    with pytest.raises(ValueError, match="at least 2 values"):
        noobase.Grid(np.array([1.0]))


# ---------------------------------------------------------------------------
# Grid.linspace / logspace / from_array
# ---------------------------------------------------------------------------


def test_linspace_default_dtype_is_f64():
    grid = noobase.Grid.linspace(0.0, 10.0, 11)
    assert grid.dtype == np.dtype(np.float64)
    assert len(grid) == 11
    np.testing.assert_allclose(grid.values, np.linspace(0.0, 10.0, 11))


def test_linspace_with_f32_dtype():
    grid = noobase.Grid.linspace(0.0, 1.0, 5, dtype=np.float32)
    assert grid.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(grid.values, np.linspace(0.0, 1.0, 5, dtype=np.float32))


def test_logspace_default_dtype_is_f64():
    grid = noobase.Grid.logspace(1.0, 1000.0, 4)
    assert grid.spacing == "log"
    assert grid.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(grid.values, [1.0, 10.0, 100.0, 1000.0], rtol=TOLERANCE)


def test_logspace_with_f32_dtype():
    grid = noobase.Grid.logspace(1.0, 100.0, 3, dtype=np.float32)
    assert grid.dtype == np.dtype(np.float32)


def test_logspace_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        noobase.Grid.logspace(-1.0, 10.0, 5)


def test_linspace_rejects_n_below_two():
    with pytest.raises(ValueError, match="n >= 2"):
        noobase.Grid.linspace(0.0, 1.0, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_from_array_detects_log_spacing(dtype):
    values = np.array([1.0, 10.0, 100.0, 1000.0], dtype=dtype)
    grid = noobase.Grid.from_array(values)
    assert grid.spacing == "log"
    assert grid.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_from_array_detects_linear_spacing(dtype):
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    grid = noobase.Grid.from_array(values)
    assert grid.spacing == "linear"
    assert grid.dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# Grid methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_edges_preserves_dtype(dtype):
    grid = noobase.Grid.linspace(0.5, 4.5, 5, dtype=dtype)
    edges = grid.to_edges()
    assert edges.kind == "edges"
    assert edges.dtype == np.dtype(dtype)
    assert len(edges) == 6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_centers_preserves_dtype(dtype):
    edges = noobase.Grid(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=dtype),
        spacing="linear",
        kind="edges",
    )
    centers = edges.to_centers()
    assert centers.kind == "centers"
    assert centers.dtype == np.dtype(dtype)
    assert len(centers) == 3


def test_is_uniform_true_for_linspace():
    grid = noobase.Grid.linspace(0.0, 1.0, 11)
    assert grid.is_uniform(1e-12) is True


def test_is_uniform_false_for_irregular():
    grid = noobase.Grid(np.array([0.0, 1.0, 2.0, 4.0]))
    assert grid.is_uniform(1e-6) is False


# ---------------------------------------------------------------------------
# overlap.rebin
# ---------------------------------------------------------------------------


def _edges_grid(values, dtype):
    return noobase.Grid(np.array(values, dtype=dtype), spacing="linear", kind="edges")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_rebin_identity(dtype):
    grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    output = noobase.overlap.rebin(grid, values, grid)
    assert output.dtype == np.dtype(dtype)
    np.testing.assert_allclose(output, values, rtol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_rebin_downsample_two_to_one(dtype):
    source = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    target = _edges_grid([0.0, 2.0, 4.0], dtype)
    values = np.array([1.0, 3.0, 5.0, 9.0], dtype=dtype)
    output = noobase.overlap.rebin(source, values, target)
    assert output.dtype == np.dtype(dtype)
    np.testing.assert_allclose(output, [2.0, 7.0], rtol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_rebin_partial_coverage(dtype):
    # Target bin [-1, 1) only overlaps source [0, 1) -> output = 1 * value / 2 = 0.5
    source = _edges_grid([0.0, 1.0, 2.0], dtype)
    target = _edges_grid([-1.0, 1.0, 3.0], dtype)
    values = np.array([1.0, 1.0], dtype=dtype)
    output = noobase.overlap.rebin(source, values, target)
    np.testing.assert_allclose(output, [0.5, 0.5], rtol=TOLERANCE)


def test_overlap_rebin_rejects_dtype_mismatch_grids():
    source = _edges_grid([0.0, 1.0, 2.0], np.float32)
    target = _edges_grid([0.0, 1.0, 2.0], np.float64)
    values = np.array([1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.overlap.rebin(source, values, target)


def test_overlap_rebin_rejects_dtype_mismatch_values():
    source = _edges_grid([0.0, 1.0, 2.0], np.float32)
    target = _edges_grid([0.0, 1.0, 2.0], np.float32)
    values = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.overlap.rebin(source, values, target)


def test_overlap_rebin_rejects_length_mismatch():
    source = _edges_grid([0.0, 1.0, 2.0, 3.0], np.float64)
    target = _edges_grid([0.0, 1.5, 3.0], np.float64)
    values = np.array([1.0, 2.0], dtype=np.float64)  # 2 != 3 source bins
    with pytest.raises(ValueError, match="does not match"):
        noobase.overlap.rebin(source, values, target)


# ---------------------------------------------------------------------------
# overlap.rebin_variance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_rebin_variance_identity(dtype):
    grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    variance = np.array([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    output = noobase.overlap.rebin_variance(grid, variance, grid)
    assert output.dtype == np.dtype(dtype)
    np.testing.assert_allclose(output, variance, rtol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_rebin_variance_downsample(dtype):
    source = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    target = _edges_grid([0.0, 2.0, 4.0], dtype)
    variance = np.full(4, 0.8, dtype=dtype)
    output = noobase.overlap.rebin_variance(source, variance, target)
    np.testing.assert_allclose(output, [0.4, 0.4], rtol=TOLERANCE)


def test_overlap_rebin_variance_rejects_dtype_mismatch():
    source = _edges_grid([0.0, 1.0, 2.0], np.float32)
    target = _edges_grid([0.0, 1.0, 2.0], np.float32)
    variance = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.overlap.rebin_variance(source, variance, target)


# ---------------------------------------------------------------------------
# overlap.coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_coverage_fully_inside(dtype):
    source = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype)
    target = _edges_grid([1.0, 2.5, 4.0], dtype)
    output = noobase.overlap.coverage(source, target)
    assert output.dtype == np.dtype(dtype)
    np.testing.assert_allclose(output, [1.0, 1.0], rtol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_coverage_half_edge_bins(dtype):
    source = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    target = _edges_grid([-1.0, 1.0, 3.0, 5.0], dtype)
    output = noobase.overlap.coverage(source, target)
    np.testing.assert_allclose(output, [0.5, 1.0, 0.5], rtol=TOLERANCE)


def test_overlap_coverage_rejects_dtype_mismatch():
    source = _edges_grid([0.0, 1.0, 2.0], np.float32)
    target = _edges_grid([0.0, 1.0, 2.0], np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.overlap.coverage(source, target)

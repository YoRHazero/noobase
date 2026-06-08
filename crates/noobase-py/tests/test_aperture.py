"""Tests for the aperture region-growing PyO3 binding.

The Rust core algorithm is pinned by its in-crate unit tests; the tests
here exercise *boundary translation* only — numpy in / pyclass out,
``GrowError`` -> ``ValueError``, and the two pure-Python wrapper
features (``snr_threshold`` auto-follows ``err``; ``seed_pixels``
accepts a numpy ``(N, 2)`` integer array).
"""

import numpy as np
import pytest

from noobase.aperture import GrowthResult, grow_mask


def _centered_gaussian(n: int, sigma: float, amplitude: float) -> np.ndarray:
    center = (n - 1) / 2.0
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return amplitude * np.exp(
        -((rows - center) ** 2 + (cols - center) ** 2) / (2.0 * sigma**2)
    )


def test_success_roundtrip_with_auto_snr():
    # err is provided but snr_threshold is left unset -> the wrapper
    # auto-enables SNR at threshold 2.0. The Gaussian's outer wings
    # decay below SNR threshold well inside the cutout, so growth must
    # terminate with stop_reason == "snr_below".
    n = 21
    data = _centered_gaussian(n, sigma=2.0, amplitude=100.0)
    err = np.ones((n, n), dtype=np.float64)

    result = grow_mask(
        data,
        [(10, 10)],
        err=err,
        gradient_ratio_threshold=None,
        check_interval=1,
        min_pixels_before_stop_check=5,
        annulus_thickness=1,
    )

    assert isinstance(result, GrowthResult)
    assert result.mask.dtype == np.bool_
    assert result.mask.shape == (n, n)
    assert result.stop_reason == "snr_below"
    assert result.n_iterations > 0
    assert result.mask[10, 10]


def test_seed_out_of_bounds_raises_value_error():
    # Core GrowError::SeedOutOfBounds must surface as ValueError with
    # the verbatim Display message.
    data = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="out of bounds"):
        grow_mask(data, [(3, 0)], gradient_ratio_threshold=1.0)


def test_label_map_without_allowed_raises_value_error():
    # Binding-layer pairing check (Rust core cannot express this
    # invariant on its own because it takes a single LabelInput).
    data = np.zeros((3, 3), dtype=np.float64)
    label_map = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="label_map and label_allowed"):
        grow_mask(
            data,
            [(1, 1)],
            label_map=label_map,
            label_allowed=None,
            gradient_ratio_threshold=1.0,
        )


def test_detection_kwarg_drives_growth_not_data():
    # The detection image must reach the core as the heap key: a bright
    # ridge to the right in detection, an equally bright ridge going
    # down in data. Growth must follow detection (reach the right edge)
    # and ignore data's ridge.
    n = 11
    detection = np.full((n, n), 1.0, dtype=np.float64)
    detection[5, 5:] = 100.0
    data = np.full((n, n), 1.0, dtype=np.float64)
    data[5:, 5] = 100.0
    err = np.ones((n, n), dtype=np.float64)

    result = grow_mask(
        data,
        [(5, 5)],
        detection=detection,
        err=err,
        snr_threshold=0.5,
        snr_hysteresis=10**9,  # SNR stop can never fire
        gradient_ratio_threshold=None,
        shape_weight=0.0,
        min_neighbor_support=1,
        check_interval=1,
        min_pixels_before_stop_check=1,
    )

    assert result.mask[5, 10], "growth must follow the detection ridge"
    assert not result.mask[10, 5], "growth must not follow the data ridge"


def test_min_neighbor_support_too_large_raises_value_error():
    # New core GrowError variant must surface as ValueError.
    data = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="min_neighbor_support"):
        grow_mask(
            data,
            [(1, 1)],
            gradient_ratio_threshold=1.0,
            connectivity="four",
            min_neighbor_support=5,
        )


def test_gradient_percentile_invalid_raises_value_error():
    # New core GrowError variant for an inverted percentile band.
    data = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="gradient percentile band"):
        grow_mask(
            data,
            [(1, 1)],
            gradient_ratio_threshold=1.0,
            gradient_lo_percentile=99.0,
            gradient_hi_percentile=75.0,
        )


def test_default_shape_weight_scaling_runs_without_err():
    # With err omitted the SNR stop is auto-disabled and the wrapper
    # scales the default shape_weight by mad_std(data) (no err path).
    # Exercises that scaling without crashing and returns a usable mask.
    n = 21
    data = _centered_gaussian(n, sigma=2.0, amplitude=100.0)

    result = grow_mask(data, [(10, 10)])

    assert isinstance(result, GrowthResult)
    assert result.mask[10, 10]

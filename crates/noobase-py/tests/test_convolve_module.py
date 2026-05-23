"""Tests for the noobase.convolve binding surface."""

import numpy as np
import pytest

import noobase


def test_gaussian1d_odd_length_and_sum_normalization():
    kernel = noobase.convolve.gaussian1d(1.0, truncate=4.0)
    assert kernel.dtype == np.float64
    assert kernel.shape == (9,)
    assert kernel.shape[0] % 2 == 1
    assert np.isclose(kernel.sum(), 1.0)


def test_gaussian1d_l2_normalization():
    kernel = noobase.convolve.gaussian1d(1.5, normalization="l2")
    assert np.isclose(np.sum(kernel * kernel), 1.0)


def test_gaussian1d_float32_dtype():
    kernel = noobase.convolve.gaussian1d(1.0, dtype=np.float32)
    assert kernel.dtype == np.float32


def test_gaussian1d_point_sampled_known_value():
    kernel = noobase.convolve.gaussian1d(
        1.0, sampling="point_sampled", normalization="none"
    )
    center = kernel.shape[0] // 2
    assert np.isclose(kernel[center], 1.0)
    assert np.isclose(kernel[center + 1], np.exp(-0.5))


def test_gaussian1d_invalid_sigma_raises():
    with pytest.raises(ValueError, match="sigma must be positive"):
        noobase.convolve.gaussian1d(0.0)


def test_gaussian1d_invalid_sampling_raises():
    with pytest.raises(ValueError, match="invalid sampling"):
        noobase.convolve.gaussian1d(1.0, sampling="bogus")


def test_conv1d_delta_is_identity():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    kernel = np.array([0.0, 1.0, 0.0])
    output = noobase.convolve.conv1d(signal, kernel)
    np.testing.assert_array_equal(output, signal)


def test_conv1d_is_correlation_not_convolution():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = np.array([1.0, 2.0, 3.0])
    output = noobase.convolve.conv1d(signal, kernel, boundary="zero")
    # out[i] = 1*s[i-1] + 2*s[i] + 3*s[i+1], zero outside
    expected = np.array([8.0, 14.0, 20.0, 26.0, 14.0])
    np.testing.assert_allclose(output, expected)


def test_conv1d_kernel_dtype_must_match():
    signal = np.array([1.0, 2.0], dtype=np.float64)
    kernel = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        noobase.convolve.conv1d(signal, kernel)


def test_conv1d_empty_kernel_raises():
    signal = np.array([1.0, 2.0])
    kernel = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="kernel must be non-empty"):
        noobase.convolve.conv1d(signal, kernel)


def test_conv_axis_matches_per_row_conv1d():
    image = np.arange(12.0).reshape(3, 4)
    kernel = np.array([1.0, 2.0, 3.0])
    via_axis = noobase.convolve.conv_axis(image, kernel, axis=1, boundary="reflect")
    for row_index in range(image.shape[0]):
        expected = noobase.convolve.conv1d(
            image[row_index], kernel, boundary="reflect"
        )
        np.testing.assert_allclose(via_axis[row_index], expected)


def test_conv_axis_invalid_axis_raises():
    image = np.ones((3, 3))
    kernel = np.array([1.0])
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        noobase.convolve.conv_axis(image, kernel, axis=2)


def test_conv2d_delta_is_identity():
    image = np.arange(9.0).reshape(3, 3)
    kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    output = noobase.convolve.conv2d(image, kernel)
    np.testing.assert_array_equal(output, image)


def test_conv2d_renorm_handles_nan_input():
    image = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    kernel = np.ones((3, 3)) / 9.0
    value, weight = noobase.convolve.conv2d_renorm(image, kernel)
    assert value.shape == image.shape
    assert weight.shape == image.shape
    # All-finite interior tap windows still produce finite values.
    assert np.isfinite(value).all()
    # The center cell loses one tap (the NaN); the weight there is 8/9.
    assert np.isclose(weight[1, 1], 8.0 / 9.0)


def test_conv_axis_renorm_handles_nan_input():
    image = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]])
    kernel = np.array([1.0, 1.0, 1.0]) / 3.0
    value, weight = noobase.convolve.conv_axis_renorm(image, kernel, axis=0)
    # Center row's first column: window samples are [1, NaN, 7]; weight loses
    # one tap, so weight = 2/3 there.
    assert np.isclose(weight[1, 0], 2.0 / 3.0)


def test_convolve_module_advertises_core_module():
    assert noobase.convolve.conv1d.__module__ == "noobase._core.convolve"
    assert noobase.convolve.gaussian1d.__module__ == "noobase._core.convolve"

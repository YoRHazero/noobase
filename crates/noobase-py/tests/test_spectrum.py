"""Tests for the noobase.Spectrum binding."""

import numpy as np
import pytest

import noobase


TOLERANCE = 1e-6


def _edges_grid(values, dtype):
    return noobase.Grid(np.array(values, dtype=dtype), spacing="linear", kind="edges")


# ---------------------------------------------------------------------------
# Spectrum construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_construct_with_array_wavelength(dtype):
    wavelength = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    flux = np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    spectrum = noobase.Spectrum(
        wavelength=wavelength,
        flux=flux,
        spacing="linear",
        kind="centers",
    )
    assert spectrum.n_bins == 4
    assert spectrum.dtype == np.dtype(dtype)
    assert isinstance(spectrum.wavelength, noobase.Grid)
    assert spectrum.wavelength.dtype == np.dtype(dtype)
    np.testing.assert_allclose(spectrum.flux, flux, rtol=TOLERANCE)
    assert spectrum.error is None
    assert spectrum.mask is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_construct_with_grid_wavelength(dtype):
    grid = noobase.Grid(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype),
        spacing="linear",
        kind="edges",
    )
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    spectrum = noobase.Spectrum(wavelength=grid, flux=flux)
    assert spectrum.n_bins == 4
    assert spectrum.wavelength.kind == "edges"


def test_construct_grid_plus_spacing_is_rejected():
    grid = noobase.Grid(np.array([1.0, 2.0, 3.0]))
    flux = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="spacing/kind must not be passed"):
        noobase.Spectrum(wavelength=grid, flux=flux, spacing="linear")


def test_construct_grid_plus_kind_is_rejected():
    grid = noobase.Grid(np.array([1.0, 2.0, 3.0]))
    flux = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="spacing/kind must not be passed"):
        noobase.Spectrum(wavelength=grid, flux=flux, kind="centers")


def test_construct_wavelength_dtype_mismatch():
    wavelength = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.Spectrum(wavelength=wavelength, flux=flux)


def test_construct_error_dtype_mismatch():
    wavelength = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    error = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.Spectrum(wavelength=wavelength, flux=flux, error=error)


def test_construct_flux_length_mismatch():
    wavelength = np.array([1.0, 2.0, 3.0])
    flux = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="flux length"):
        noobase.Spectrum(wavelength=wavelength, flux=flux)


def test_construct_error_length_mismatch():
    wavelength = np.array([1.0, 2.0, 3.0])
    flux = np.array([1.0, 2.0, 3.0])
    error = np.array([0.1, 0.1])
    with pytest.raises(ValueError, match="error length"):
        noobase.Spectrum(wavelength=wavelength, flux=flux, error=error)


def test_construct_mask_length_mismatch():
    wavelength = np.array([1.0, 2.0, 3.0])
    flux = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False], dtype=np.bool_)
    with pytest.raises(ValueError, match="mask length"):
        noobase.Spectrum(wavelength=wavelength, flux=flux, mask=mask)


def test_mask_rejects_non_bool_dtype():
    wavelength = np.array([1.0, 2.0, 3.0])
    flux = np.array([1.0, 2.0, 3.0])
    mask = np.array([1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="mask must be"):
        noobase.Spectrum(wavelength=wavelength, flux=flux, mask=mask)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_accessor_roundtrip(dtype):
    wavelength = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    flux = np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    error = np.array([0.5, 0.6, 0.7, 0.8], dtype=dtype)
    mask = np.array([True, True, False, True], dtype=np.bool_)
    spectrum = noobase.Spectrum(
        wavelength=wavelength,
        flux=flux,
        error=error,
        mask=mask,
    )
    assert spectrum.n_bins == 4
    np.testing.assert_allclose(spectrum.flux, flux, rtol=TOLERANCE)
    np.testing.assert_allclose(spectrum.error, error, rtol=TOLERANCE)
    np.testing.assert_array_equal(spectrum.mask, mask)


def test_returned_arrays_are_copies():
    wavelength = np.array([1.0, 2.0, 3.0])
    flux = np.array([10.0, 20.0, 30.0])
    spectrum = noobase.Spectrum(wavelength=wavelength, flux=flux)
    fetched = spectrum.flux
    fetched[0] = -999.0
    # Subsequent access yields an unmutated copy.
    assert spectrum.flux[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# rebin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rebin_with_array_target(dtype):
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([1.0, 3.0, 5.0, 9.0], dtype=dtype)
    spectrum = noobase.Spectrum(wavelength=wavelength, flux=flux)
    target_edges = np.array([0.0, 2.0, 4.0], dtype=dtype)
    rebinned = spectrum.rebin(target_edges, kind="edges")
    assert rebinned.n_bins == 2
    np.testing.assert_allclose(rebinned.flux, [2.0, 7.0], rtol=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rebin_with_grid_target(dtype):
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([1.0, 3.0, 5.0, 9.0], dtype=dtype)
    spectrum = noobase.Spectrum(wavelength=wavelength, flux=flux)
    target = _edges_grid([0.0, 2.0, 4.0], dtype)
    rebinned = spectrum.rebin(target)
    np.testing.assert_allclose(rebinned.flux, [2.0, 7.0], rtol=TOLERANCE)


def test_rebin_rejects_dtype_mismatch_with_target():
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float32)
    flux = np.array([1.0, 3.0, 5.0, 9.0], dtype=np.float32)
    spectrum = noobase.Spectrum(wavelength=wavelength, flux=flux)
    target = _edges_grid([0.0, 2.0, 4.0], np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        spectrum.rebin(target)


def test_rebin_rejects_grid_with_spacing():
    wavelength = _edges_grid([0.0, 1.0, 2.0], np.float64)
    flux = np.array([1.0, 2.0], dtype=np.float64)
    spectrum = noobase.Spectrum(wavelength=wavelength, flux=flux)
    target = _edges_grid([0.0, 2.0], np.float64)
    with pytest.raises(ValueError, match="spacing/kind must not be passed"):
        spectrum.rebin(target, spacing="linear")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rebin_identity_roundtrip(dtype):
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    error = np.array([0.5, 0.4, 0.3, 0.2], dtype=dtype)
    mask = np.array([True, True, False, True], dtype=np.bool_)
    spectrum = noobase.Spectrum(
        wavelength=wavelength,
        flux=flux,
        error=error,
        mask=mask,
    )
    rebinned = spectrum.rebin(wavelength)
    np.testing.assert_allclose(rebinned.flux, flux, rtol=TOLERANCE)
    np.testing.assert_allclose(rebinned.error, error, rtol=TOLERANCE)
    np.testing.assert_array_equal(rebinned.mask, mask)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rebin_mask_propagation_two_to_one(dtype):
    source_wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    target = _edges_grid([0.0, 2.0, 4.0], dtype)
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    mask = np.array([True, False, True, True], dtype=np.bool_)
    spectrum = noobase.Spectrum(
        wavelength=source_wavelength,
        flux=flux,
        mask=mask,
    )
    rebinned = spectrum.rebin(target)
    np.testing.assert_array_equal(rebinned.mask, np.array([False, True]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rebin_downsample_error_quadrature(dtype):
    # Four width-1 source bins with sigma=sqrt(v); target width-2 bins => sigma=sqrt(v/2)
    source_wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    target = _edges_grid([0.0, 2.0, 4.0], dtype)
    variance = dtype(0.8)
    sigma = dtype(np.sqrt(variance))
    flux = np.zeros(4, dtype=dtype)
    error = np.full(4, sigma, dtype=dtype)
    spectrum = noobase.Spectrum(
        wavelength=source_wavelength,
        flux=flux,
        error=error,
    )
    rebinned = spectrum.rebin(target)
    expected = float(np.sqrt(variance / 2.0))
    np.testing.assert_allclose(rebinned.error, [expected, expected], rtol=TOLERANCE)

"""Tests for the noobase.photometry submodule and Spectrum.synthetic_photometry."""

import numpy as np
import pytest

import noobase


TOLERANCE = 1e-6


def _edges_grid(values, dtype):
    return noobase.Grid(np.array(values, dtype=dtype), spacing="linear", kind="edges")


def _centers_grid(values, dtype):
    return noobase.Grid(np.array(values, dtype=dtype), spacing="linear", kind="centers")


# ---------------------------------------------------------------------------
# Free function: noobase.photometry.synthetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_with_all_grid_args_no_error(dtype):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    transmission = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    band_flux, band_error, coverage = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert band_flux == pytest.approx(1.0, rel=TOLERANCE)
    assert band_error is None
    assert coverage == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_with_ndarray_transmission_grid_centers(dtype):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    # Transmission as ndarray of length == transmission_values length -> centers.
    transmission_grid_centers = np.array([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    transmission = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    band_flux, band_error, coverage = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid_centers,
        transmission_values=transmission,
    )
    assert band_flux == pytest.approx(1.0, rel=TOLERANCE)
    assert coverage == pytest.approx(1.0, rel=TOLERANCE)
    assert band_error is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_with_ndarray_transmission_grid_edges(dtype):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    # Transmission as ndarray of length == transmission_values length + 1 -> edges.
    transmission_grid_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    transmission = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    band_flux, _band_error, coverage = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid_edges,
        transmission_values=transmission,
    )
    assert band_flux == pytest.approx(1.0, rel=TOLERANCE)
    assert coverage == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_with_ndarray_spectrum_grid_centers_and_edges(dtype):
    # spectrum_grid passed as ndarray of length == flux length -> centers
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    spec_centers = np.array([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    band_flux_c, _e, cov_c = noobase.photometry.synthetic(
        spectrum_grid=spec_centers,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert band_flux_c == pytest.approx(1.0, rel=TOLERANCE)
    assert cov_c == pytest.approx(1.0, rel=TOLERANCE)

    spec_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
    band_flux_e, _e2, cov_e = noobase.photometry.synthetic(
        spectrum_grid=spec_edges,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert band_flux_e == pytest.approx(1.0, rel=TOLERANCE)
    assert cov_e == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_error_propagation(dtype):
    # 4 bins fully covered, rect filter, energy_weighted.
    # band_error = sigma / sqrt(N) = sigma / 2 for N=4.
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.zeros(4, dtype=dtype)
    sigma = dtype(0.5)
    error = np.full(4, sigma, dtype=dtype)
    transmission = np.ones(4, dtype=dtype)
    band_flux, band_error, coverage = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        spectrum_error=error,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="energy_weighted",
    )
    assert band_flux == pytest.approx(0.0, abs=TOLERANCE)
    assert band_error is not None
    assert band_error == pytest.approx(float(sigma) / 2.0, rel=1e-4)
    assert coverage == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_synthetic_conventions_differ(dtype):
    # Sloped flux: convention matters.
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([0.5, 1.5, 2.5, 3.5], dtype=dtype)
    transmission = np.ones(4, dtype=dtype)
    photon_flux, _, _ = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="photon_counting",
    )
    energy_flux, _, _ = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="energy_weighted",
    )
    assert energy_flux == pytest.approx(2.0, rel=TOLERANCE)
    assert photon_flux == pytest.approx(21.0 / 8.0, rel=TOLERANCE)
    assert abs(photon_flux - energy_flux) > 0.1


def test_synthetic_return_shape():
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    flux = np.array([1.0, 1.0], dtype=np.float64)
    transmission = np.array([1.0, 1.0], dtype=np.float64)
    result = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    band_flux, band_error, coverage = result
    assert isinstance(band_flux, float)
    assert band_error is None
    assert isinstance(coverage, float)


def test_synthetic_default_convention_is_photon_counting():
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    flux = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    transmission = np.ones(4, dtype=np.float64)
    default_flux, _, _ = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    explicit_flux, _, _ = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="photon_counting",
    )
    assert default_flux == pytest.approx(explicit_flux, rel=TOLERANCE)


def test_synthetic_rejects_invalid_convention():
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    flux = np.array([1.0, 1.0], dtype=np.float64)
    transmission = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="invalid convention"):
        noobase.photometry.synthetic(
            spectrum_grid=spectrum_grid,
            spectrum_flux=flux,
            transmission_grid=transmission_grid,
            transmission_values=transmission,
            convention="weird_convention",
        )


def test_synthetic_rejects_dtype_mismatch_flux_vs_transmission():
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0], np.float64)
    flux = np.array([1.0, 1.0], dtype=np.float64)
    transmission = np.array([1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="transmission_values dtype"):
        noobase.photometry.synthetic(
            spectrum_grid=spectrum_grid,
            spectrum_flux=flux,
            transmission_grid=transmission_grid,
            transmission_values=transmission,
        )


def test_synthetic_rejects_wrong_length_spec_grid_ndarray():
    flux = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission = np.ones(4, dtype=np.float64)
    bad_spec_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    with pytest.raises(ValueError, match="does not match paired array"):
        noobase.photometry.synthetic(
            spectrum_grid=bad_spec_grid,
            spectrum_flux=flux,
            transmission_grid=transmission_grid,
            transmission_values=transmission,
        )


# ---------------------------------------------------------------------------
# SyntheticOperator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_operator_construction_coverage_in_unit_interval(dtype):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission_grid = _edges_grid([1.0, 2.0, 3.0], dtype)
    transmission = np.array([1.0, 1.0], dtype=dtype)
    operator = noobase.photometry.SyntheticOperator(
        spectrum_grid=spectrum_grid,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert isinstance(operator.coverage, float)
    assert 0.0 < operator.coverage <= 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("convention", ["photon_counting", "energy_weighted"])
@pytest.mark.parametrize("with_error", [False, True])
def test_operator_apply_matches_synthetic(dtype, convention, with_error):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype)
    transmission_grid = _edges_grid([1.0, 2.0, 3.0, 4.0, 5.0], dtype)
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
    transmission = np.array([0.4, 0.7, 0.9, 0.5], dtype=dtype)
    error = (
        np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35], dtype=dtype) if with_error else None
    )

    direct = noobase.photometry.synthetic(
        spectrum_grid=spectrum_grid,
        spectrum_flux=flux,
        spectrum_error=error,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention=convention,
    )
    operator = noobase.photometry.SyntheticOperator(
        spectrum_grid=spectrum_grid,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention=convention,
    )
    applied = operator.apply(flux, spectrum_error=error)

    rtol = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(applied[0], direct[0], rtol=rtol)
    if with_error:
        np.testing.assert_allclose(applied[1], direct[1], rtol=rtol)
    else:
        assert applied[1] is None
    np.testing.assert_allclose(operator.coverage, direct[2], rtol=rtol)


def test_operator_requires_spectrum_grid_as_grid():
    spec_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission = np.ones(4, dtype=np.float64)
    with pytest.raises(ValueError, match="SyntheticOperator requires spectrum_grid"):
        noobase.photometry.SyntheticOperator(
            spectrum_grid=spec_array,
            transmission_grid=transmission_grid,
            transmission_values=transmission,
        )


def test_operator_accepts_ndarray_transmission_grid_length_match():
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission = np.ones(4, dtype=np.float64)
    # length == transmission length -> centers
    transmission_grid_centers = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    operator = noobase.photometry.SyntheticOperator(
        spectrum_grid=spectrum_grid,
        transmission_grid=transmission_grid_centers,
        transmission_values=transmission,
    )
    assert operator.coverage == pytest.approx(1.0, rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_operator_dtype_matches_channel(dtype):
    spectrum_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission = np.ones(4, dtype=dtype)
    operator = noobase.photometry.SyntheticOperator(
        spectrum_grid=spectrum_grid,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert operator.dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# Spectrum.synthetic_photometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectrum_synthetic_photometry_matches_free_function(dtype):
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    transmission_grid = _edges_grid([1.0, 2.0, 3.0], dtype)
    transmission = np.array([0.5, 0.7], dtype=dtype)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=wavelength, flux=flux)
    free = noobase.photometry.synthetic(
        spectrum_grid=wavelength,
        spectrum_flux=flux,
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    via_method = spectrum.synthetic_photometry(
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    rtol = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(via_method[0], free[0], rtol=rtol)
    assert via_method[1] is None and free[1] is None
    np.testing.assert_allclose(via_method[2], free[2], rtol=rtol)


def test_spectrum_synthetic_photometry_default_convention():
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    flux = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission = np.ones(4, dtype=np.float64)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=wavelength, flux=flux)
    default_result = spectrum.synthetic_photometry(
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    explicit_result = spectrum.synthetic_photometry(
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="photon_counting",
    )
    assert default_result[0] == pytest.approx(explicit_result[0], rel=TOLERANCE)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectrum_synthetic_photometry_propagates_error_when_present(dtype):
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    flux = np.zeros(4, dtype=dtype)
    sigma = dtype(0.5)
    error = np.full(4, sigma, dtype=dtype)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=wavelength, flux=flux, error=error)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], dtype)
    transmission = np.ones(4, dtype=dtype)
    band_flux, band_error, coverage = spectrum.synthetic_photometry(
        transmission_grid=transmission_grid,
        transmission_values=transmission,
        convention="energy_weighted",
    )
    assert band_flux == pytest.approx(0.0, abs=TOLERANCE)
    assert band_error is not None
    assert band_error == pytest.approx(float(sigma) / 2.0, rel=1e-4)
    assert coverage == pytest.approx(1.0, rel=TOLERANCE)


def test_spectrum_synthetic_photometry_no_error_yields_none():
    wavelength = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    flux = np.ones(4, dtype=np.float64)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=wavelength, flux=flux)
    transmission_grid = _edges_grid([0.0, 1.0, 2.0, 3.0, 4.0], np.float64)
    transmission = np.ones(4, dtype=np.float64)
    _band_flux, band_error, _coverage = spectrum.synthetic_photometry(
        transmission_grid=transmission_grid,
        transmission_values=transmission,
    )
    assert band_error is None


def test_centers_grid_helper_exists():
    # Sanity check helper imports work as expected.
    grid = _centers_grid([0.5, 1.5, 2.5], np.float64)
    assert grid.kind == "centers"

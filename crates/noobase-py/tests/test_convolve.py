"""Tests for the convolve bindings: Spectrum.convolve_lsf,
image.convolve_psf, image.convolve_gaussian_axis."""

import numpy as np
import pytest

import noobase


TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Spectrum.convolve_lsf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec": "constant_r", "resolving_power": 2000.0},
        {
            "spec": "constant_velocity",
            "sigma": 60.0,
            "speed_of_light": 299792.458,
        },
    ],
)
def test_convolve_lsf_constant_template_is_conserved(dtype, kwargs):
    grid = noobase.axis.Grid.logspace(4000.0, 7000.0, 256, dtype=dtype)
    flux = np.full(256, 3.5, dtype=dtype)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=grid, flux=flux)
    out = spectrum.convolve_lsf(**kwargs)
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out.flux, 3.5, rtol=TOLERANCE)
    assert out.error is None
    assert out.mask is None
    # Wavelength axis is unchanged.
    np.testing.assert_array_equal(out.wavelength.values, grid.values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convolve_lsf_general_path_linear_grid(dtype):
    wavelength = np.linspace(4000.0, 7000.0, 200, dtype=dtype)
    flux = np.full(200, -1.5, dtype=dtype)
    spectrum = noobase.spectroscopy.Spectrum(
        wavelength=wavelength, flux=flux, spacing="linear", kind="centers"
    )
    out = spectrum.convolve_lsf(spec="constant_r", resolving_power=3000.0)
    np.testing.assert_allclose(out.flux, -1.5, rtol=TOLERANCE)


def test_convolve_lsf_rejects_error_or_mask():
    grid = noobase.axis.Grid.logspace(4000.0, 7000.0, 32, dtype=np.float64)
    flux = np.ones(32)
    with_error = noobase.spectroscopy.Spectrum(wavelength=grid, flux=flux, error=np.full(32, 0.1))
    with pytest.raises(ValueError, match="noise-free templates"):
        with_error.convolve_lsf(spec="constant_r", resolving_power=2000.0)
    with_mask = noobase.spectroscopy.Spectrum(
        wavelength=grid, flux=flux, mask=np.zeros(32, dtype=bool)
    )
    with pytest.raises(ValueError, match="noise-free templates"):
        with_mask.convolve_lsf(spec="constant_r", resolving_power=2000.0)


def test_convolve_lsf_invalid_resolution():
    grid = noobase.axis.Grid.logspace(4000.0, 7000.0, 16, dtype=np.float64)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=grid, flux=np.ones(16))
    with pytest.raises(ValueError, match="resolution must be positive"):
        spectrum.convolve_lsf(spec="constant_r", resolving_power=0.0)
    with pytest.raises(ValueError, match="resolution must be positive"):
        spectrum.convolve_lsf(
            spec="constant_velocity", sigma=-1.0, speed_of_light=3e5
        )


def test_convolve_lsf_invalid_spec_and_missing_companions():
    grid = noobase.axis.Grid.logspace(4000.0, 7000.0, 16, dtype=np.float64)
    spectrum = noobase.spectroscopy.Spectrum(wavelength=grid, flux=np.ones(16))
    with pytest.raises(ValueError, match="invalid spec"):
        spectrum.convolve_lsf(spec="bogus")
    with pytest.raises(ValueError, match="resolving_power is required"):
        spectrum.convolve_lsf(spec="constant_r")
    with pytest.raises(ValueError, match="sigma is required"):
        spectrum.convolve_lsf(spec="constant_velocity", speed_of_light=3e5)
    with pytest.raises(ValueError, match="speed_of_light is required"):
        spectrum.convolve_lsf(spec="constant_velocity", sigma=50.0)


# ---------------------------------------------------------------------------
# image.convolve_psf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convolve_psf_sum_normalized_preserves_constant(dtype):
    image = np.full((9, 9), 4.2, dtype=dtype)
    image[4, 4] = np.nan
    psf = np.full((3, 3), 1.0 / 9.0, dtype=dtype)
    out = noobase.image.convolve_psf(image, psf)
    assert out.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 4.2, rtol=1e-4)


def test_convolve_psf_true_convolution_orientation():
    image = np.zeros((5, 5), dtype=np.float64)
    image[2, 2] = 1.0
    psf = np.zeros((3, 3), dtype=np.float64)
    psf[0, 1] = 1.0  # mass one row above center
    out = noobase.image.convolve_psf(image, psf)
    # True convolution reproduces the PSF as-is: spike one row above.
    assert out[1, 2] == pytest.approx(1.0)
    assert out[3, 2] == pytest.approx(0.0)


def test_convolve_psf_even_dim_psf_is_rejected():
    image = np.zeros((4, 4), dtype=np.float64)
    psf = np.full((2, 2), 0.25, dtype=np.float64)
    with pytest.raises(ValueError, match="odd dimensions"):
        noobase.image.convolve_psf(image, psf)


def test_convolve_psf_dtype_mismatch_is_rejected():
    image = np.zeros((5, 5), dtype=np.float64)
    psf = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        noobase.image.convolve_psf(image, psf)


# ---------------------------------------------------------------------------
# image.convolve_gaussian_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("normalization", ["sum", "l2", "none"])
@pytest.mark.parametrize("boundary", ["zero", "reflect", "nearest"])
def test_convolve_gaussian_axis_runs_over_matrix(dtype, normalization, boundary):
    image = np.ones((12, 7), dtype=dtype)
    out = noobase.image.convolve_gaussian_axis(
        image,
        sigma=1.5,
        axis=0,
        normalization=normalization,
        boundary=boundary,
        renormalize=False,
    )
    assert out.dtype == np.dtype(dtype)
    assert out.shape == image.shape


def test_convolve_gaussian_axis_renormalize_conserves_constant():
    image = np.full((20, 3), 5.0, dtype=np.float64)
    renormed = noobase.image.convolve_gaussian_axis(
        image, sigma=2.0, axis=0, normalization="sum", renormalize=True
    )
    np.testing.assert_allclose(renormed, 5.0, rtol=1e-9)
    zero_pad = noobase.image.convolve_gaussian_axis(
        image, sigma=2.0, axis=0, normalization="sum", renormalize=False
    )
    assert zero_pad[0, 0] < 5.0 - 1e-3  # edge darkened
    assert zero_pad[10, 0] == pytest.approx(5.0, rel=1e-6)  # interior intact


def test_convolve_gaussian_axis_selects_axis():
    # Image varies only along axis 0; smoothing axis 1 (constant
    # direction) with a sum-normalized renormalized kernel is identity.
    base = np.arange(9, dtype=np.float64) ** 2
    image = np.tile(base[:, None], (1, 6))
    along_constant = noobase.image.convolve_gaussian_axis(
        image, sigma=1.5, axis=1, normalization="sum", renormalize=True
    )
    np.testing.assert_allclose(along_constant, image, rtol=1e-9)
    along_varying = noobase.image.convolve_gaussian_axis(
        image, sigma=1.5, axis=0, normalization="sum", renormalize=True
    )
    assert not np.allclose(along_varying, image, rtol=1e-6)


def test_convolve_gaussian_axis_invalid_arguments():
    image = np.ones((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="sigma must be positive"):
        noobase.image.convolve_gaussian_axis(image, sigma=0.0)
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        noobase.image.convolve_gaussian_axis(image, sigma=1.0, axis=2)
    with pytest.raises(ValueError, match="invalid normalization"):
        noobase.image.convolve_gaussian_axis(
            image, sigma=1.0, normalization="bogus"
        )
    with pytest.raises(ValueError, match="invalid boundary"):
        noobase.image.convolve_gaussian_axis(image, sigma=1.0, boundary="bogus")


def test_convolve_gaussian_axis_matched_filter_peaks_at_injected_sigma():
    rows, cols = 121, 4
    line_row, injected_sigma = 60.0, 5.0
    idx = np.arange(rows, dtype=np.float64)
    profile = np.exp(-0.5 * ((idx - line_row) / injected_sigma) ** 2)
    image = np.tile(profile[:, None], (1, cols))

    def peak(probe):
        response = noobase.image.convolve_gaussian_axis(
            image, sigma=probe, axis=0, normalization="l2", renormalize=False
        )
        return response[60, 0]

    matched = peak(injected_sigma)
    assert matched > peak(injected_sigma * 0.5)
    assert matched > peak(injected_sigma * 2.0)

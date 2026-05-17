"""Tests for the Phase 8 thin PyO3 binding of the point-source PSF
pipeline.

The Rust core math is already pinned by 242 in-crate tests. These tests
exercise *boundary translation* only: dtype dispatch, ``None`` optional
arguments, ``*Error`` -> ``ValueError`` per precondition, ``build_stamp``
three-state, per-star / per-pixel sentinel pass-through, pyclass getter
shapes / dtypes, kwargs defaults == Rust ``Default``, string-enum parse,
and end-to-end chaining against synthetic ground truth.
"""

import numpy as np
import pytest

import noobase
from noobase.image import build_stamp
from noobase.image import psf as psf

# ---------------------------------------------------------------------------
# Synthetic helpers (consistent with decision 12: os odd, s odd,
# K_c = (os*s - 1)/2, c_det = (s - 1)/2; delta = 0 -> the native sample
# is a zero-interpolation pick of the oversampled grid).
# ---------------------------------------------------------------------------


def oversampled_gaussian(os: int, s: int, sigma_native: float) -> np.ndarray:
    """Unit-volume (sum/os^2 == 1) oversampled Gaussian core."""
    side = os * s
    center = (side - 1) / 2.0
    coords = np.arange(side, dtype=np.float64)
    sigma_os = sigma_native * os
    g1 = np.exp(-0.5 * ((coords - center) / sigma_os) ** 2)
    psi = np.outer(g1, g1)
    psi /= psi.sum() / (os * os)
    return psi


def native_from_oversampled(psi: np.ndarray, os: int, s: int) -> np.ndarray:
    """The delta=0 native sample of an oversampled core."""
    start = (os - 1) // 2
    return np.ascontiguousarray(psi[start::os, start::os][:s, :s])


def gaussian_native(side: int, sigma: float, peak: float) -> np.ndarray:
    center = (side - 1) / 2.0
    coords = np.arange(side, dtype=np.float64)
    g1 = np.exp(-0.5 * ((coords - center) / sigma) ** 2)
    return peak * np.outer(g1, g1)


# ---------------------------------------------------------------------------
# build_stamp -- three-state (None / pyclass / ValueError), dtype, Option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_build_stamp_success_pyclass_getters(dtype):
    cutout = gaussian_native(15, 2.0, 100.0).astype(dtype)
    result = build_stamp(cutout, 7)
    assert isinstance(result, noobase.image.StampResult)
    assert result.stamp.shape == (7, 7)
    assert result.stamp.dtype == np.float64  # output always f64
    assert result.valid.shape == (7, 7)
    assert result.valid.dtype == np.bool_
    assert result.error is None  # no error supplied
    delta = result.delta
    assert isinstance(delta, tuple) and len(delta) == 2
    assert -0.5 <= delta[0] < 0.5 and -0.5 <= delta[1] < 0.5
    origin = result.origin
    assert isinstance(origin, tuple) and len(origin) == 2
    assert all(isinstance(v, int) for v in origin)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_build_stamp_with_error_option(dtype):
    cutout = gaussian_native(15, 2.0, 100.0).astype(dtype)
    error = np.ones_like(cutout)
    result = build_stamp(cutout, 7, error=error)
    assert result is not None
    assert result.error is not None
    assert result.error.shape == (7, 7)
    assert result.error.dtype == np.float64


def test_build_stamp_dtype_dispatch_equivalent():
    cutout64 = gaussian_native(15, 2.0, 100.0)
    r64 = build_stamp(cutout64.astype(np.float64), 7)
    r32 = build_stamp(cutout64.astype(np.float32), 7)
    np.testing.assert_allclose(r64.stamp, r32.stamp, rtol=1e-5, atol=1e-4)


def test_build_stamp_none_when_too_few_valid_pixels():
    cutout = np.full((15, 15), np.nan, dtype=np.float64)
    assert build_stamp(cutout, 7) is None


def test_build_stamp_mask_polarity_excludes_true():
    cutout = gaussian_native(15, 2.0, 100.0)
    mask = np.zeros((15, 15), dtype=bool)
    result = build_stamp(cutout, 7, mask=mask)
    assert result is not None
    # A true=invalid pixel must be valid=False in the windowed output.
    mask_all = np.ones((15, 15), dtype=bool)
    assert build_stamp(cutout, 7, mask=mask_all) is None  # all invalid -> skip


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stamp_size": 8},  # even
        {"stamp_size": 21},  # larger than cutout
    ],
)
def test_build_stamp_precondition_valueerror(kwargs):
    cutout = gaussian_native(15, 2.0, 100.0)
    with pytest.raises(ValueError):
        build_stamp(cutout, **kwargs)


def test_build_stamp_error_shape_mismatch_valueerror():
    cutout = gaussian_native(15, 2.0, 100.0)
    with pytest.raises(ValueError):
        build_stamp(cutout, 7, error=np.ones((10, 10)))


def test_build_stamp_mask_wrong_dtype_valueerror():
    cutout = gaussian_native(15, 2.0, 100.0)
    with pytest.raises(ValueError, match="bool"):
        build_stamp(cutout, 7, mask=np.zeros((15, 15), dtype=np.float64))


def test_build_stamp_unsupported_dtype_valueerror():
    with pytest.raises(ValueError, match="float32 or float64"):
        build_stamp(np.zeros((15, 15), dtype=np.int32), 7)


# ---------------------------------------------------------------------------
# robust_combine -- reducer, sentinel, string enum, dtype
# ---------------------------------------------------------------------------


def test_robust_combine_median_and_count_dtype():
    stack = np.zeros((5, 2, 2), dtype=np.float64)
    for n in range(5):
        stack[n, 0, 0] = float(n)  # 0,1,2,3,4 -> median 2
    out = psf.robust_combine(stack, method="median")
    assert isinstance(out, psf.RobustCombined)
    assert out.combined.shape == (2, 2)
    assert out.combined.dtype == np.float64
    assert out.count.dtype == np.uint32
    assert out.combined[0, 0] == pytest.approx(2.0)
    assert out.count[0, 0] == 5


def test_robust_combine_all_nan_pixel_sentinel_not_raised():
    stack = np.full((4, 1, 1), np.nan, dtype=np.float64)
    out = psf.robust_combine(stack, method="median")
    assert np.isnan(out.combined[0, 0])
    assert out.weight[0, 0] == 0.0
    assert out.count[0, 0] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_robust_combine_dtype_dispatch_output_f64(dtype):
    stack = np.ones((3, 2, 2), dtype=dtype)
    out = psf.robust_combine(stack, method="clipped_mean")
    assert out.combined.dtype == np.float64
    np.testing.assert_allclose(out.combined, 1.0)


def test_robust_combine_weight_option_and_shape_mismatch():
    stack = np.ones((3, 2, 2), dtype=np.float64)
    out = psf.robust_combine(stack, weight=np.ones((3, 2, 2)), method="median")
    assert out.combined.shape == (2, 2)
    with pytest.raises(ValueError):
        psf.robust_combine(stack, weight=np.ones((3, 2, 3)))


def test_robust_combine_invalid_method_and_params():
    stack = np.ones((3, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="invalid combine"):
        psf.robust_combine(stack, method="bogus")
    with pytest.raises(ValueError):
        psf.robust_combine(stack, method="clipped_mean", combine_kappa=-1.0)


# ---------------------------------------------------------------------------
# solve_flux_background -- exact (f,b) recovery, sentinel, dtype, errors
# ---------------------------------------------------------------------------

_OS = 3
_S = 7


def _epsf_and_native():
    psi = oversampled_gaussian(_OS, _S, 1.3)
    native = native_from_oversampled(psi, _OS, _S)
    return psi, native


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solve_flux_background_recovers_known_fb(dtype):
    psi, native = _epsf_and_native()
    flux_true = np.array([10.0, 25.0, 4.0])
    bg_true = np.array([1.0, -0.5, 2.0])
    data = np.empty((3, _S, _S), dtype=dtype)
    for n in range(3):
        data[n] = (flux_true[n] * native + bg_true[n]).astype(dtype)
    delta = np.zeros((3, 2), dtype=np.float64)
    out = psf.solve_flux_background(psi, _OS, data, delta)
    assert isinstance(out, psf.FluxBackground)
    assert out.flux.dtype == np.float64
    assert out.ok.dtype == np.bool_
    assert out.ok.all()
    np.testing.assert_allclose(out.flux, flux_true, rtol=1e-4)
    np.testing.assert_allclose(out.background, bg_true, atol=1e-3)


def test_solve_flux_background_all_invalid_star_sentinel():
    psi, native = _epsf_and_native()
    data = np.stack([native, np.full((_S, _S), np.nan)]).astype(np.float64)
    delta = np.zeros((2, 2), dtype=np.float64)
    out = psf.solve_flux_background(psi, _OS, data, delta)
    assert out.ok[0]
    assert not out.ok[1]
    assert np.isnan(out.flux[1]) and np.isnan(out.background[1])


def test_solve_flux_background_preconditions():
    psi, native = _epsf_and_native()
    data = native[None, :, :].astype(np.float64)
    delta = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError):  # epsf not square
        psf.solve_flux_background(psi[:, :-1], _OS, data, delta)
    with pytest.raises(ValueError):  # oversample even
        psf.solve_flux_background(psi, 2, data, delta)
    with pytest.raises(ValueError):  # delta not (N,2)
        psf.solve_flux_background(psi, _OS, data, np.zeros((1, 3)))
    with pytest.raises(ValueError):  # weight shape mismatch
        psf.solve_flux_background(
            psi, _OS, data, delta, weight=np.ones((1, _S, _S + 1))
        )
    with pytest.raises(ValueError, match="float64"):  # epsf must be f64
        psf.solve_flux_background(
            psi.astype(np.float32), _OS, data, delta
        )


# ---------------------------------------------------------------------------
# build_epsf -- warm-start recovery, gauge, sentinel, defaults, errors
# ---------------------------------------------------------------------------


def _synth_stack(dtype=np.float64):
    psi, native = _epsf_and_native()
    flux_true = np.array([20.0, 35.0, 12.0, 28.0])
    bg_true = np.array([0.5, -0.3, 1.0, 0.0])
    data = np.empty((4, _S, _S), dtype=dtype)
    for n in range(4):
        data[n] = (flux_true[n] * native + bg_true[n]).astype(dtype)
    delta_init = np.zeros((4, 2), dtype=np.float64)
    return psi, data, delta_init, flux_true


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_build_epsf_warm_start_recovers_truth(dtype):
    psi, data, delta_init, flux_true = _synth_stack(dtype)
    out = psf.build_epsf(
        data, delta_init, _OS, psi_init=psi, max_iter=8, tol=1e-9
    )
    assert isinstance(out, psf.BuildEpsf)
    assert out.epsf.shape == (_OS * _S, _OS * _S)
    assert out.epsf.dtype == np.float64
    assert out.ok.all()
    # Unit-volume gauge (fork 4).
    assert out.epsf.sum() / (_OS * _OS) == pytest.approx(1.0, abs=1e-9)
    assert out.flux.dtype == np.float64
    assert out.delta.shape == (4, 2)
    assert isinstance(out.iterations, int)
    assert isinstance(out.converged, bool)
    np.testing.assert_allclose(out.flux, flux_true, rtol=2e-2)


def test_build_epsf_seed_path_runs():
    _, data, delta_init, _ = _synth_stack()
    out = psf.build_epsf(data, delta_init, _OS, max_iter=5)
    assert np.isfinite(out.epsf).all()
    assert out.epsf.sum() / (_OS * _OS) == pytest.approx(1.0, abs=1e-9)


def test_build_epsf_per_star_sentinel():
    psi, data, delta_init, _ = _synth_stack()
    data = data.copy()
    data[2] = np.nan
    out = psf.build_epsf(data, delta_init, _OS, psi_init=psi, max_iter=5)
    assert not out.ok[2]
    assert np.isnan(out.flux[2])
    assert out.ok[0] and out.ok[1] and out.ok[3]


def test_build_epsf_defaults_match_rust_default():
    psi, data, delta_init, _ = _synth_stack()
    a = psf.build_epsf(data, delta_init, _OS, psi_init=psi)
    b = psf.build_epsf(
        data,
        delta_init,
        _OS,
        psi_init=psi,
        max_iter=50,
        tol=1e-4,
        step=1.0,
        residual_reweight="none",
        reweight_c=4.0,
        nuisance_max_iter=3,
        nuisance_tol=1e-4,
    )
    np.testing.assert_array_equal(a.epsf, b.epsf)
    np.testing.assert_array_equal(a.flux, b.flux)


@pytest.mark.parametrize("reweight", ["none", "huber", "tukey"])
def test_build_epsf_residual_reweight_parse(reweight):
    psi, data, delta_init, _ = _synth_stack()
    out = psf.build_epsf(
        data, delta_init, _OS, psi_init=psi, max_iter=3,
        residual_reweight=reweight, reweight_c=4.0,
    )
    assert out.epsf.shape == (_OS * _S, _OS * _S)


def test_build_epsf_preconditions():
    psi, data, delta_init, _ = _synth_stack()
    with pytest.raises(ValueError):  # oversample even
        psf.build_epsf(data, delta_init, 2)
    with pytest.raises(ValueError):  # max_iter == 0
        psf.build_epsf(data, delta_init, _OS, max_iter=0)
    with pytest.raises(ValueError, match="invalid residual_reweight"):
        psf.build_epsf(data, delta_init, _OS, residual_reweight="bogus")
    with pytest.raises(ValueError):  # psi_init shape mismatch
        psf.build_epsf(data, delta_init, _OS, psi_init=np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# stitch_psf -- hybrid output, EE-normalization, pure-core fallback, f64-only
# ---------------------------------------------------------------------------

_STITCH_OS = 3
_STITCH_S = 25  # core_native_half = 12
_WING_SIDE = 51  # wing_half = 25 (>= the Rust default annulus r_out = 24)


def _core_and_wing():
    core = oversampled_gaussian(_STITCH_OS, _STITCH_S, 2.0)
    wing = gaussian_native(_WING_SIDE, 5.0, 1.0)
    return core, wing


def test_stitch_psf_hybrid_shapes_and_meta():
    core, wing = _core_and_wing()
    out = psf.stitch_psf(
        core, _STITCH_OS, wing,
        match_radius=6.0, feather_width=2.0, ee_aperture_radius=12.0,
    )
    assert isinstance(out, psf.ExtendedPsf)
    assert out.core.shape == (_STITCH_OS * _STITCH_S, _STITCH_OS * _STITCH_S)
    assert out.core.dtype == np.float64
    assert out.wing.shape == (_WING_SIDE, _WING_SIDE)
    assert out.oversample == _STITCH_OS
    assert out.match_radius == pytest.approx(6.0)
    assert out.feather_width == pytest.approx(2.0)
    assert out.ee_aperture_radius == pytest.approx(12.0)


def test_stitch_psf_pure_core_when_wing_all_nan():
    core, _ = _core_and_wing()
    wing = np.full((_WING_SIDE, _WING_SIDE), np.nan, dtype=np.float64)
    out = psf.stitch_psf(
        core, _STITCH_OS, wing,
        match_radius=6.0, feather_width=2.0, ee_aperture_radius=12.0,
    )
    # Degenerate wing -> pure-core EE-normalized result, no exception.
    assert np.allclose(np.nan_to_num(out.wing), 0.0)
    assert np.isfinite(out.core).all()


def test_stitch_psf_defaults_match_rust_default():
    core, wing = _core_and_wing()
    a = psf.stitch_psf(core, _STITCH_OS, wing)
    b = psf.stitch_psf(
        core, _STITCH_OS, wing,
        match_radius=8.0, feather_width=4.0, ee_aperture_radius=15.0,
    )
    np.testing.assert_array_equal(a.core, b.core)
    np.testing.assert_array_equal(a.wing, b.wing)


def test_stitch_psf_f64_only_and_preconditions():
    core, wing = _core_and_wing()
    with pytest.raises(ValueError, match="float64"):
        psf.stitch_psf(core.astype(np.float32), _STITCH_OS, wing)
    with pytest.raises(ValueError):  # core not square
        psf.stitch_psf(core[:, :-3], _STITCH_OS, wing)
    with pytest.raises(ValueError):  # oversample even
        psf.stitch_psf(core, 2, wing)
    with pytest.raises(ValueError):  # wing even dims
        psf.stitch_psf(core, _STITCH_OS, np.ones((40, 40)))
    with pytest.raises(ValueError):  # confidence shape mismatch
        psf.stitch_psf(
            core, _STITCH_OS, wing, wing_confidence=np.ones((10, 10))
        )
    with pytest.raises(ValueError):  # infeasible params
        psf.stitch_psf(
            core, _STITCH_OS, wing, match_radius=500.0, feather_width=2.0
        )


# ---------------------------------------------------------------------------
# build_extended_psf -- orchestrator, sentinel, dtype, defaults, errors
# ---------------------------------------------------------------------------


def _bright_stack(dtype=np.float64):
    core, _ = _core_and_wing()
    wing_data = np.empty((4, _WING_SIDE, _WING_SIDE), dtype=dtype)
    for n in range(4):
        wing_data[n] = gaussian_native(
            _WING_SIDE, 5.0, 50.0 + 10.0 * n
        ).astype(dtype)
    wing_delta = np.zeros((4, 2), dtype=np.float64)
    return core, wing_data, wing_delta


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_build_extended_psf_runs_and_shapes(dtype):
    core, wing_data, wing_delta = _bright_stack(dtype)
    out = psf.build_extended_psf(
        wing_data, wing_delta, core, _STITCH_OS,
        match_radius=6.0, feather_width=2.0, ee_aperture_radius=12.0,
        scale_aperture_radius=5.0, scale_background_annulus=(14.0, 18.0),
    )
    assert isinstance(out, psf.ExtendedPsfBuilt)
    assert isinstance(out.extended, psf.ExtendedPsf)
    assert out.extended.core.shape == (
        _STITCH_OS * _STITCH_S,
        _STITCH_OS * _STITCH_S,
    )
    assert out.star_flux.shape == (4,)
    assert out.star_flux.dtype == np.float64
    assert out.star_ok.dtype == np.bool_
    assert out.star_scale_from_core.dtype == np.bool_
    assert out.star_ok.all()


def test_build_extended_psf_uncalibratable_star_sentinel():
    core, wing_data, wing_delta = _bright_stack()
    wing_weight = np.ones_like(wing_data)
    wing_weight[2] = 0.0  # star 2 fully masked -> uncalibratable
    out = psf.build_extended_psf(
        wing_data, wing_delta, core, _STITCH_OS, wing_weight=wing_weight,
        match_radius=6.0, feather_width=2.0, ee_aperture_radius=12.0,
        scale_aperture_radius=5.0, scale_background_annulus=(14.0, 18.0),
    )
    assert not out.star_ok[2]
    assert np.isnan(out.star_flux[2])


def test_build_extended_psf_defaults_match_rust_default():
    core, wing_data, wing_delta = _bright_stack()
    a = psf.build_extended_psf(wing_data, wing_delta, core, _STITCH_OS)
    b = psf.build_extended_psf(
        wing_data, wing_delta, core, _STITCH_OS,
        match_radius=8.0, feather_width=4.0, ee_aperture_radius=15.0,
        combine="clipped_mean", combine_kappa=3.0, combine_max_iter=5,
        scale_aperture_radius=6.0, scale_background_annulus=(18.0, 24.0),
    )
    np.testing.assert_array_equal(a.extended.wing, b.extended.wing)
    np.testing.assert_array_equal(a.star_flux, b.star_flux)


def test_build_extended_psf_preconditions():
    core, wing_data, wing_delta = _bright_stack()
    with pytest.raises(ValueError):  # core not square
        psf.build_extended_psf(wing_data, wing_delta, core[:, :-3], _STITCH_OS)
    with pytest.raises(ValueError):  # oversample even
        psf.build_extended_psf(wing_data, wing_delta, core, 2)
    with pytest.raises(ValueError):  # wing_data even spatial dims
        psf.build_extended_psf(
            np.ones((4, 40, 40)), wing_delta, core, _STITCH_OS
        )
    with pytest.raises(ValueError):  # wing_delta not (M,2)
        psf.build_extended_psf(
            wing_data, np.zeros((4, 3)), core, _STITCH_OS
        )
    with pytest.raises(ValueError):  # wing_weight shape mismatch
        psf.build_extended_psf(
            wing_data, wing_delta, core, _STITCH_OS,
            wing_weight=np.ones((4, _WING_SIDE, _WING_SIDE - 1)),
        )
    with pytest.raises(ValueError, match="invalid combine"):
        psf.build_extended_psf(
            wing_data, wing_delta, core, _STITCH_OS, combine="bogus"
        )


# ---------------------------------------------------------------------------
# End-to-end: build_stamp x N -> stack -> build_epsf (boundary plumbing)
# ---------------------------------------------------------------------------


def test_end_to_end_build_stamp_to_build_epsf():
    psi, native = _epsf_and_native()
    flux_true = np.array([18.0, 30.0, 9.0])
    stamps = []
    deltas = []
    for n in range(3):
        cutout = np.zeros((13, 13), dtype=np.float64)
        # Embed the native PSF in a larger rough cutout, centered.
        off = (13 - _S) // 2
        cutout[off : off + _S, off : off + _S] = flux_true[n] * native + 0.2
        res = build_stamp(cutout, _S)
        assert res is not None
        stamps.append(res.stamp)
        deltas.append(res.delta)
    stack = np.stack(stamps).astype(np.float64)
    delta_init = np.array(deltas, dtype=np.float64)
    out = psf.build_epsf(stack, delta_init, _OS, psi_init=psi, max_iter=8)
    assert out.epsf.shape == (_OS * _S, _OS * _S)
    assert out.ok.all()
    assert out.epsf.sum() / (_OS * _OS) == pytest.approx(1.0, abs=1e-9)

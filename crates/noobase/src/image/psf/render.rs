//! Forward PSF model: oversampled effective PSF -> predicted detector
//! stamps.
//!
//! Given an oversampled effective-PSF model `epsf` and, for each stamp in
//! a batch, a sub-pixel offset, a flux and a background, this renders the
//! predicted detector-resolution stamp by *point sampling* the model with
//! a separable bicubic Catmull-Rom kernel. The pixel response (Pi) is
//! considered already absorbed into the empirical effective PSF, so the
//! sampler is a point evaluation, not a box/area integration.
//!
//! Design notes:
//!
//! - **Model grid / oversample convention.** `epsf` is a square
//!   `(os*s, os*s)` array where `os = oversample` is forced odd and
//!   `s = stamp_size` is the (odd) detector stamp edge. `s` is *derived*
//!   from `epsf.shape()` and `os`, never passed separately, so the two
//!   can never disagree. The model is evaluated through the interpolation
//!   kernel (never read by integer index), so it carries no load-bearing
//!   parity of its own, but `os*s` is kept odd so the global center cell
//!   `K_c = (os*s - 1) / 2` lands exactly on the PSF center and the
//!   detector center `c_det = (s - 1) / 2` nests symmetrically inside it.
//!
//! - **Sampling.** The source center of stamp `n` sits at detector
//!   coordinate `c_det + delta_n`. Detector pixel `(i, j)` therefore maps
//!   to model coordinates
//!   `k_u = K_c + os * ((i - c_det) - delta_row)`,
//!   `k_v = K_c + os * ((j - c_det) - delta_col)`,
//!   and the prediction is
//!   `pred = flux * CatmullRom(epsf; k_u, k_v) + background`.
//!   The `delta` sign is fixed by this formula: it is exactly the offset
//!   reported by `build_stamp` (`centroid - round(centroid)`), so feeding
//!   a `build_stamp` delta straight back in reproduces the source at its
//!   true sub-pixel location (pinned by a round-trip test).
//!
//! - **Odd `os` is an intentional, scoped convention.** With `s` odd and
//!   `os` odd, `os*s` is odd, so a `delta = 0` source lands on integer
//!   model coordinates (`k_u = K_c + os*(i - c_det)`), the Catmull-Rom
//!   weights collapse to `[0, 1, 0, 0]`, and the render reduces to a
//!   zero-interpolation grid extraction. An even `os` would force the PSF
//!   center onto a cell boundary and interpolate even at `delta = 0`;
//!   that is rejected up front (`OversampleNotOdd`). Tools such as
//!   photutils do not impose this; the restriction is deliberate here for
//!   the exact-nesting property above.
//!
//! - **Interpolation kernel.** Separable bicubic Catmull-Rom (Keys cubic
//!   convolution with `a = -1/2`), four taps per axis. It is a pure-Rust
//!   module-private helper that never crosses the FFI boundary and is
//!   never visible to the caller. The later adjoint phase must be the
//!   *exact* transpose of this sampler (the scatter of the very same
//!   per-tap weights), which a dedicated test will pin.
//!
//! - **Zero padding.** Model taps that fall outside `[0, os*s)` on either
//!   axis contribute zero. A source pushed entirely off the model grid
//!   therefore renders to a flat `background` stamp.
//!
//! - **Errors are hard preconditions only.** Every [`RenderError`] is a
//!   shape/parity precondition; there is no algorithmic `Ok(None)` path
//!   (contrast `build_stamp`). The model is `f64` end to end (decision:
//!   the internal effective-PSF model is always `f64`), so `render` is
//!   not generic over the float type.

use ndarray::{Array3, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;
use thiserror::Error;

/// Errors returned by [`render`] for ill-shaped / ill-parameterized
/// inputs.
///
/// All variants are hard preconditions. Unlike `build_stamp`, `render`
/// has no algorithmic skip path: a well-formed call always yields an
/// `(N, s, s)` array.
#[derive(Debug, Error, PartialEq)]
pub enum RenderError {
    #[error("epsf must be square; got ({rows}, {cols})")]
    EpsfNotSquare { rows: usize, cols: usize },
    #[error("oversample must be odd; got {oversample}")]
    OversampleNotOdd { oversample: usize },
    #[error(
        "epsf side ({epsf_side}) must be an integer multiple of oversample ({oversample})"
    )]
    EpsfSizeNotMultiple { epsf_side: usize, oversample: usize },
    #[error(
        "derived stamp_size (epsf_side / oversample) must be odd; got {stamp_size}"
    )]
    DerivedStampSizeEven { stamp_size: usize },
    #[error(
        "batch dimensions disagree: delta shape {delta:?} must be (N, 2) with N == flux len ({flux}) == background len ({background})"
    )]
    BatchLengthMismatch {
        delta: (usize, usize),
        flux: usize,
        background: usize,
    },
}

/// Render a batch of predicted detector stamps from an oversampled
/// effective-PSF model.
///
/// See the module-level documentation for the model-grid / oversample
/// convention, the sampling formula and `delta` sign, the odd-`os`
/// rationale, and the zero-padding rule.
///
/// # Parameters
///
/// - `epsf`: oversampled effective PSF, square `(os*s, os*s)`. Always
///   `f64` (the internal model is `f64` end to end).
/// - `oversample`: `os`, forced odd.
/// - `delta`: `(N, 2)` per-stamp sub-pixel offset `(row, col)`, as
///   produced by `build_stamp` (`centroid - round(centroid)`).
/// - `flux`: `(N,)` per-stamp scale `f_i`.
/// - `background`: `(N,)` per-stamp constant offset `b_i`.
///
/// # Returns
///
/// `Ok(Array3<f64>)` of shape `(N, s, s)`, where `s = (os*s) / os` is
/// derived from `epsf.shape()` and `oversample`. Stamp `n` is
/// `flux[n] * CatmullRom(epsf; ...) + background[n]`.
///
/// # Errors
///
/// Returns a [`RenderError`] if `epsf` is not square, if `oversample` is
/// not odd, if the `epsf` side is not an integer multiple of
/// `oversample`, if the derived `stamp_size` is even, or if `delta`,
/// `flux` and `background` do not share a consistent batch length
/// (`delta` must be `(N, 2)`).
pub fn render(
    epsf: ArrayView2<f64>,
    oversample: usize,
    delta: ArrayView2<f64>,
    flux: ArrayView1<f64>,
    background: ArrayView1<f64>,
) -> Result<Array3<f64>, RenderError> {
    let epsf_rows = epsf.shape()[0];
    let epsf_cols = epsf.shape()[1];

    // --- Hard preconditions (returned as Err; no Ok(None) path). ---
    if epsf_rows != epsf_cols {
        return Err(RenderError::EpsfNotSquare {
            rows: epsf_rows,
            cols: epsf_cols,
        });
    }
    if oversample % 2 == 0 {
        // Also rejects oversample == 0 (0 is even, hence not odd).
        return Err(RenderError::OversampleNotOdd { oversample });
    }
    let epsf_side = epsf_rows;
    if epsf_side % oversample != 0 {
        return Err(RenderError::EpsfSizeNotMultiple {
            epsf_side,
            oversample,
        });
    }
    let stamp_size = epsf_side / oversample;
    if stamp_size % 2 == 0 {
        // Catches the degenerate empty model (stamp_size == 0) too.
        return Err(RenderError::DerivedStampSizeEven { stamp_size });
    }
    let batch_size = flux.len();
    if delta.shape()[1] != 2
        || delta.shape()[0] != batch_size
        || background.len() != batch_size
    {
        return Err(RenderError::BatchLengthMismatch {
            delta: (delta.shape()[0], delta.shape()[1]),
            flux: batch_size,
            background: background.len(),
        });
    }

    // `epsf_side` and `stamp_size` are both odd, so these centers are
    // exact integers in f64 (no rounding): the delta = 0 source lands on
    // an integer model coordinate and the kernel collapses to a pick.
    let psf_center = (epsf_side as f64 - 1.0) / 2.0;
    let detector_center = (stamp_size as f64 - 1.0) / 2.0;
    let oversample_f = oversample as f64;

    let mut output = Array3::<f64>::zeros((batch_size, stamp_size, stamp_size));

    // Stamps are independent: each writes only its own `(s, s)` slice and
    // reads only the shared immutable model and per-stamp scalars. Drive
    // the batch in parallel via rayon (the iterative solver's hot path).
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(stamp_index, stamp_out)| {
            render_stamp(
                stamp_out,
                epsf,
                delta[(stamp_index, 0)],
                delta[(stamp_index, 1)],
                flux[stamp_index],
                background[stamp_index],
                oversample_f,
                stamp_size,
                psf_center,
                detector_center,
            );
        });

    Ok(output)
}

/// Render one detector stamp in place. Factored out of the driver so the
/// parallel and any future sequential driver share the exact same
/// per-stamp work unit.
#[allow(clippy::too_many_arguments)]
fn render_stamp(
    mut stamp_out: ArrayViewMut2<f64>,
    epsf: ArrayView2<f64>,
    delta_row: f64,
    delta_column: f64,
    flux: f64,
    background: f64,
    oversample: f64,
    stamp_size: usize,
    psf_center: f64,
    detector_center: f64,
) {
    for i in 0..stamp_size {
        let k_u = psf_center + oversample * ((i as f64 - detector_center) - delta_row);
        for j in 0..stamp_size {
            let k_v =
                psf_center + oversample * ((j as f64 - detector_center) - delta_column);
            stamp_out[(i, j)] = flux * catmull_rom_sample(&epsf, k_u, k_v) + background;
        }
    }
}

/// Separable bicubic Catmull-Rom point sample of `epsf` at the continuous
/// model coordinate `(k_u, k_v)`. Taps that fall outside the model grid
/// are zero (zero padding).
fn catmull_rom_sample(epsf: &ArrayView2<f64>, k_u: f64, k_v: f64) -> f64 {
    let side = epsf.shape()[0] as i64; // square: shape()[0] == shape()[1]

    let u_floor = k_u.floor();
    let v_floor = k_v.floor();
    let weights_u = catmull_rom_weights(k_u - u_floor);
    let weights_v = catmull_rom_weights(k_v - v_floor);

    // The four taps of a Catmull-Rom segment sit at integer offsets
    // -1, 0, +1, +2 from floor(k); `weights_*[t]` is the weight of tap
    // `base + t`. Phase 3's adjoint must scatter these very same weights.
    let base_u = u_floor as i64 - 1;
    let base_v = v_floor as i64 - 1;

    let mut value = 0.0_f64;
    for tap_u in 0..4 {
        let row = base_u + tap_u as i64;
        if row < 0 || row >= side {
            continue; // zero-pad outside the model grid
        }
        let weight_u = weights_u[tap_u];
        for tap_v in 0..4 {
            let column = base_v + tap_v as i64;
            if column < 0 || column >= side {
                continue; // zero-pad outside the model grid
            }
            value += epsf[(row as usize, column as usize)] * weight_u * weights_v[tap_v];
        }
    }
    value
}

/// Keys cubic-convolution (Catmull-Rom, `a = -1/2`) weights for the four
/// taps surrounding a fractional position `frac` in `[0, 1)`. The
/// returned array is ordered `[w(-1), w(0), w(+1), w(+2)]` relative to
/// `floor(coordinate)`.
///
/// The weights sum to one (partition of unity) and at `frac = 0` collapse
/// to `[0, 1, 0, 0]`, which is what makes an integer-aligned sample an
/// exact, interpolation-free grid pick.
#[inline]
fn catmull_rom_weights(frac: f64) -> [f64; 4] {
    // Catmull-Rom basis on the unit interval, parameter t = frac:
    //   w(-1) = -0.5 t^3 +      t^2 - 0.5 t
    //   w( 0) =  1.5 t^3 -  2.5 t^2       + 1
    //   w(+1) = -1.5 t^3 +  2.0 t^2 + 0.5 t
    //   w(+2) =  0.5 t^3 -  0.5 t^2
    let t = frac;
    [
        ((-0.5 * t + 1.0) * t - 0.5) * t,
        ((1.5 * t - 2.5) * t) * t + 1.0,
        ((-1.5 * t + 2.0) * t + 0.5) * t,
        ((0.5 * t - 0.5) * t) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, arr2};

    const TOL: f64 = 1e-9;

    /// Oversampled Gaussian effective PSF of side `epsf_side`, centered on
    /// the global center cell `K_c = (epsf_side - 1) / 2`, with standard
    /// deviation `sigma_model` in model-grid pixels and peak `amplitude`.
    fn gaussian_epsf(epsf_side: usize, sigma_model: f64, amplitude: f64) -> Array2<f64> {
        let center = (epsf_side as f64 - 1.0) / 2.0;
        let mut model = Array2::<f64>::zeros((epsf_side, epsf_side));
        for p in 0..epsf_side {
            for q in 0..epsf_side {
                let dp = p as f64 - center;
                let dq = q as f64 - center;
                model[(p, q)] =
                    amplitude * (-0.5 * (dp * dp + dq * dq) / (sigma_model * sigma_model)).exp();
            }
        }
        model
    }

    /// A deterministic, non-separable, non-symmetric model so an
    /// off-by-row/col or transposed indexing bug cannot hide.
    fn ramp_epsf(epsf_side: usize) -> Array2<f64> {
        let mut model = Array2::<f64>::zeros((epsf_side, epsf_side));
        for p in 0..epsf_side {
            for q in 0..epsf_side {
                model[(p, q)] = 1.0 + p as f64 + 3.0 * q as f64 + 0.5 * (p * q) as f64;
            }
        }
        model
    }

    #[test]
    fn catmull_rom_weights_partition_of_unity_and_integer_pick() {
        // At frac = 0 the kernel must collapse to a pure pick.
        assert_eq!(catmull_rom_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
        // Weights sum to one for arbitrary fractions.
        for &frac in &[0.1, 0.25, 0.5, 0.7, 0.9999] {
            let weights = catmull_rom_weights(frac);
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "sum = {sum} at frac = {frac}");
        }
    }

    #[test]
    fn delta_zero_is_exact_grid_extraction() {
        // os = 5, s = 7 -> epsf_side = 35, K_c = 17, c_det = 3.
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = ramp_epsf(epsf_side);

        let delta = arr2(&[[0.0, 0.0]]);
        let flux = Array1::from_elem(1, 1.0);
        let background = Array1::from_elem(1, 0.0);

        let out = render(
            epsf.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, stamp_size, stamp_size]);

        let psf_center = (epsf_side - 1) / 2; // 17
        let detector_center = (stamp_size - 1) / 2; // 3
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                // delta = 0 -> integer model coordinate -> exact pick.
                let p = psf_center + oversample * i - oversample * detector_center;
                let q = psf_center + oversample * j - oversample * detector_center;
                assert!(
                    (out[(0, i, j)] - epsf[(p, q)]).abs() < 1e-12,
                    "({i},{j}) rendered {} != model {}",
                    out[(0, i, j)],
                    epsf[(p, q)]
                );
            }
        }
    }

    #[test]
    fn subpixel_shift_matches_analytic_gaussian() {
        // sigma_model = os * sigma_det so the rendered stamp equals an
        // analytic Gaussian with detector-pixel sigma_det.
        let oversample = 5;
        let stamp_size = 9;
        let epsf_side = oversample * stamp_size;
        let sigma_det = 1.5;
        let amplitude = 50.0;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * sigma_det, amplitude);

        let delta_row = 0.3;
        let delta_col = -0.2;
        let flux = 3.0;
        let back = 2.0;
        let delta = arr2(&[[delta_row, delta_col]]);
        let out = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, flux).view(),
            Array1::from_elem(1, back).view(),
        )
        .unwrap();

        let detector_center = (stamp_size as f64 - 1.0) / 2.0;
        let mut max_abs_error = 0.0_f64;
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                let du = (i as f64 - detector_center) - delta_row;
                let dv = (j as f64 - detector_center) - delta_col;
                let expected = flux
                    * amplitude
                    * (-0.5 * (du * du + dv * dv) / (sigma_det * sigma_det)).exp()
                    + back;
                max_abs_error = max_abs_error.max((out[(0, i, j)] - expected).abs());
            }
        }
        // Catmull-Rom on a well-sampled (sigma_model = 7.5 grid px)
        // Gaussian is far tighter than this; the loose bound still
        // rejects any off-by-os / sign / center bug (those are gross).
        assert!(
            max_abs_error < 5e-3 * flux * amplitude,
            "max abs error = {max_abs_error}"
        );
    }

    #[test]
    fn flux_and_background_are_linear() {
        let oversample = 3;
        let stamp_size = 5;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.2, 10.0);
        let delta = arr2(&[[0.27, -0.13]]);

        let unit = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap();
        let flux = 7.5;
        let back = -3.25;
        let scaled = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, flux).view(),
            Array1::from_elem(1, back).view(),
        )
        .unwrap();

        for i in 0..stamp_size {
            for j in 0..stamp_size {
                let expected = flux * unit[(0, i, j)] + back;
                assert!(
                    (scaled[(0, i, j)] - expected).abs() < 1e-12,
                    "({i},{j}) {} != {}",
                    scaled[(0, i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn source_pushed_off_grid_renders_flat_background() {
        // A huge delta moves every tap outside the model -> zero-padded
        // sample -> flat `background` stamp (also exercises zero-pad).
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.5, 100.0);
        let delta = arr2(&[[1000.0, -1000.0]]);
        let back = 4.0;
        let out = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, 9.0).view(),
            Array1::from_elem(1, back).view(),
        )
        .unwrap();
        for value in out.iter() {
            assert!((value - back).abs() < TOL, "value = {value}");
        }
    }

    #[test]
    fn uniform_model_interior_is_partition_of_unity() {
        // For a uniform model every fully-in-bounds 4x4 stencil sums the
        // separable weights to exactly 1, so interior pixels render to
        // the uniform value regardless of the (fractional) delta. Edge
        // pixels lose clipped taps; assert only the interior, and that
        // nothing is NaN anywhere.
        let oversample = 5;
        let stamp_size = 9;
        let epsf_side = oversample * stamp_size;
        let epsf = Array2::<f64>::from_elem((epsf_side, epsf_side), 1.0);
        let delta = arr2(&[[0.3, 0.3]]);
        let out = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap();

        for value in out.iter() {
            assert!(value.is_finite(), "non-finite render value {value}");
        }
        // Interior: drop a 1-pixel rim so every contributing model tap is
        // safely in bounds for this small delta.
        for i in 1..stamp_size - 1 {
            for j in 1..stamp_size - 1 {
                assert!(
                    (out[(0, i, j)] - 1.0).abs() < 1e-12,
                    "interior ({i},{j}) = {}",
                    out[(0, i, j)]
                );
            }
        }
    }

    #[test]
    fn batch_stamps_are_independent() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.4, 80.0);

        let delta = arr2(&[[0.10, -0.20], [-0.35, 0.05], [0.40, 0.40]]);
        let flux = Array1::from(vec![1.0, 2.5, 0.7]);
        let background = Array1::from(vec![0.0, -1.0, 3.0]);

        let batch = render(
            epsf.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();
        assert_eq!(batch.shape(), &[3, stamp_size, stamp_size]);

        for n in 0..3 {
            let single = render(
                epsf.view(),
                oversample,
                delta.slice(ndarray::s![n..n + 1, ..]),
                flux.slice(ndarray::s![n..n + 1]),
                background.slice(ndarray::s![n..n + 1]),
            )
            .unwrap();
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    assert!(
                        (batch[(n, i, j)] - single[(0, i, j)]).abs() < 1e-12,
                        "stamp {n} ({i},{j}) differs from single render"
                    );
                }
            }
        }
    }

    #[test]
    fn error_epsf_not_square() {
        let epsf = Array2::<f64>::zeros((10, 15));
        let err = render(
            epsf.view(),
            5,
            arr2(&[[0.0, 0.0]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(err, RenderError::EpsfNotSquare { rows: 10, cols: 15 });
    }

    #[test]
    fn error_oversample_not_odd() {
        let epsf = Array2::<f64>::zeros((20, 20));
        let err = render(
            epsf.view(),
            4,
            arr2(&[[0.0, 0.0]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(err, RenderError::OversampleNotOdd { oversample: 4 });
    }

    #[test]
    fn error_epsf_size_not_multiple() {
        // 34 is not a multiple of 5.
        let epsf = Array2::<f64>::zeros((34, 34));
        let err = render(
            epsf.view(),
            5,
            arr2(&[[0.0, 0.0]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            RenderError::EpsfSizeNotMultiple {
                epsf_side: 34,
                oversample: 5,
            }
        );
    }

    #[test]
    fn error_derived_stamp_size_even() {
        // os = 3, epsf_side = 18 -> stamp_size = 6 (even).
        let epsf = Array2::<f64>::zeros((18, 18));
        let err = render(
            epsf.view(),
            3,
            arr2(&[[0.0, 0.0]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(err, RenderError::DerivedStampSizeEven { stamp_size: 6 });
    }

    #[test]
    fn error_batch_length_mismatch_row_count() {
        let epsf = Array2::<f64>::zeros((35, 35));
        // delta has 2 rows, flux/background have 1.
        let err = render(
            epsf.view(),
            5,
            arr2(&[[0.0, 0.0], [0.1, 0.1]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            RenderError::BatchLengthMismatch {
                delta: (2, 2),
                flux: 1,
                background: 1,
            }
        );
    }

    #[test]
    fn error_batch_length_mismatch_delta_not_two_columns() {
        let epsf = Array2::<f64>::zeros((35, 35));
        // delta is (1, 3): not (N, 2).
        let err = render(
            epsf.view(),
            5,
            arr2(&[[0.0, 0.0, 0.0]]).view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            RenderError::BatchLengthMismatch {
                delta: (1, 3),
                flux: 1,
                background: 1,
            }
        );
    }

    #[test]
    fn empty_batch_yields_empty_output() {
        let epsf = Array2::<f64>::zeros((35, 35));
        let out = render(
            epsf.view(),
            5,
            Array2::<f64>::zeros((0, 2)).view(),
            Array1::<f64>::zeros(0).view(),
            Array1::<f64>::zeros(0).view(),
        )
        .unwrap();
        assert_eq!(out.shape(), &[0, 7, 7]);
    }

    #[test]
    fn build_stamp_delta_round_trip() {
        // End-to-end pin of the delta sign/center convention shared by
        // build_stamp and render. A flipped delta sign would shift the
        // rendered PSF by ~2*delta and blow this tolerance.
        use crate::image::stamp::{
            DEFAULT_CENTROID_MAX_ITER, DEFAULT_CENTROID_TOL, DEFAULT_WEIGHT_FWHM, build_stamp,
        };

        let true_row = 7.35_f64;
        let true_col = 6.80_f64;
        let amplitude = 100.0;
        let sigma_det = 1.6;
        let back = 5.0;

        // Isolated Gaussian point source in a generous rough cutout.
        let cutout_size = 15;
        let mut cutout = Array2::<f64>::from_elem((cutout_size, cutout_size), back);
        for r in 0..cutout_size {
            for c in 0..cutout_size {
                let dr = r as f64 - true_row;
                let dc = c as f64 - true_col;
                cutout[(r, c)] +=
                    amplitude * (-0.5 * (dr * dr + dc * dc) / (sigma_det * sigma_det)).exp();
            }
        }

        let stamp_size = 7;
        let stamp_result = build_stamp(
            cutout.view(),
            stamp_size,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp for an isolated source");

        // Reconstructed centroid (origin + center + delta) must recover
        // the true source location.
        let detector_center = (stamp_size as i64 - 1) / 2;
        let recon_row =
            (stamp_result.origin.0 + detector_center) as f64 + stamp_result.delta.0;
        let recon_col =
            (stamp_result.origin.1 + detector_center) as f64 + stamp_result.delta.1;
        assert!(
            (recon_row - true_row).abs() < 0.05 && (recon_col - true_col).abs() < 0.05,
            "reconstructed centroid ({recon_row}, {recon_col}) far from ({true_row}, {true_col})"
        );

        // Build the matching oversampled Gaussian model and render with
        // the delta build_stamp reported.
        let oversample = 5;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * sigma_det, amplitude);
        let delta = arr2(&[[stamp_result.delta.0, stamp_result.delta.1]]);
        let rendered = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, back).view(),
        )
        .unwrap();

        let mut max_abs_error = 0.0_f64;
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                max_abs_error =
                    max_abs_error.max((rendered[(0, i, j)] - stamp_result.stamp[(i, j)]).abs());
            }
        }
        // Centroid-bias + interpolation residual stay well under 2% of
        // the peak; a sign flip would be an order of magnitude larger.
        assert!(
            max_abs_error < 0.02 * amplitude,
            "round-trip max abs error = {max_abs_error}"
        );
    }

    #[test]
    fn rejects_transposed_indexing_with_asymmetric_delta() {
        // Asymmetric model + asymmetric delta: swapping (row, col)
        // anywhere in the pipeline would change the result.
        let oversample = 5;
        let stamp_size = 5;
        let epsf_side = oversample * stamp_size;
        let epsf = ramp_epsf(epsf_side);
        let delta = arr2(&[[0.4, -0.1]]);
        let out = render(
            epsf.view(),
            oversample,
            delta.view(),
            Array1::from_elem(1, 1.0).view(),
            Array1::from_elem(1, 0.0).view(),
        )
        .unwrap();
        // Hand-compute the (0, 0) detector pixel via the documented
        // formula and the same kernel.
        let psf_center = (epsf_side as f64 - 1.0) / 2.0;
        let detector_center = (stamp_size as f64 - 1.0) / 2.0;
        let k_u = psf_center + oversample as f64 * ((0.0 - detector_center) - 0.4);
        let k_v = psf_center + oversample as f64 * ((0.0 - detector_center) - (-0.1));
        let expected = catmull_rom_sample(&epsf.view(), k_u, k_v);
        assert!(
            (out[(0, 0, 0)] - expected).abs() < 1e-12,
            "{} != {}",
            out[(0, 0, 0)],
            expected
        );
        // Sanity: the model is genuinely asymmetric so this is a real
        // constraint.
        assert!((epsf[(0, 1)] - epsf[(1, 0)]).abs() > 1.0);
    }
}

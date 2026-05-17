//! Per-star nuisance refinement: given the current effective PSF, re-fit
//! each star's flux, background and sub-pixel centroid against its own
//! detector stamp.
//!
//! The forward model for one stamp is
//! `pred = flux * S(epsf; delta) + background`, where `S` is the
//! Catmull-Rom point-sampling operator of [`super::render`] (decision
//! 12). Given `epsf`, the stars are mutually independent; each is a
//! small weighted-chi-square fit on its own stamp. With `delta` fixed,
//! `(flux, background)` is an *exact* 2x2 weighted linear least-squares
//! solve. `delta` is nonlinear but differentiable: the analytic
//! Catmull-Rom shift derivative (decisions 4/10, no autodiff) gives a
//! Gauss-Newton step. The two are alternated as a block-coordinate
//! inner iteration.
//!
//! Design notes:
//!
//! - **Hybrid API.** [`solve_flux_background`] is the public leaf: the
//!   exact 2x2 conditional `(flux, background)` solve at a *fixed*
//!   `delta`, with no iteration -- an independently usable "given
//!   `epsf` + `delta`, give me `(f, b)`" primitive.
//!   [`refine_nuisance`] is the public batched operator: it owns the
//!   `{solve (f,b) | delta ; Gauss-Newton delta | (f,b)}` alternation
//!   and the convergence control, calling the leaf internally plus a
//!   *private* GN-delta step. The GN-delta step is only meaningful
//!   inside the alternation, so it is never exposed (a half-finished
//!   primitive invites misuse). All business logic stays in the core
//!   crate (decision 9): a Rust-only caller gets the full refinement in
//!   one call.
//!
//! - **Inner iteration drive.** `refine_nuisance` exposes `max_iter`
//!   and `tol` (mirroring `build_stamp`). It runs to per-star
//!   convergence or `max_iter`. `max_iter = 1` is exactly one
//!   block-coordinate step, which is what the Phase 6 super-resolution
//!   outer loop calls once per round; a large `max_iter` refines a
//!   batch of stars standalone against a fixed `epsf`.
//!
//! - **Data path, generic.** Like `robust_combine`/`build_stamp` (and
//!   unlike `render`/`accumulate`, whose `epsf` model is always `f64`),
//!   the detector `data` is generic over `T: Float`; `weight` shares
//!   `T` with `data`. The `epsf` model is always `f64` (same as
//!   `render`). Computation is in `f64`; outputs are `f64`/`bool`.
//!
//! - **Weight / validity.** `weight` is the *complete* per-pixel weight
//!   the solver has already formed (`valid * 1/sigma^2 * robust`);
//!   nuisance computes no robust factor itself (decision 7), mirroring
//!   `accumulate`. `None` is unit weight everywhere, matching the
//!   `Option` polarity of the rest of the module. A pixel is valid iff
//!   its `data` is finite AND (when a weight array is given) its weight
//!   is finite and strictly positive; mask polarity is the caller's job
//!   (a `true = invalid` pixel folded into `weight = 0` upstream).
//!
//! - **`flux`/`background` are pure outputs.** They are never inputs
//!   (decision 6 -- the caller passes neither). `background` is a scalar
//!   in v1; a plane/gradient model is left as a hook. Gauge fixing /
//!   `epsf` normalization is *not* done here -- that belongs to the
//!   Phase 6 `build_epsf` loop (decision 5). `delta_init` must be
//!   supplied (it comes from `build_stamp`): this is in-basin
//!   refinement, not from-scratch astrometry (decision 4).
//!
//! - **delta coordinate / sign convention (inherited, not new).**
//!   `delta = (delta_row, delta_col)` on axes (0, 1), the same column
//!   order as `build_stamp`'s `(f64, f64)` and `render`'s `(N, 2)`;
//!   semantically `delta = centroid - round(centroid)` in
//!   `[-0.5, 0.5)` (decision 12 / Phase 2). The render sampling is
//!   `k_u = K_c + os * ((i - c_det) - delta_row)` (and `k_v`
//!   symmetrically), so `dk_u/ddelta_row = -os` and the analytic
//!   centroid Jacobian is
//!   `dS/ddelta_row = -os * sum epsf * w'_u * w_v`,
//!   `dpred/ddelta_row = flux * dS/ddelta_row`. The sign is *derived*
//!   from the render formula, not chosen independently; it is pinned by
//!   the analytic-vs-finite-difference and the
//!   build_stamp -> render -> refine round-trip tests so a GN-delta
//!   sign flip cannot pass.
//!
//! - **Per-star failure vs convergence (two booleans).** `ok = false`
//!   means the algorithm failed for that star (too few valid pixels, a
//!   singular 2x2 normal matrix in the `(f,b)` or GN-delta solve, or
//!   `delta` leaving the convergence basin). On failure all three of
//!   `flux`, `background`, `delta` are set *atomically* to the sentinel
//!   `(NaN, NaN, delta_init)` so Phase 6 can take the triple or drop it
//!   as a unit. `converged` is pure information: `‖step‖ < tol` reached
//!   *before* the `max_iter` budget was exhausted. Phase 6 keys off
//!   `ok` only; `converged` is commonly `false` when `!ok` or
//!   `max_iter = 1`. None of this is a global `Err` (the same spirit as
//!   `robust_combine`'s per-pixel NaN sentinel).
//!
//! - **Errors are hard preconditions only.** Every shared
//!   [`NuisanceError`] variant is a shape/parity precondition; there is
//!   no algorithmic `Ok(None)` path (the `render`/`accumulate`/`robust`
//!   family, contrasting `build_stamp`). They are checked in
//!   declaration order: the shared shape/parity set first, then
//!   `RefineParamsInvalid`, which is specific to `refine_nuisance`
//!   (`max_iter == 0` or `tol` not finite-and-positive; mirroring
//!   `robust_combine`'s `ClippedMeanInvalidParams`).
//!   `solve_flux_background` has no `max_iter`/`tol`, so it is exempt
//!   from that one precondition (as `Median` is exempt from
//!   `ClippedMeanInvalidParams`). `N = 0` is legal (empty outputs).
//!
//! - **Parallel over N.** Stars are independent given `epsf`; the batch
//!   is driven with rayon over `N`, each star writing only its own
//!   outputs (the `render` parallel structure, no write contention).
//!
//! - **Landing decisions (solver internals; decisions 4/5/10
//!   compliant).** A symmetric 2x2 normal matrix `[[a11, a12],
//!   [a12, a22]]` is treated singular when a diagonal entry is
//!   non-positive or `det <= 1e-12 * a11 * a22`; this guards both the
//!   `(f,b)` solve (a constant `S` over the stamp cannot separate flux
//!   from background) and the GN-delta solve. A star needs at least
//!   `MIN_VALID_PIXELS = 3` valid pixels (more than the two `(f,b)`
//!   parameters) or it fails. The GN-delta update is pure Gauss-Newton
//!   with a per-component clamp of `MAX_DELTA_STEP = 1.0` detector
//!   pixel (a divergence safety net only; well-posed steps are far
//!   smaller, so convergence to the truth is unaffected); after the
//!   update, `max(|delta_row|, |delta_col|) > DELTA_BASIN = 1.0` is
//!   declared out-of-basin and fails the star (the true and
//!   `build_stamp`-init `delta` both lie in `[-0.5, 0.5)`, so wandering
//!   more than a pixel signals a wrong/blended star, decision 4).
//!   Alternation order is `solve (f,b) | delta` then GN-delta; the
//!   convergence step norm is the Euclidean norm of the *applied*
//!   (clamped) `delta` update; after the loop `(f,b)` is re-solved once
//!   at the final `delta` so the returned triple is mutually
//!   consistent. The clip/GN statistics use the full per-pixel weight
//!   `W` over the valid set (decision 7); no robust factor is computed
//!   here. `data`/`weight` are upcast per pixel via `T::to_f64()`
//!   rather than copied into full `f64` arrays, since `data` is the
//!   large 3-D input.

use ndarray::{Array1, Array2, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use thiserror::Error;

use crate::float::Float;

use super::kernel::{
    catmull_rom_sample, catmull_rom_weight_derivatives, catmull_rom_weights,
};

/// Minimum valid-pixel count for a star to be solvable (more than the
/// two `(flux, background)` parameters).
const MIN_VALID_PIXELS: usize = 3;

/// A symmetric 2x2 normal matrix is treated singular when its
/// determinant drops below this fraction of the product of its diagonal
/// entries (or a diagonal entry is non-positive).
const SINGULAR_RELATIVE_EPS: f64 = 1e-12;

/// Per-component clamp on a single Gauss-Newton `delta` update
/// (divergence safety net; well-posed steps are far smaller).
const MAX_DELTA_STEP: f64 = 1.0;

/// After a `delta` update, `max(|delta_row|, |delta_col|)` beyond this
/// is declared out-of-basin and fails the star (decision 4: the true
/// and init `delta` both lie in `[-0.5, 0.5)`).
const DELTA_BASIN: f64 = 1.0;

/// Errors returned by [`solve_flux_background`] / [`refine_nuisance`]
/// for ill-shaped / ill-parameterized inputs.
///
/// Every variant is a hard precondition. Like `render`/`accumulate`/
/// `robust_combine` (and unlike `build_stamp`), there is no algorithmic
/// skip path: a well-formed call always yields the per-star outputs (a
/// star that cannot be solved is handled per-star with the
/// `(NaN, NaN, delta_init)` sentinel and `ok = false`, not an `Err`).
/// `RefineParamsInvalid` is checked only by `refine_nuisance`, after
/// the shared shape/parity preconditions.
#[derive(Debug, Error, PartialEq)]
pub enum NuisanceError {
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
        "batch dimensions disagree: data shape {data:?} must be (N, s, s) with s the derived stamp size and delta shape {delta:?} must be (N, 2)"
    )]
    BatchLengthMismatch {
        data: (usize, usize, usize),
        delta: (usize, usize),
    },
    #[error("weight shape {weight:?} must equal data shape {data:?}")]
    WeightShapeMismatch {
        weight: (usize, usize, usize),
        data: (usize, usize, usize),
    },
    #[error(
        "refine_nuisance requires max_iter > 0 and tol finite and > 0; got max_iter = {max_iter}, tol = {tol}"
    )]
    RefineParamsInvalid { max_iter: usize, tol: f64 },
}

/// Output of [`solve_flux_background`]: the per-star `(flux,
/// background)` at the supplied fixed `delta`.
#[derive(Debug, Clone, PartialEq)]
pub struct FluxBackground {
    /// `(N,)` per-star flux `f_i` (pure output; decision 6). `NaN` for
    /// a star that could not be solved.
    pub flux: Array1<f64>,
    /// `(N,)` per-star scalar background `b_i` (plane/gradient hook
    /// left for later; decision 6). `NaN` for an unsolved star.
    pub background: Array1<f64>,
    /// `(N,)` per-star solvability: the 2x2 normal matrix was
    /// non-singular and there were enough valid pixels. When `false`,
    /// `flux` and `background` are both the `NaN` sentinel.
    pub ok: Array1<bool>,
}

/// Output of [`refine_nuisance`]: the per-star refined `(flux,
/// background, delta)` plus the failure / convergence flags.
#[derive(Debug, Clone, PartialEq)]
pub struct NuisanceRefined {
    /// `(N,)` refined flux (pure output). `NaN` when `!ok`.
    pub flux: Array1<f64>,
    /// `(N,)` refined scalar background. `NaN` when `!ok`.
    pub background: Array1<f64>,
    /// `(N, 2)` refined `(delta_row, delta_col)`. Equals the supplied
    /// `delta_init` row when `!ok` (the sentinel is atomic with
    /// `flux`/`background`).
    pub delta: Array2<f64>,
    /// `(N,)` per-star success: the star produced a usable finite
    /// `(f, b, delta)`. `false` means all three are sentinels (atomic);
    /// Phase 6 keys off this flag.
    pub ok: Array1<bool>,
    /// `(N,)` informational: `‖step‖ < tol` was reached *before* the
    /// `max_iter` budget was exhausted. Commonly `false` when `!ok` or
    /// `max_iter = 1`. Phase 6 does *not* key off this flag.
    pub converged: Array1<bool>,
}

/// Resolved, validated stamp geometry shared by both entry points.
struct StampGeometry {
    stamp_size: usize,
    batch_size: usize,
    psf_center: f64,
    detector_center: f64,
    oversample_f: f64,
}

/// Validate the shape/parity preconditions shared by both entry points
/// and resolve the stamp geometry. Checked in declaration order, the
/// first failure returned (mirrors `render`'s precondition order; the
/// derived `stamp_size` comes from `epsf.shape()` / `oversample`, never
/// passed separately).
fn validate_shared<T: Float>(
    epsf: &ArrayView2<f64>,
    oversample: usize,
    data: &ArrayView3<T>,
    weight: &Option<ArrayView3<T>>,
    delta: &ArrayView2<f64>,
) -> Result<StampGeometry, NuisanceError> {
    let epsf_rows = epsf.shape()[0];
    let epsf_cols = epsf.shape()[1];
    if epsf_rows != epsf_cols {
        return Err(NuisanceError::EpsfNotSquare {
            rows: epsf_rows,
            cols: epsf_cols,
        });
    }
    if oversample % 2 == 0 {
        // Also rejects oversample == 0 (0 is even, hence not odd).
        return Err(NuisanceError::OversampleNotOdd { oversample });
    }
    let epsf_side = epsf_rows;
    if epsf_side % oversample != 0 {
        return Err(NuisanceError::EpsfSizeNotMultiple {
            epsf_side,
            oversample,
        });
    }
    let stamp_size = epsf_side / oversample;
    if stamp_size % 2 == 0 {
        // Catches the degenerate empty model (stamp_size == 0) too.
        return Err(NuisanceError::DerivedStampSizeEven { stamp_size });
    }
    let batch_size = data.shape()[0];
    if data.shape()[1] != stamp_size
        || data.shape()[2] != stamp_size
        || delta.shape()[0] != batch_size
        || delta.shape()[1] != 2
    {
        return Err(NuisanceError::BatchLengthMismatch {
            data: (data.shape()[0], data.shape()[1], data.shape()[2]),
            delta: (delta.shape()[0], delta.shape()[1]),
        });
    }
    if let Some(weight_view) = weight {
        let weight_shape = weight_view.shape();
        if weight_shape[0] != batch_size
            || weight_shape[1] != stamp_size
            || weight_shape[2] != stamp_size
        {
            return Err(NuisanceError::WeightShapeMismatch {
                weight: (weight_shape[0], weight_shape[1], weight_shape[2]),
                data: (batch_size, stamp_size, stamp_size),
            });
        }
    }

    // `epsf_side` and `stamp_size` are both odd, so these centers are
    // exact integers in f64 -- identical to `render`, which keeps the
    // analytic Jacobian the exact derivative of the forward sample.
    Ok(StampGeometry {
        stamp_size,
        batch_size,
        psf_center: (epsf_side as f64 - 1.0) / 2.0,
        detector_center: (stamp_size as f64 - 1.0) / 2.0,
        oversample_f: oversample as f64,
    })
}

/// One valid pixel of a stamp: its `(row, column)` and the upcast
/// `(data, weight)`. The validity gate is data-and-weight only (it does
/// not depend on `delta` or the model), so it is gathered once per star
/// and reused across the alternation.
struct ValidPixel {
    row: usize,
    column: usize,
    data: f64,
    weight: f64,
}

/// Gather the valid pixels of one stamp. A pixel is valid iff its data
/// is finite AND (when a weight array is given) its weight is finite
/// and strictly positive; `weight == None` is unit weight (decision 7).
fn gather_valid_pixels<T: Float>(
    data_stamp: &ArrayView2<T>,
    weight_stamp: &Option<ArrayView2<T>>,
    stamp_size: usize,
) -> Vec<ValidPixel> {
    let mut valid = Vec::with_capacity(stamp_size * stamp_size);
    for row in 0..stamp_size {
        for column in 0..stamp_size {
            let value = data_stamp[(row, column)].to_f64().unwrap_or(f64::NAN);
            if !value.is_finite() {
                continue;
            }
            let pixel_weight = match weight_stamp {
                Some(weight_view) => {
                    let w = weight_view[(row, column)].to_f64().unwrap_or(f64::NAN);
                    if !(w.is_finite() && w > 0.0) {
                        continue;
                    }
                    w
                }
                None => 1.0,
            };
            valid.push(ValidPixel {
                row,
                column,
                data: value,
                weight: pixel_weight,
            });
        }
    }
    valid
}

/// Solve a symmetric 2x2 system `[[a11, a12], [a12, a22]] x = (r1, r2)`.
///
/// Returns `None` (singular) when a diagonal entry is non-positive or
/// `det <= SINGULAR_RELATIVE_EPS * a11 * a22`, or when the solution is
/// not finite. Used by both the `(f, b)` solve and the GN-delta solve.
fn solve_symmetric_2x2(
    a11: f64,
    a12: f64,
    a22: f64,
    r1: f64,
    r2: f64,
) -> Option<(f64, f64)> {
    // Reject a non-positive or non-finite diagonal (a constant model
    // gives a zero/near-zero diagonal). Plain comparisons plus an
    // explicit finiteness check stay NaN/inf-safe without a negated
    // partial-order comparison.
    if !a11.is_finite() || a11 <= 0.0 || !a22.is_finite() || a22 <= 0.0 {
        return None;
    }
    let determinant = a11 * a22 - a12 * a12;
    let threshold = SINGULAR_RELATIVE_EPS * a11 * a22;
    if !determinant.is_finite() || determinant <= threshold {
        return None;
    }
    let x1 = (r1 * a22 - r2 * a12) / determinant;
    let x2 = (a11 * r2 - a12 * r1) / determinant;
    if x1.is_finite() && x2.is_finite() {
        Some((x1, x2))
    } else {
        None
    }
}

/// Catmull-Rom sample of `epsf` at `(k_u, k_v)` together with its
/// derivatives with respect to the centroid offset.
///
/// The value is computed with exactly the weights and tap base of
/// [`catmull_rom_sample`] (it equals it tap-for-tap, pinned by a test).
/// The derivatives reuse [`catmull_rom_weight_derivatives`] over the
/// *same* base and zero-padding, then apply `dk/ddelta = -oversample`
/// (and `dfrac/dk = 1`): `dS/ddelta_row = -os * sum epsf * w'_u * w_v`
/// and `dS/ddelta_col = -os * sum epsf * w_u * w'_v`. Sharing the
/// kernel weight/derivative source (rather than re-deriving Catmull-Rom
/// here) is what makes this the exact derivative of the forward sample.
fn sample_value_and_delta_jacobian(
    epsf: &ArrayView2<f64>,
    k_u: f64,
    k_v: f64,
    oversample: f64,
) -> (f64, f64, f64) {
    let side = epsf.shape()[0] as i64; // square: shape()[0] == shape()[1]

    let u_floor = k_u.floor();
    let v_floor = k_v.floor();
    let frac_u = k_u - u_floor;
    let frac_v = k_v - v_floor;
    let weights_u = catmull_rom_weights(frac_u);
    let weights_v = catmull_rom_weights(frac_v);
    let dweights_u = catmull_rom_weight_derivatives(frac_u);
    let dweights_v = catmull_rom_weight_derivatives(frac_v);

    // Same floor-derived tap base and zero-padding as
    // `catmull_rom_sample`; the value below is identical to it.
    let base_u = u_floor as i64 - 1;
    let base_v = v_floor as i64 - 1;

    let mut value = 0.0_f64;
    let mut dvalue_dku = 0.0_f64;
    let mut dvalue_dkv = 0.0_f64;
    for tap_u in 0..4 {
        let model_row = base_u + tap_u as i64;
        if model_row < 0 || model_row >= side {
            continue; // zero-pad outside the model grid
        }
        let weight_u = weights_u[tap_u];
        let dweight_u = dweights_u[tap_u];
        for tap_v in 0..4 {
            let model_column = base_v + tap_v as i64;
            if model_column < 0 || model_column >= side {
                continue; // zero-pad outside the model grid
            }
            let psi = epsf[(model_row as usize, model_column as usize)];
            value += psi * weight_u * weights_v[tap_v];
            dvalue_dku += psi * dweight_u * weights_v[tap_v];
            dvalue_dkv += psi * weight_u * dweights_v[tap_v];
        }
    }
    // dk_u/ddelta_row = -oversample, dk_v/ddelta_col = -oversample.
    (value, -oversample * dvalue_dku, -oversample * dvalue_dkv)
}

/// Model coordinate of detector pixel `(i, j)` for the given `delta`
/// (the render / decision-12 sampling formula).
#[inline]
fn model_coordinate(
    pixel: usize,
    delta: f64,
    psf_center: f64,
    detector_center: f64,
    oversample_f: f64,
) -> f64 {
    psf_center + oversample_f * ((pixel as f64 - detector_center) - delta)
}

/// Exact `(flux, background)` solve at a fixed `delta` for one star:
/// the weighted normal equations
/// `[[sum W g^2, sum W g], [sum W g, sum W]] (f, b) = (sum W g D, sum W D)`
/// with `g = S(epsf; delta)`. `None` when there are too few valid
/// pixels or the 2x2 is singular (a constant `g` cannot separate flux
/// from background).
fn solve_flux_background_one_star(
    epsf: &ArrayView2<f64>,
    valid: &[ValidPixel],
    delta_row: f64,
    delta_column: f64,
    geometry: &StampGeometry,
) -> Option<(f64, f64)> {
    if valid.len() < MIN_VALID_PIXELS {
        return None;
    }
    let mut sum_w_gg = 0.0;
    let mut sum_w_g = 0.0;
    let mut sum_w = 0.0;
    let mut sum_w_g_d = 0.0;
    let mut sum_w_d = 0.0;
    for pixel in valid {
        let k_u = model_coordinate(
            pixel.row,
            delta_row,
            geometry.psf_center,
            geometry.detector_center,
            geometry.oversample_f,
        );
        let k_v = model_coordinate(
            pixel.column,
            delta_column,
            geometry.psf_center,
            geometry.detector_center,
            geometry.oversample_f,
        );
        let g = catmull_rom_sample(epsf, k_u, k_v);
        let w = pixel.weight;
        sum_w_gg += w * g * g;
        sum_w_g += w * g;
        sum_w += w;
        sum_w_g_d += w * g * pixel.data;
        sum_w_d += w * pixel.data;
    }
    solve_symmetric_2x2(sum_w_gg, sum_w_g, sum_w, sum_w_g_d, sum_w_d)
}

/// One private Gauss-Newton step on `delta` with `(flux, background)`
/// held fixed. Linearizing `pred = flux * S + b` about `delta`, the
/// normal equations are `(sum W p p^T) step = sum W p r` with
/// `p = dpred/ddelta = flux * dS/ddelta` and `r = D - pred`. Returns
/// the *raw* (unclamped) `(step_row, step_col)`, or `None` when the 2x2
/// is singular / non-finite. Private: only meaningful inside the
/// alternation (decision: GN-delta step is never exposed).
fn gauss_newton_delta_step(
    epsf: &ArrayView2<f64>,
    valid: &[ValidPixel],
    delta_row: f64,
    delta_column: f64,
    flux: f64,
    background: f64,
    geometry: &StampGeometry,
) -> Option<(f64, f64)> {
    let mut h_rr = 0.0;
    let mut h_rc = 0.0;
    let mut h_cc = 0.0;
    let mut g_r = 0.0;
    let mut g_c = 0.0;
    for pixel in valid {
        let k_u = model_coordinate(
            pixel.row,
            delta_row,
            geometry.psf_center,
            geometry.detector_center,
            geometry.oversample_f,
        );
        let k_v = model_coordinate(
            pixel.column,
            delta_column,
            geometry.psf_center,
            geometry.detector_center,
            geometry.oversample_f,
        );
        let (sample, ds_ddelta_row, ds_ddelta_col) =
            sample_value_and_delta_jacobian(epsf, k_u, k_v, geometry.oversample_f);
        let residual = pixel.data - (flux * sample + background);
        let p_row = flux * ds_ddelta_row;
        let p_col = flux * ds_ddelta_col;
        let w = pixel.weight;
        h_rr += w * p_row * p_row;
        h_rc += w * p_row * p_col;
        h_cc += w * p_col * p_col;
        g_r += w * p_row * residual;
        g_c += w * p_col * residual;
    }
    solve_symmetric_2x2(h_rr, h_rc, h_cc, g_r, g_c)
}

/// Refine one star: alternate `solve (f,b) | delta` and a Gauss-Newton
/// `delta | (f,b)` step until `‖step‖ < tol` or `max_iter`. Returns
/// `(flux, background, delta_row, delta_col, ok, converged)`; on any
/// failure all of `(flux, background, delta)` are the atomic sentinel
/// `(NaN, NaN, delta_init)`.
#[allow(clippy::too_many_arguments)]
fn refine_one_star<T: Float>(
    epsf: &ArrayView2<f64>,
    data_stamp: &ArrayView2<T>,
    weight_stamp: &Option<ArrayView2<T>>,
    delta_init_row: f64,
    delta_init_column: f64,
    max_iter: usize,
    tol: f64,
    geometry: &StampGeometry,
) -> (f64, f64, f64, f64, bool, bool) {
    let failure = (
        f64::NAN,
        f64::NAN,
        delta_init_row,
        delta_init_column,
        false,
        false,
    );

    let valid = gather_valid_pixels(data_stamp, weight_stamp, geometry.stamp_size);
    if valid.len() < MIN_VALID_PIXELS {
        return failure;
    }

    let mut delta_row = delta_init_row;
    let mut delta_column = delta_init_column;
    let mut converged = false;

    // `max_iter >= 1` and `tol` finite-and-positive are guaranteed by
    // the RefineParamsInvalid precondition.
    for iteration in 0..max_iter {
        let (flux, background) = match solve_flux_background_one_star(
            epsf,
            &valid,
            delta_row,
            delta_column,
            geometry,
        ) {
            Some(value) => value,
            None => return failure,
        };
        let (step_row_raw, step_column_raw) = match gauss_newton_delta_step(
            epsf,
            &valid,
            delta_row,
            delta_column,
            flux,
            background,
            geometry,
        ) {
            Some(value) => value,
            None => return failure,
        };

        // Pure Gauss-Newton with a per-component clamp (divergence
        // safety net only -- well-posed steps are far below the clamp).
        let step_row = step_row_raw.clamp(-MAX_DELTA_STEP, MAX_DELTA_STEP);
        let step_column = step_column_raw.clamp(-MAX_DELTA_STEP, MAX_DELTA_STEP);
        let new_delta_row = delta_row + step_row;
        let new_delta_column = delta_column + step_column;
        if !new_delta_row.is_finite()
            || !new_delta_column.is_finite()
            || new_delta_row.abs() > DELTA_BASIN
            || new_delta_column.abs() > DELTA_BASIN
        {
            // Out of the convergence basin -> wrong/blended star
            // (decision 4): atomic sentinel.
            return failure;
        }
        delta_row = new_delta_row;
        delta_column = new_delta_column;

        // Convergence is measured on the *applied* (clamped) step.
        let step_norm = (step_row * step_row + step_column * step_column).sqrt();
        if step_norm < tol {
            // "converged" only if we stopped strictly before exhausting
            // the budget (so max_iter = 1 is always `false`).
            converged = iteration + 1 < max_iter;
            break;
        }
    }

    // Re-solve (f,b) once at the final delta so the returned triple is
    // mutually consistent.
    match solve_flux_background_one_star(epsf, &valid, delta_row, delta_column, geometry)
    {
        Some((flux, background)) => {
            (flux, background, delta_row, delta_column, true, converged)
        }
        None => failure,
    }
}

/// Solve the per-star `(flux, background)` at a *fixed* `delta` against
/// the current effective PSF -- the exact 2x2 weighted linear
/// least-squares leaf, with no iteration.
///
/// See the module-level documentation for the model, the data-path
/// generic / `f64` `epsf` split, the weight/validity semantics, and the
/// per-star `ok` sentinel.
///
/// # Parameters
///
/// - `epsf`: oversampled effective PSF, square `(os*s, os*s)`, always
///   `f64`.
/// - `oversample`: `os`, forced odd.
/// - `data`: `(N, s, s)` per-stamp detector data; may be `f32` or
///   `f64`, upcast internally. `s` is derived from `epsf.shape()` /
///   `oversample`.
/// - `weight`: optional `(N, s, s)` complete per-pixel weight; `None`
///   is unit weight.
/// - `delta`: `(N, 2)` fixed per-stamp `(row, col)` offset (this leaf
///   does not refine `delta`).
///
/// # Returns
///
/// `Ok(FluxBackground)` with `(N,)` `flux`, `background`, `ok`. A star
/// with too few valid pixels or a singular 2x2 has `ok = false` and
/// `flux = background = NaN`. `N = 0` yields empty arrays.
///
/// # Errors
///
/// Returns a [`NuisanceError`] for the shared shape/parity
/// preconditions (square `epsf`, odd `oversample`, `oversample`
/// dividing the `epsf` side, odd derived `stamp_size`, `data` being
/// `(N, s, s)` with `delta` `(N, 2)`, and a `Some` `weight` matching
/// `data`). This leaf has no `max_iter`/`tol`, so it never returns
/// `RefineParamsInvalid`.
pub fn solve_flux_background<T: Float>(
    epsf: ArrayView2<f64>,
    oversample: usize,
    data: ArrayView3<T>,
    weight: Option<ArrayView3<T>>,
    delta: ArrayView2<f64>,
) -> Result<FluxBackground, NuisanceError> {
    let geometry = validate_shared(&epsf, oversample, &data, &weight, &delta)?;

    let per_star: Vec<(f64, f64, bool)> = (0..geometry.batch_size)
        .into_par_iter()
        .map(|star_index| {
            let data_stamp = data.index_axis(Axis(0), star_index);
            let weight_stamp =
                weight.as_ref().map(|w| w.index_axis(Axis(0), star_index));
            let valid =
                gather_valid_pixels(&data_stamp, &weight_stamp, geometry.stamp_size);
            match solve_flux_background_one_star(
                &epsf,
                &valid,
                delta[(star_index, 0)],
                delta[(star_index, 1)],
                &geometry,
            ) {
                Some((flux, background)) => (flux, background, true),
                None => (f64::NAN, f64::NAN, false),
            }
        })
        .collect();

    let mut flux = Vec::with_capacity(geometry.batch_size);
    let mut background = Vec::with_capacity(geometry.batch_size);
    let mut ok = Vec::with_capacity(geometry.batch_size);
    for (star_flux, star_background, star_ok) in per_star {
        flux.push(star_flux);
        background.push(star_background);
        ok.push(star_ok);
    }

    Ok(FluxBackground {
        flux: Array1::from(flux),
        background: Array1::from(background),
        ok: Array1::from(ok),
    })
}

/// Refine the per-star `(flux, background, delta)` against the current
/// effective PSF: the batched block-coordinate operator that alternates
/// the exact `(f,b)` solve and a private Gauss-Newton `delta` step.
///
/// See the module-level documentation for the alternation, the
/// `max_iter`/`tol` drive, the `delta` sign convention and analytic
/// Jacobian, and the atomic per-star failure sentinel / `ok` vs
/// `converged` split.
///
/// # Parameters
///
/// - `epsf`, `oversample`, `data`, `weight`: as in
///   [`solve_flux_background`].
/// - `delta_init`: `(N, 2)` per-stamp `delta` initial value (from
///   `build_stamp`). This is in-basin refinement, not from-scratch
///   astrometry (decision 4).
/// - `max_iter`: per-star alternation cap (must be `> 0`).
///   `max_iter = 1` is exactly one block-coordinate step.
/// - `tol`: `‖delta step‖` convergence threshold (must be finite and
///   `> 0`).
///
/// # Returns
///
/// `Ok(NuisanceRefined)` with `(N,)` `flux`, `background`, `ok`,
/// `converged` and `(N, 2)` `delta`. A failed star (`ok = false`) has
/// the atomic sentinel `flux = background = NaN`, `delta = delta_init`.
/// `converged` is informational only. `N = 0` yields empty arrays.
///
/// # Errors
///
/// Returns a [`NuisanceError`]: the shared shape/parity preconditions
/// (as in [`solve_flux_background`]) checked first, then
/// `RefineParamsInvalid` if `max_iter == 0` or `tol` is not finite and
/// strictly positive.
pub fn refine_nuisance<T: Float>(
    epsf: ArrayView2<f64>,
    oversample: usize,
    data: ArrayView3<T>,
    weight: Option<ArrayView3<T>>,
    delta_init: ArrayView2<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<NuisanceRefined, NuisanceError> {
    let geometry = validate_shared(&epsf, oversample, &data, &weight, &delta_init)?;

    // refine_nuisance-specific precondition, after the shared set
    // (mirrors robust_combine's ClippedMeanInvalidParams ordering).
    let params_invalid = max_iter == 0 || !(tol > 0.0 && tol.is_finite());
    if params_invalid {
        return Err(NuisanceError::RefineParamsInvalid { max_iter, tol });
    }

    let per_star: Vec<(f64, f64, f64, f64, bool, bool)> = (0..geometry.batch_size)
        .into_par_iter()
        .map(|star_index| {
            let data_stamp = data.index_axis(Axis(0), star_index);
            let weight_stamp =
                weight.as_ref().map(|w| w.index_axis(Axis(0), star_index));
            refine_one_star(
                &epsf,
                &data_stamp,
                &weight_stamp,
                delta_init[(star_index, 0)],
                delta_init[(star_index, 1)],
                max_iter,
                tol,
                &geometry,
            )
        })
        .collect();

    let mut flux = Vec::with_capacity(geometry.batch_size);
    let mut background = Vec::with_capacity(geometry.batch_size);
    let mut delta_flat = Vec::with_capacity(geometry.batch_size * 2);
    let mut ok = Vec::with_capacity(geometry.batch_size);
    let mut converged = Vec::with_capacity(geometry.batch_size);
    for (star_flux, star_background, delta_row, delta_column, star_ok, star_converged) in
        per_star
    {
        flux.push(star_flux);
        background.push(star_background);
        delta_flat.push(delta_row);
        delta_flat.push(delta_column);
        ok.push(star_ok);
        converged.push(star_converged);
    }

    Ok(NuisanceRefined {
        flux: Array1::from(flux),
        background: Array1::from(background),
        delta: Array2::from_shape_vec((geometry.batch_size, 2), delta_flat)
            .expect("delta length == batch_size * 2"),
        ok: Array1::from(ok),
        converged: Array1::from(converged),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::psf::render::render;
    use ndarray::{Array1, Array2, Array3, arr2};

    /// SplitMix64: a tiny, dependency-free, deterministic PRNG so the
    /// random comparisons are reproducible (same generator as the
    /// `accumulate`/`robust` tests).
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Oversampled Gaussian effective PSF (matches the `render` test
    /// helper): centered on the global center cell, `sigma_model` in
    /// model-grid pixels, peak `amplitude`.
    fn gaussian_epsf(epsf_side: usize, sigma_model: f64, amplitude: f64) -> Array2<f64> {
        let center = (epsf_side as f64 - 1.0) / 2.0;
        let mut model = Array2::<f64>::zeros((epsf_side, epsf_side));
        for p in 0..epsf_side {
            for q in 0..epsf_side {
                let dp = p as f64 - center;
                let dq = q as f64 - center;
                model[(p, q)] = amplitude
                    * (-0.5 * (dp * dp + dq * dq) / (sigma_model * sigma_model)).exp();
            }
        }
        model
    }

    /// Deterministic non-separable, non-symmetric model so a row/col or
    /// transpose slip in the Jacobian cannot hide.
    fn ramp_epsf(epsf_side: usize) -> Array2<f64> {
        let mut model = Array2::<f64>::zeros((epsf_side, epsf_side));
        for p in 0..epsf_side {
            for q in 0..epsf_side {
                model[(p, q)] = 1.0 + p as f64 + 3.0 * q as f64 + 0.5 * (p * q) as f64;
            }
        }
        model
    }

    /// Naive per-star 2x2 weighted LLSQ reference (independent of the
    /// production normal-equation assembly; the kernel sample is the
    /// trusted shared primitive, pinned by its own tests).
    fn naive_solve_flux_background(
        epsf: &Array2<f64>,
        oversample: usize,
        data_stamp: &Array2<f64>,
        weight_stamp: Option<&Array2<f64>>,
        delta_row: f64,
        delta_column: f64,
    ) -> Option<(f64, f64)> {
        let epsf_side = epsf.shape()[0];
        let stamp_size = epsf_side / oversample;
        let psf_center = (epsf_side as f64 - 1.0) / 2.0;
        let detector_center = (stamp_size as f64 - 1.0) / 2.0;
        let os = oversample as f64;
        let mut s_gg = 0.0;
        let mut s_g = 0.0;
        let mut s_1 = 0.0;
        let mut s_gd = 0.0;
        let mut s_d = 0.0;
        let mut valid = 0usize;
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                let value = data_stamp[(i, j)];
                if !value.is_finite() {
                    continue;
                }
                let w = match weight_stamp {
                    Some(weight_array) => {
                        let w = weight_array[(i, j)];
                        if !(w.is_finite() && w > 0.0) {
                            continue;
                        }
                        w
                    }
                    None => 1.0,
                };
                let k_u = psf_center + os * ((i as f64 - detector_center) - delta_row);
                let k_v =
                    psf_center + os * ((j as f64 - detector_center) - delta_column);
                let g = catmull_rom_sample(&epsf.view(), k_u, k_v);
                s_gg += w * g * g;
                s_g += w * g;
                s_1 += w;
                s_gd += w * g * value;
                s_d += w * value;
                valid += 1;
            }
        }
        if valid < 3 {
            return None;
        }
        let det = s_gg * s_1 - s_g * s_g;
        if !(s_gg > 0.0) || !(s_1 > 0.0) || !(det > 1e-12 * s_gg * s_1) {
            return None;
        }
        Some((
            (s_gd * s_1 - s_d * s_g) / det,
            (s_gg * s_d - s_g * s_gd) / det,
        ))
    }

    /// Render a single synthetic stamp with known `(flux, background,
    /// delta)` so a refinement can be checked against the truth.
    fn synth_stamp(
        epsf: &Array2<f64>,
        oversample: usize,
        delta_row: f64,
        delta_column: f64,
        flux: f64,
        background: f64,
    ) -> Array2<f64> {
        let rendered = render(
            epsf.view(),
            oversample,
            arr2(&[[delta_row, delta_column]]).view(),
            Array1::from_elem(1, flux).view(),
            Array1::from_elem(1, background).view(),
        )
        .unwrap();
        rendered.index_axis(Axis(0), 0).to_owned()
    }

    fn stack_one(stamp: &Array2<f64>) -> Array3<f64> {
        let (h, w) = (stamp.shape()[0], stamp.shape()[1]);
        let mut stack = Array3::<f64>::zeros((1, h, w));
        stack.index_axis_mut(Axis(0), 0).assign(stamp);
        stack
    }

    // --- solve_flux_background vs naive reference. ---

    #[test]
    fn solve_flux_background_matches_naive_equal_weight() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.4, 80.0);
        let delta = arr2(&[[0.13, -0.21], [-0.30, 0.05], [0.40, 0.37]]);
        let mut rng = SplitMix64::new(0xA11C_E5F0_1234_5678);
        let mut data = Array3::<f64>::zeros((3, stamp_size, stamp_size));
        for n in 0..3 {
            let stamp = synth_stamp(
                &epsf,
                oversample,
                delta[(n, 0)],
                delta[(n, 1)],
                rng.range(1.0, 5.0),
                rng.range(-2.0, 4.0),
            );
            data.index_axis_mut(Axis(0), n).assign(&stamp);
        }
        let got =
            solve_flux_background(epsf.view(), oversample, data.view(), None, delta.view())
                .unwrap();
        for n in 0..3 {
            let (f_naive, b_naive) = naive_solve_flux_background(
                &epsf,
                oversample,
                &data.index_axis(Axis(0), n).to_owned(),
                None,
                delta[(n, 0)],
                delta[(n, 1)],
            )
            .unwrap();
            assert!(got.ok[n]);
            assert!(
                (got.flux[n] - f_naive).abs() < 1e-9 * f_naive.abs().max(1.0),
                "flux {} vs naive {f_naive}",
                got.flux[n]
            );
            assert!(
                (got.background[n] - b_naive).abs() < 1e-9 * b_naive.abs().max(1.0),
                "background {} vs naive {b_naive}",
                got.background[n]
            );
        }
    }

    #[test]
    fn solve_flux_background_matches_naive_weighted() {
        let oversample = 3;
        let stamp_size = 9;
        let epsf_side = oversample * stamp_size;
        let epsf = ramp_epsf(epsf_side);
        let delta = arr2(&[[0.22, -0.17]]);
        let data = stack_one(&synth_stamp(&epsf, oversample, 0.22, -0.17, 2.5, 1.5));
        let mut rng = SplitMix64::new(0x0BAD_F00D_9876_5432);
        let weight = Array3::from_shape_fn((1, stamp_size, stamp_size), |_| {
            rng.range(0.1, 3.0)
        });
        let got = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            Some(weight.view()),
            delta.view(),
        )
        .unwrap();
        let (f_naive, b_naive) = naive_solve_flux_background(
            &epsf,
            oversample,
            &data.index_axis(Axis(0), 0).to_owned(),
            Some(&weight.index_axis(Axis(0), 0).to_owned()),
            0.22,
            -0.17,
        )
        .unwrap();
        assert!(got.ok[0]);
        assert!((got.flux[0] - f_naive).abs() < 1e-9 * f_naive.abs().max(1.0));
        assert!((got.background[0] - b_naive).abs() < 1e-9 * b_naive.abs().max(1.0));
        // Synthetic truth was (2.5, 1.5); the exact LLSQ recovers it.
        assert!((got.flux[0] - 2.5).abs() < 1e-6);
        assert!((got.background[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn weight_none_equals_unit_weight_some() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.3, 50.0);
        let delta = arr2(&[[0.1, 0.2]]);
        let data = stack_one(&synth_stamp(&epsf, oversample, 0.1, 0.2, 3.0, 0.5));
        let none = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap();
        let unit = Array3::<f64>::ones((1, stamp_size, stamp_size));
        let some = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            Some(unit.view()),
            delta.view(),
        )
        .unwrap();
        assert!((none.flux[0] - some.flux[0]).abs() < 1e-12);
        assert!((none.background[0] - some.background[0]).abs() < 1e-12);
    }

    // --- refine_nuisance convergence to the truth. ---

    #[test]
    fn refine_converges_to_known_truth() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.4, 100.0);
        let true_flux = 4.5;
        let true_background = 3.2;
        let true_delta_row = 0.17;
        let true_delta_column = -0.23;
        let data = stack_one(&synth_stamp(
            &epsf,
            oversample,
            true_delta_row,
            true_delta_column,
            true_flux,
            true_background,
        ));
        // Perturbed init within the basin.
        let delta_init = arr2(&[[true_delta_row + 0.06, true_delta_column - 0.05]]);
        let tol = 1e-9;
        let result = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta_init.view(),
            60,
            tol,
        )
        .unwrap();
        assert!(result.ok[0]);
        assert!(result.converged[0], "expected early convergence");
        assert!(
            (result.delta[(0, 0)] - true_delta_row).abs() < 1e-4,
            "delta_row {} vs {true_delta_row}",
            result.delta[(0, 0)]
        );
        assert!(
            (result.delta[(0, 1)] - true_delta_column).abs() < 1e-4,
            "delta_col {} vs {true_delta_column}",
            result.delta[(0, 1)]
        );
        assert!(
            (result.flux[0] - true_flux).abs() < 1e-3 * true_flux,
            "flux {} vs {true_flux}",
            result.flux[0]
        );
        assert!(
            (result.background[0] - true_background).abs() < 1e-3,
            "background {} vs {true_background}",
            result.background[0]
        );
    }

    #[test]
    fn max_iter_one_is_single_block_step() {
        // max_iter = 1 must equal exactly one manual round:
        // solve(f,b)|d0 -> GN step -> clamp -> d1 -> solve(f,b)|d1,
        // with converged = false.
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.5, 70.0);
        let data = stack_one(&synth_stamp(&epsf, oversample, 0.12, -0.08, 2.0, 1.0));
        let delta_init_row = 0.20;
        let delta_init_column = -0.15;
        let delta_init = arr2(&[[delta_init_row, delta_init_column]]);
        let result = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta_init.view(),
            1,
            1e-9,
        )
        .unwrap();
        assert!(result.ok[0]);
        assert!(!result.converged[0], "max_iter = 1 must report converged = false");

        // Manual one round using the production helpers directly.
        let geometry = StampGeometry {
            stamp_size,
            batch_size: 1,
            psf_center: (epsf_side as f64 - 1.0) / 2.0,
            detector_center: (stamp_size as f64 - 1.0) / 2.0,
            oversample_f: oversample as f64,
        };
        let data_stamp = data.index_axis(Axis(0), 0);
        let weight_stamp: Option<ArrayView2<f64>> = None;
        let valid = gather_valid_pixels(&data_stamp, &weight_stamp, stamp_size);
        let (f0, b0) = solve_flux_background_one_star(
            &epsf.view(),
            &valid,
            delta_init_row,
            delta_init_column,
            &geometry,
        )
        .unwrap();
        let (sr_raw, sc_raw) = gauss_newton_delta_step(
            &epsf.view(),
            &valid,
            delta_init_row,
            delta_init_column,
            f0,
            b0,
            &geometry,
        )
        .unwrap();
        let sr = sr_raw.clamp(-MAX_DELTA_STEP, MAX_DELTA_STEP);
        let sc = sc_raw.clamp(-MAX_DELTA_STEP, MAX_DELTA_STEP);
        let d1_row = delta_init_row + sr;
        let d1_column = delta_init_column + sc;
        let (f1, b1) = solve_flux_background_one_star(
            &epsf.view(),
            &valid,
            d1_row,
            d1_column,
            &geometry,
        )
        .unwrap();
        assert!((result.delta[(0, 0)] - d1_row).abs() < 1e-12);
        assert!((result.delta[(0, 1)] - d1_column).abs() < 1e-12);
        assert!((result.flux[0] - f1).abs() < 1e-12);
        assert!((result.background[0] - b1).abs() < 1e-12);
    }

    // --- analytic delta Jacobian vs finite difference. ---

    #[test]
    fn sampler_value_equals_kernel_sample() {
        // The Jacobian sampler's value must be identical to the forward
        // kernel sample tap-for-tap (it is the exact same gather).
        let epsf = ramp_epsf(25);
        for &(k_u, k_v) in &[(11.3, 12.7), (5.0, 18.2), (0.4, 23.9), (12.0, 12.0)] {
            let (value, _, _) =
                sample_value_and_delta_jacobian(&epsf.view(), k_u, k_v, 5.0);
            let reference = catmull_rom_sample(&epsf.view(), k_u, k_v);
            assert!(
                (value - reference).abs() < 1e-15,
                "sampler value {value} != kernel {reference}"
            );
        }
    }

    #[test]
    fn delta_jacobian_matches_finite_difference() {
        // dS/ddelta vs a central difference of the kernel sample with
        // the model coordinate shifted by -+ os*h (k = ... - os*delta).
        // A sign flip here would reverse the GN-delta step.
        let oversample = 5.0;
        let epsf = ramp_epsf(35);
        let h = 1e-6;
        for &(k_u, k_v) in &[(16.4, 18.1), (10.7, 22.3), (20.0, 14.6)] {
            let (_, ds_drow, ds_dcol) =
                sample_value_and_delta_jacobian(&epsf.view(), k_u, k_v, oversample);
            // delta_row enters as k_u = ... - os*delta_row, so a +h step
            // in delta_row shifts k_u by -os*h.
            let plus_row = catmull_rom_sample(&epsf.view(), k_u - oversample * h, k_v);
            let minus_row = catmull_rom_sample(&epsf.view(), k_u + oversample * h, k_v);
            let fd_row = (plus_row - minus_row) / (2.0 * h);
            let plus_col = catmull_rom_sample(&epsf.view(), k_u, k_v - oversample * h);
            let minus_col = catmull_rom_sample(&epsf.view(), k_u, k_v + oversample * h);
            let fd_col = (plus_col - minus_col) / (2.0 * h);
            assert!(
                (ds_drow - fd_row).abs() < 1e-4 * fd_row.abs().max(1.0),
                "dS/ddelta_row {ds_drow} vs fd {fd_row}"
            );
            assert!(
                (ds_dcol - fd_col).abs() < 1e-4 * fd_col.abs().max(1.0),
                "dS/ddelta_col {ds_dcol} vs fd {fd_col}"
            );
        }
    }

    #[test]
    fn delta_sign_round_trip_build_stamp_render_refine() {
        // build_stamp delta -> render synthesizes the stamp -> refine
        // recovers it. A flipped GN-delta sign would diverge / leave
        // the basin (decision 4 / 12 sign convention).
        use crate::image::stamp::{
            DEFAULT_CENTROID_MAX_ITER, DEFAULT_CENTROID_TOL, DEFAULT_WEIGHT_FWHM,
            build_stamp,
        };
        let true_row = 7.35_f64;
        let true_col = 6.80_f64;
        let amplitude = 100.0;
        let sigma_det = 1.6;
        let background = 5.0;
        let cutout_size = 15;
        let mut cutout = Array2::<f64>::from_elem((cutout_size, cutout_size), background);
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

        let oversample = 5;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * sigma_det, amplitude);
        let true_flux = 2.0;
        let data = stack_one(&synth_stamp(
            &epsf,
            oversample,
            stamp_result.delta.0,
            stamp_result.delta.1,
            true_flux,
            background,
        ));
        // Start a touch off the build_stamp delta; a sign flip would
        // push refinement away rather than back to it.
        let delta_init = arr2(&[[
            stamp_result.delta.0 + 0.05,
            stamp_result.delta.1 - 0.05,
        ]]);
        let result = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta_init.view(),
            50,
            1e-9,
        )
        .unwrap();
        assert!(result.ok[0]);
        assert!(
            (result.delta[(0, 0)] - stamp_result.delta.0).abs() < 1e-3,
            "delta_row {} vs build_stamp {}",
            result.delta[(0, 0)],
            stamp_result.delta.0
        );
        assert!(
            (result.delta[(0, 1)] - stamp_result.delta.1).abs() < 1e-3,
            "delta_col {} vs build_stamp {}",
            result.delta[(0, 1)],
            stamp_result.delta.1
        );
        assert!((result.flux[0] - true_flux).abs() < 1e-2 * true_flux);
        assert!((result.background[0] - background).abs() < 1e-2 * background);
    }

    // --- per-star failure sentinels. ---

    #[test]
    fn too_few_valid_pixels_is_sentinel() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.4, 50.0);
        // All-NaN data -> zero valid pixels.
        let data = Array3::<f64>::from_elem((1, stamp_size, stamp_size), f64::NAN);
        let delta = arr2(&[[0.1, 0.1]]);
        let leaf = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap();
        assert!(!leaf.ok[0]);
        assert!(leaf.flux[0].is_nan());
        assert!(leaf.background[0].is_nan());

        let refined = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
            10,
            1e-8,
        )
        .unwrap();
        assert!(!refined.ok[0]);
        assert!(!refined.converged[0]);
        assert!(refined.flux[0].is_nan());
        assert!(refined.background[0].is_nan());
        // Atomic: delta resets to delta_init.
        assert_eq!(refined.delta[(0, 0)], 0.1);
        assert_eq!(refined.delta[(0, 1)], 0.1);
    }

    #[test]
    fn singular_2x2_constant_model_is_sentinel() {
        // A uniform epsf samples to a constant g over the stamp -> the
        // (f, b) 2x2 cannot separate flux from background -> sentinel.
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = Array2::<f64>::from_elem((epsf_side, epsf_side), 1.0);
        let data = Array3::<f64>::from_elem((1, stamp_size, stamp_size), 4.0);
        let delta = arr2(&[[0.05, -0.05]]);
        let leaf = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap();
        assert!(!leaf.ok[0]);
        assert!(leaf.flux[0].is_nan());

        let refined = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
            5,
            1e-8,
        )
        .unwrap();
        assert!(!refined.ok[0]);
        assert!(refined.flux[0].is_nan());
        assert_eq!(refined.delta[(0, 0)], 0.05);
        assert_eq!(refined.delta[(0, 1)], -0.05);
    }

    // --- batch shapes. ---

    #[test]
    fn single_star_batch() {
        let oversample = 3;
        let stamp_size = 5;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.0, 40.0);
        let data = stack_one(&synth_stamp(&epsf, oversample, 0.1, -0.1, 1.5, 0.3));
        let delta = arr2(&[[0.1, -0.1]]);
        let got = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap();
        assert_eq!(got.flux.len(), 1);
        assert!(got.ok[0]);
        assert!((got.flux[0] - 1.5).abs() < 1e-6);
        assert!((got.background[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn empty_batch_yields_empty_outputs() {
        let oversample = 5;
        let epsf = Array2::<f64>::zeros((35, 35));
        let data = Array3::<f64>::zeros((0, 7, 7));
        let delta = Array2::<f64>::zeros((0, 2));
        let leaf = solve_flux_background(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap();
        assert_eq!(leaf.flux.len(), 0);
        assert_eq!(leaf.ok.len(), 0);
        let refined = refine_nuisance(
            epsf.view(),
            oversample,
            data.view(),
            None,
            delta.view(),
            5,
            1e-8,
        )
        .unwrap();
        assert_eq!(refined.flux.len(), 0);
        assert_eq!(refined.delta.shape(), &[0, 2]);
        assert_eq!(refined.converged.len(), 0);
    }

    #[test]
    fn f32_and_f64_dual_path_agree() {
        let oversample = 5;
        let stamp_size = 7;
        let epsf_side = oversample * stamp_size;
        let epsf = gaussian_epsf(epsf_side, oversample as f64 * 1.4, 60.0);
        let stamp = synth_stamp(&epsf, oversample, 0.18, -0.12, 3.3, 1.7);
        let data_f64 = stack_one(&stamp);
        let data_f32: Array3<f32> = data_f64.mapv(|v| v as f32);
        let delta = arr2(&[[0.18, -0.12]]);

        let from_f64 = solve_flux_background(
            epsf.view(),
            oversample,
            data_f64.view(),
            None,
            delta.view(),
        )
        .unwrap();
        let from_f32 = solve_flux_background(
            epsf.view(),
            oversample,
            data_f32.view(),
            None,
            delta.view(),
        )
        .unwrap();
        assert_eq!(from_f64.ok, from_f32.ok);
        assert!(
            (from_f64.flux[0] - from_f32.flux[0]).abs()
                < 1e-3 * from_f64.flux[0].abs().max(1.0)
        );
        assert!(
            (from_f64.background[0] - from_f32.background[0]).abs()
                < 1e-3 * from_f64.background[0].abs().max(1.0)
        );

        let refined_f32 = refine_nuisance(
            epsf.view(),
            oversample,
            data_f32.view(),
            None,
            delta.view(),
            40,
            1e-8,
        )
        .unwrap();
        assert!(refined_f32.ok[0]);
        assert!((refined_f32.flux[0] - 3.3).abs() < 1e-2 * 3.3);
    }

    // --- hard preconditions. ---

    #[test]
    fn error_epsf_not_square() {
        let epsf = Array2::<f64>::zeros((10, 12));
        let data = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        let err = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(err, NuisanceError::EpsfNotSquare { rows: 10, cols: 12 });
    }

    #[test]
    fn error_oversample_not_odd() {
        let epsf = Array2::<f64>::zeros((20, 20));
        let data = Array3::<f64>::zeros((1, 5, 5));
        let delta = arr2(&[[0.0, 0.0]]);
        let err = refine_nuisance(
            epsf.view(),
            4,
            data.view(),
            None,
            delta.view(),
            5,
            1e-8,
        )
        .unwrap_err();
        assert_eq!(err, NuisanceError::OversampleNotOdd { oversample: 4 });
    }

    #[test]
    fn error_epsf_size_not_multiple() {
        let epsf = Array2::<f64>::zeros((34, 34));
        let data = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        let err = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NuisanceError::EpsfSizeNotMultiple {
                epsf_side: 34,
                oversample: 5,
            }
        );
    }

    #[test]
    fn error_derived_stamp_size_even() {
        // os = 3, epsf_side = 18 -> stamp_size = 6 (even).
        let epsf = Array2::<f64>::zeros((18, 18));
        let data = Array3::<f64>::zeros((1, 6, 6));
        let delta = arr2(&[[0.0, 0.0]]);
        let err = solve_flux_background(
            epsf.view(),
            3,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(err, NuisanceError::DerivedStampSizeEven { stamp_size: 6 });
    }

    #[test]
    fn error_batch_length_mismatch_data_shape() {
        // Derived stamp_size = 7 but data stamps are 5x5.
        let epsf = Array2::<f64>::zeros((35, 35));
        let data = Array3::<f64>::zeros((2, 5, 5));
        let delta = arr2(&[[0.0, 0.0], [0.1, 0.1]]);
        let err = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NuisanceError::BatchLengthMismatch {
                data: (2, 5, 5),
                delta: (2, 2),
            }
        );
    }

    #[test]
    fn error_batch_length_mismatch_delta_shape() {
        // data is (2, 7, 7) but delta is (2, 3): not (N, 2).
        let epsf = Array2::<f64>::zeros((35, 35));
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = arr2(&[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]);
        let err = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NuisanceError::BatchLengthMismatch {
                data: (2, 7, 7),
                delta: (2, 3),
            }
        );
    }

    #[test]
    fn error_weight_shape_mismatch() {
        let epsf = Array2::<f64>::zeros((35, 35));
        let data = Array3::<f64>::zeros((2, 7, 7));
        let weight = Array3::<f64>::zeros((2, 7, 8));
        let delta = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        let err = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            Some(weight.view()),
            delta.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            NuisanceError::WeightShapeMismatch {
                weight: (2, 7, 8),
                data: (2, 7, 7),
            }
        );
    }

    #[test]
    fn error_refine_params_invalid_only_refine() {
        let epsf = Array2::<f64>::zeros((35, 35));
        let data = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        // max_iter == 0.
        let err = refine_nuisance(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
            0,
            1e-8,
        )
        .unwrap_err();
        assert_eq!(
            err,
            NuisanceError::RefineParamsInvalid {
                max_iter: 0,
                tol: 1e-8,
            }
        );
        // Bad tol values.
        for bad_tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = refine_nuisance(
                epsf.view(),
                5,
                data.view(),
                None,
                delta.view(),
                5,
                bad_tol,
            )
            .unwrap_err();
            match err {
                NuisanceError::RefineParamsInvalid { max_iter, tol } => {
                    assert_eq!(max_iter, 5);
                    assert!(tol.to_bits() == bad_tol.to_bits() || tol == bad_tol);
                }
                other => panic!("expected RefineParamsInvalid, got {other:?}"),
            }
        }
        // solve_flux_background has no max_iter/tol, so the same shapes
        // succeed there (it is exempt from RefineParamsInvalid).
        let ok = solve_flux_background(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn shared_precondition_checked_before_refine_params() {
        // Both a shared shape error and bad refine params: the shared
        // set is checked first (declaration order).
        let epsf = Array2::<f64>::zeros((10, 12)); // not square
        let data = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        let err = refine_nuisance(
            epsf.view(),
            5,
            data.view(),
            None,
            delta.view(),
            0,
            -1.0,
        )
        .unwrap_err();
        assert_eq!(err, NuisanceError::EpsfNotSquare { rows: 10, cols: 12 });
    }
}

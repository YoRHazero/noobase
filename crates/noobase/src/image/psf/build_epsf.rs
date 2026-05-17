//! Core super-resolution iteration driver: a caller-assembled stamp
//! stack -> an oversampled effective PSF plus a refined star table.
//!
//! This is the Phase 6 *orchestrator*. It owns no operator of its own:
//! every forward / adjoint / centroid-Jacobian / 2x2 solve is delegated
//! to the operator stack built in Phases 2/3/5
//! ([`super::render::render`], [`super::accumulate::accumulate`],
//! [`super::nuisance::refine_nuisance`]) and, *only* to seed `psi` when
//! the caller supplies no warm start, to the Phase 4 cross-`N`
//! combiner ([`super::robust::robust_combine`]). Nothing in this file
//! re-implements Catmull-Rom sampling, the exact transpose, the delta
//! Jacobian or the weighted normal equations: that coupling to the
//! operator stack is deliberate and is what keeps the driver a driver.
//!
//! Design notes:
//!
//! - **Generic only at the boundary ("internal f64").** `build_epsf` is
//!   generic over the detector-data float `T`, but `data`/`weight` are
//!   upcast to owned `f64` arrays on entry and the entire
//!   super-resolution outer loop runs in `f64` (`render`/`accumulate`
//!   are already `f64`; `refine_nuisance` is called as
//!   `refine_nuisance::<f64>`). `T` never threads through the loop,
//!   mirroring the `robust`/`nuisance` boundary-generic convention. The
//!   residual reweight is multiplied into an *internal copy*; the
//!   caller's arrays are never mutated.
//!
//! - **`psi` is an output, so `s` is recovered from `data`.** Unlike
//!   `render`/`nuisance` (which recover `s` from `epsf.shape()`), there
//!   is no `epsf` input here. `s` is `data.shape()[1]`; `data` is
//!   validated as an `(N, s, s)` square stamp stack with `s` odd and
//!   `oversample` odd. The model grid `(os*s, os*s)` is passed
//!   *explicitly* to `accumulate` (its `(oversample, stamp_size)`),
//!   symmetric to "accumulate is told the grid because psi is its
//!   output" -- except `stamp_size` is carried by `data`, not passed
//!   separately.
//!
//! - **Six load-bearing forks (decided, authoritative).**
//!   1. SR update law = projected Landweber / Irani-Peleg fixed point
//!      `psi <- max(0, psi + lambda * accumulate(W_eff o (D -
//!      render(psi; f, b, delta))))`, with a non-negative projection
//!      every round (decision 5; negative *data* pixels are still all
//!      kept, decision 5). `accumulate` is the Phase 3 exact adjoint =
//!      the back-projection update direction (decision 10, no autodiff).
//!   2. Convergence = `‖Δpsi‖_rel < tol` or `max_iter` (global psi
//!      fixed point; the `max_iter` + `tol` two-knob drive mirrors
//!      `build_stamp`/`refine_nuisance`).
//!   3. Residual reweight is in v1: each outer round a Huber/Tukey
//!      M-estimator factor is formed from the residual magnitude and
//!      multiplied into the effective weight `W_eff = W_base x
//!      reweight(r)`, fed to *both* `accumulate` and that round's
//!      `refine_nuisance` (decision 7: "the robust step multiplies on
//!      the weight; accumulate/nuisance do not self-compute it" -- this
//!      solver is exactly where it is computed). Symmetric,
//!      sign-agnostic (decision 5). Named [`ResidualReweight`]
//!      deliberately *without* "robust": Phase 4 `robust_combine` is a
//!      cross-`N` *stack combine* (hard-rejects whole samples, the
//!      Phase 7 wing stack); Phase 6 is per-pixel *residual
//!      reweighting* (smoothly down-weights forward-model residuals,
//!      the core SR). Two independent mechanisms; the prose verbs
//!      "combine" and "reweight" are never mixed.
//!   4. Gauge = unit volume: every outer round `psi` is normalized to
//!      `sum(psi) / os^2 = 1` and the scale is folded into `flux`
//!      (fixes the `psi <-> f` degeneracy, decision 5). The
//!      encircled-energy radius normalization (the decision-5 science
//!      semantics) is left to Phase 7.
//!   5. `psi_init = None` => internal seed = the `delta ~ 0`
//!      integer-aligned stack's robust mean (decision 4 init idea;
//!      Phase 4 `robust_combine` reused *only* as the seed cross-`N`
//!      coadd -- a completely different mechanism from fork 3's
//!      `ResidualReweight`; legal reuse here).
//!   6. Control parameters are packed in [`BuildEpsfParams`] (with an
//!      `impl Default` of sane Rust-only defaults, decision 9), not a
//!      flat parameter list.
//!
//! - **Conventions inherited, not re-decided.** `data`/`weight` are the
//!   data-path generic `T` (upcast to `f64`); `psi` is `f64`
//!   throughout. `weight` is the complete per-pixel base weight
//!   (`Option`; `None` = unit), mask polarity folded into `weight = 0`
//!   by the caller (decision 7). The `delta` coordinate/sign is
//!   inherited verbatim from decision 12 + Phase 2/5 -- the loop calls
//!   `render` and `refine_nuisance`, whose round-trip/Jacobian tests
//!   already pin the sign, so this phase does not re-fix it. A per-star
//!   `!ok` (the `nuisance` failure sentinel) is excluded from the `psi`
//!   accumulate (a bad `(f, b, delta)` must not back-project into the
//!   model). The star table reuses `NuisanceRefined` semantics
//!   (`flux`/`background`/`delta`/`ok`); `iterations`/`converged` are
//!   *global* psi-fixed-point quantities (not per-star, so not in the
//!   star table). `render`/`accumulate`/`refine_nuisance` each already
//!   parallelize over `N`; the driver only orchestrates.
//!
//! - **Errors are hard preconditions only; no `Ok(None)`** (the
//!   `render`/`accumulate`/`robust`/`nuisance` family, unlike
//!   `build_stamp`). The check order is: `oversample` odd -> recovered
//!   `s` even/non-square -> batch dims -> `weight` shape -> `psi_init`
//!   shape -> [`BuildEpsfError::ParamsInvalid`] (any invalid scalar
//!   parameter, a single disambiguating variant mirroring
//!   `RefineParamsInvalid`/`ClippedMeanInvalidParams`). `N = 0` is
//!   legal (no stars -> the seed/`psi_init`/zero `psi`, an empty star
//!   table, `converged = true` trivially).
//!
//! Landing decisions (the Phase-6 "implementation details decided
//! during development", recorded here as Phase 5 recorded
//! `MIN_VALID_PIXELS` &c.):
//!
//! - **Step `lambda` is operator-norm adaptive.** `params.step` is a
//!   dimensionless Landweber relaxation; the raw update is divided by a
//!   power-iteration estimate of `‖A^T W A‖` (recomputed each outer
//!   round, since `flux`/`delta` drift), so the default `step = 1.0`
//!   converges regardless of the flux scale. If the operator norm is
//!   non-finite or zero (e.g. every star `!ok`), the step is taken as
//!   zero (`psi` unchanged) and the loop converges trivially.
//! - **Residual reweight scale.** The robust sigma is the per-pixel
//!   `1/sqrt(W_base)` when a base weight is given, else a global
//!   `1.4826 * MAD` of the finite residuals; the standardized residual
//!   is `z = r / sigma`; `c` is in those sigma units. A non-finite
//!   residual gets factor `1` (cannot judge outlierness; validity is
//!   handled elsewhere). Huber: `1` for `|z| <= c`, else `c / |z|`.
//!   Tukey biweight: `(1 - (z/c)^2)^2` for `|z| <= c`, else `0`.
//! - **Seed.** Per-stamp background = robust median of the border
//!   ring; per-stamp flux = sum of the background-subtracted finite
//!   pixels (a star with non-positive/non-finite flux is dropped from
//!   the seed coadd). The normalized stamps are coadded across `N` by
//!   `robust_combine` with `ClippedMean` (the explicit "robust_combine
//!   only for the seed" reuse; outlier-resistant). The native
//!   `(s, s)` coadd is lifted to `(os*s, os*s)` by `os x os` nearest
//!   block replication, then unit-volume gauged. A degenerate (all
//!   non-finite / non-positive) seed falls back to a flat positive
//!   `psi`.
//! - **`!ok` handling.** A `!ok` star is excluded from the `psi`
//!   accumulate by zeroing its weight stamp (its residual stamp is
//!   also zeroed so a sentinel `NaN` cannot reach the scatter). It is
//!   *not* excluded from `refine_nuisance` (it is given a fresh attempt
//!   against the improved `psi` each round), so `ok` can recover across
//!   rounds (`refine_nuisance` recomputes it from scratch).
//! - **Gauge position.** Unit-volume gauge is applied after the
//!   non-negative projection and before that round's
//!   `refine_nuisance`, so `refine_nuisance` sees the gauged model and
//!   `flux` consistently absorbs the scale.
//! - **`‖Δpsi‖_rel`** is the Euclidean (Frobenius) norm of
//!   `psi_new - psi` over the relative `‖psi‖`; chi-square history is
//!   *not* tracked (the trunk `BuildEpsf` stays
//!   `iterations: usize, converged: bool`; richer diagnostics are
//!   deferred). `converged = ‖Δpsi‖_rel < tol && iter + 1 < max_iter`
//!   (reached strictly before the budget was exhausted; mirrors
//!   `refine_nuisance`).
//! - **v1 regularization is the non-negative projection only.** No
//!   explicit Tikhonov / smoothness prior (decision 10: a caller who
//!   wants a learnable prior takes the adjoint and wraps autograd;
//!   noobase neither forces nor blocks it -- no knobs for hypothetical
//!   needs).

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use thiserror::Error;

use crate::float::Float;

use super::accumulate::accumulate;
use super::nuisance::{refine_nuisance, solve_flux_background};
use super::render::render;
use super::robust::{CombineMethod, robust_combine};

/// Default outer super-resolution iteration cap.
pub const DEFAULT_MAX_ITER: usize = 50;
/// Default `‖Δpsi‖_rel` convergence threshold.
pub const DEFAULT_TOL: f64 = 1e-4;
/// Default Landweber relaxation (operator-norm adaptive, so `1.0` is a
/// safe full step regardless of the flux scale).
pub const DEFAULT_STEP: f64 = 1.0;
/// Default per-round `refine_nuisance` inner-iteration cap.
pub const DEFAULT_NUISANCE_MAX_ITER: usize = 3;
/// Default per-round `refine_nuisance` `‖Δdelta‖` threshold.
pub const DEFAULT_NUISANCE_TOL: f64 = 1e-4;

/// Power-iteration count for the `‖A^T W A‖` estimate (a scale, not a
/// precise eigenvalue, so a small fixed budget is enough).
const POWER_ITERATIONS: usize = 12;
/// Clip threshold (robust sigmas) for the seed `robust_combine`.
const SEED_CLIP_KAPPA: f64 = 3.0;
/// Clip-iteration cap for the seed `robust_combine`.
const SEED_CLIP_MAX_ITER: usize = 5;
/// `1 / 0.6745`: scales a median-absolute-deviation to a Gaussian sigma.
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

/// Per-pixel residual reweighting applied each outer round (decision 7
/// / fork 3). Deliberately *not* named "robust": this is the core-SR
/// per-pixel down-weighting, distinct from Phase 4 `robust_combine`'s
/// cross-`N` stack combine. Clipping is intentionally absent -- the
/// core SR down-weights smoothly to avoid a hard-cut bias; hard
/// rejection stays in the Phase 4 / Phase 7 wing stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualReweight {
    /// Pure inverse-variance: no reweighting.
    None,
    /// Huber M-estimator; `c` is the transition in robust sigmas.
    Huber { c: f64 },
    /// Tukey biweight; `c` is the rejection knee in robust sigmas.
    Tukey { c: f64 },
}

/// Packed control parameters for [`build_epsf`] (fork 6). `impl
/// Default` gives Rust-only callers a sane configuration (decision 9).
#[derive(Debug, Clone, PartialEq)]
pub struct BuildEpsfParams {
    /// Outer SR iteration cap (must be `> 0`).
    pub max_iter: usize,
    /// `‖Δpsi‖_rel < tol` convergence threshold (finite, `> 0`).
    pub tol: f64,
    /// Landweber relaxation (operator-norm adaptive; finite, `> 0`).
    pub step: f64,
    /// Per-pixel residual reweighting (fork 3).
    pub residual_reweight: ResidualReweight,
    /// Per-round `refine_nuisance` inner-iteration cap (must be `> 0`;
    /// `1` = one block-coordinate step).
    pub nuisance_max_iter: usize,
    /// Per-round `refine_nuisance` `‖Δdelta‖` threshold (finite, `> 0`).
    pub nuisance_tol: f64,
}

impl Default for BuildEpsfParams {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            tol: DEFAULT_TOL,
            step: DEFAULT_STEP,
            residual_reweight: ResidualReweight::None,
            nuisance_max_iter: DEFAULT_NUISANCE_MAX_ITER,
            nuisance_tol: DEFAULT_NUISANCE_TOL,
        }
    }
}

/// Output of [`build_epsf`]: the solved oversampled effective PSF plus
/// the refined per-star table and the global fixed-point flags.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildEpsf {
    /// `(os*s, os*s)` solved oversampled effective PSF, unit-volume
    /// gauged (`sum(epsf) / os^2 = 1`; fork 4).
    pub epsf: Array2<f64>,
    /// `(N,)` refined per-star flux (`NuisanceRefined` semantics;
    /// `NaN` when `!ok`).
    pub flux: Array1<f64>,
    /// `(N,)` refined per-star scalar background (`NaN` when `!ok`).
    pub background: Array1<f64>,
    /// `(N, 2)` refined per-star centroid `(delta_row, delta_col)`.
    pub delta: Array2<f64>,
    /// `(N,)` per-star usability (the `nuisance` `ok`); a `!ok` star
    /// does not enter the `psi` accumulate.
    pub ok: Array1<bool>,
    /// Number of outer SR iterations actually run.
    pub iterations: usize,
    /// Global convergence: `‖Δpsi‖_rel < tol` was reached strictly
    /// before `max_iter` (distinct from the per-star `nuisance`
    /// `converged`).
    pub converged: bool,
}

/// Errors returned by [`build_epsf`] for ill-shaped / ill-parameterized
/// inputs.
///
/// Every variant is a hard precondition. Like the
/// `render`/`accumulate`/`robust`/`nuisance` family (and unlike
/// `build_stamp`), there is no algorithmic skip path: a well-formed
/// call always yields a [`BuildEpsf`] (a star that cannot be solved is
/// handled per-star with the `nuisance` sentinel and `ok = false`, not
/// an `Err`). `ParamsInvalid` is checked after the shape preconditions.
#[derive(Debug, Error, PartialEq)]
pub enum BuildEpsfError {
    #[error("oversample must be odd; got {oversample}")]
    OversampleNotOdd { oversample: usize },
    #[error(
        "recovered stamp_size (data.shape()[1]) must be odd; got {stamp_size}"
    )]
    StampSizeEven { stamp_size: usize },
    #[error(
        "batch dimensions disagree: data shape {data:?} must be (N, s, s) with s odd and delta_init shape {delta_init:?} must be (N, 2)"
    )]
    BatchLengthMismatch {
        data: (usize, usize, usize),
        delta_init: (usize, usize),
    },
    #[error("weight shape {weight:?} must equal data shape {data:?}")]
    WeightShapeMismatch {
        weight: (usize, usize, usize),
        data: (usize, usize, usize),
    },
    #[error(
        "psi_init shape {psi_init:?} must equal the model grid {expected:?}"
    )]
    PsiInitShapeMismatch {
        psi_init: (usize, usize),
        expected: (usize, usize),
    },
    #[error(
        "build_epsf requires max_iter > 0, tol/step/nuisance_tol finite and > 0, nuisance_max_iter > 0, and any ResidualReweight c finite and > 0; got max_iter = {max_iter}, tol = {tol}, step = {step}"
    )]
    ParamsInvalid { max_iter: usize, tol: f64, step: f64 },
}

/// Solve the core oversampled effective PSF from a caller-assembled
/// stamp stack by projected-Landweber super-resolution, jointly
/// refining the per-star nuisance table.
///
/// See the module-level documentation for the operator-stack
/// composition, the six forks, the `delta` sign inheritance, and the
/// landing decisions.
///
/// # Parameters
///
/// - `data`: `(N, s, s)` caller-assembled stamp stack (from
///   `build_stamp`, scheme C). May be `f32` or `f64`; upcast to `f64`
///   on entry.
/// - `weight`: optional `(N, s, s)` complete per-pixel *base* weight
///   (`valid * 1/sigma^2 * ...`, formed by the caller); `None` is unit
///   weight (decision 7). The residual reweight multiplies an internal
///   copy; this array is never mutated.
/// - `delta_init`: `(N, 2)` per-stamp `delta` initial value (from
///   `build_stamp`; in-basin, decision 4).
/// - `oversample`: `os`, forced odd (decisions 1/12). `s` is recovered
///   from the `data` stamp dimension.
/// - `psi_init`: optional `(os*s, os*s)` warm-start `psi`; `None`
///   triggers the internal seed (fork 5).
/// - `params`: see [`BuildEpsfParams`].
///
/// # Returns
///
/// `Ok(BuildEpsf)` with the unit-volume-gauged `(os*s, os*s)` `epsf`,
/// the `(N,)` `flux`/`background`/`ok`, the `(N, 2)` `delta`, and the
/// global `iterations`/`converged`. `N = 0` yields the
/// seed/`psi_init`/zero `psi`, empty star arrays, `iterations = 0`,
/// `converged = true`.
///
/// # Errors
///
/// Returns a [`BuildEpsfError`] if `oversample` is not odd, if the
/// recovered `s` is even, if `data` is not an `(N, s, s)` square stack
/// or `delta_init` is not `(N, 2)`, if `weight` is `Some` with a shape
/// different from `data`, if `psi_init` is `Some` with a shape other
/// than `(os*s, os*s)`, or if any scalar parameter is invalid.
pub fn build_epsf<T: Float>(
    data: ArrayView3<T>,
    weight: Option<ArrayView3<T>>,
    delta_init: ArrayView2<f64>,
    oversample: usize,
    psi_init: Option<ArrayView2<f64>>,
    params: BuildEpsfParams,
) -> Result<BuildEpsf, BuildEpsfError> {
    // --- Hard preconditions (returned as Err; no Ok(None) path). The
    // check order is the documented order. ---
    if oversample % 2 == 0 {
        // Also rejects oversample == 0 (0 is even, hence not odd).
        return Err(BuildEpsfError::OversampleNotOdd { oversample });
    }
    let batch_size = data.shape()[0];
    let stamp_size = data.shape()[1];
    if stamp_size % 2 == 0 {
        // Catches the degenerate empty stamp (stamp_size == 0) too.
        return Err(BuildEpsfError::StampSizeEven { stamp_size });
    }
    if data.shape()[2] != stamp_size
        || delta_init.shape()[0] != batch_size
        || delta_init.shape()[1] != 2
    {
        return Err(BuildEpsfError::BatchLengthMismatch {
            data: (batch_size, data.shape()[1], data.shape()[2]),
            delta_init: (delta_init.shape()[0], delta_init.shape()[1]),
        });
    }
    if let Some(weight_view) = weight.as_ref() {
        let weight_shape = weight_view.shape();
        if weight_shape[0] != batch_size
            || weight_shape[1] != stamp_size
            || weight_shape[2] != stamp_size
        {
            return Err(BuildEpsfError::WeightShapeMismatch {
                weight: (weight_shape[0], weight_shape[1], weight_shape[2]),
                data: (batch_size, stamp_size, stamp_size),
            });
        }
    }
    let side = oversample * stamp_size;
    if let Some(psi_view) = psi_init.as_ref() {
        let psi_shape = psi_view.shape();
        if psi_shape[0] != side || psi_shape[1] != side {
            return Err(BuildEpsfError::PsiInitShapeMismatch {
                psi_init: (psi_shape[0], psi_shape[1]),
                expected: (side, side),
            });
        }
    }
    let reweight_c_invalid = match params.residual_reweight {
        ResidualReweight::None => false,
        ResidualReweight::Huber { c } | ResidualReweight::Tukey { c } => {
            !(c.is_finite() && c > 0.0)
        }
    };
    let params_invalid = params.max_iter == 0
        || !(params.tol.is_finite() && params.tol > 0.0)
        || !(params.step.is_finite() && params.step > 0.0)
        || params.nuisance_max_iter == 0
        || !(params.nuisance_tol.is_finite() && params.nuisance_tol > 0.0)
        || reweight_c_invalid;
    if params_invalid {
        return Err(BuildEpsfError::ParamsInvalid {
            max_iter: params.max_iter,
            tol: params.tol,
            step: params.step,
        });
    }

    // --- Internal f64: upcast once at the boundary; T never threads
    // through the loop (mirrors robust/nuisance). ---
    let data_f: Array3<f64> =
        data.mapv(|value| value.to_f64().unwrap_or(f64::NAN));
    let weight_base: Option<Array3<f64>> = weight
        .as_ref()
        .map(|w| w.mapv(|value| value.to_f64().unwrap_or(f64::NAN)));

    // Seed psi (fork 5) or take the caller's warm start.
    let mut psi: Array2<f64> = match psi_init.as_ref() {
        Some(view) => view.to_owned(),
        None => seed_psi(&data_f, oversample, stamp_size, side),
    };

    // N = 0 is legal: nothing to solve. Gauge whatever psi we have and
    // return empty per-star arrays (converged trivially).
    if batch_size == 0 {
        gauge_unit_volume(&mut psi, &mut Array1::<f64>::zeros(0), oversample);
        return Ok(BuildEpsf {
            epsf: psi,
            flux: Array1::<f64>::zeros(0),
            background: Array1::<f64>::zeros(0),
            delta: Array2::<f64>::zeros((0, 2)),
            ok: Array1::from_vec(Vec::<bool>::new()),
            iterations: 0,
            converged: true,
        });
    }

    let mut delta: Array2<f64> = delta_init.to_owned();
    let zeros_background = Array1::<f64>::zeros(batch_size);

    // Pre-loop: a consistent (f, b) for psi_0 at the *trusted*
    // build_stamp delta_init via the Phase 5 solve_flux_background
    // leaf -- no delta Gauss-Newton against the crude seed (a
    // premature GN step on a blocky model can push a star out of
    // basin and never recover). delta refinement begins only once a
    // Landweber step has sharpened psi.
    gauge_unit_volume(&mut psi, &mut Array1::<f64>::zeros(0), oversample);
    let initial = solve_flux_background::<f64>(
        psi.view(),
        oversample,
        data_f.view(),
        weight_base.as_ref().map(|w| w.view()),
        delta.view(),
    )
    .expect(
        "solve_flux_background preconditions are guaranteed by build_epsf validation",
    );
    let mut flux = initial.flux;
    let mut background = initial.background;
    let mut ok = initial.ok;

    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..params.max_iter {
        let flux_safe = sanitize_1d(&flux);
        let background_safe = sanitize_1d(&background);
        let delta_safe = sanitize_2d(&delta);

        // Model and residual at the current (psi, f, b, delta).
        let model = render(
            psi.view(),
            oversample,
            delta_safe.view(),
            flux_safe.view(),
            background_safe.view(),
        )
        .expect("render preconditions are guaranteed by build_epsf validation");
        let mut residual = Array3::<f64>::zeros((batch_size, stamp_size, stamp_size));
        for n in 0..batch_size {
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    residual[(n, i, j)] =
                        data_f[(n, i, j)] - model[(n, i, j)];
                }
            }
        }

        // Per-pixel reweight factor (fork 3): sign-agnostic, formed
        // from the residual magnitude in robust-sigma units.
        let factor = reweight_factor(
            &residual,
            weight_base.as_ref(),
            &ok,
            params.residual_reweight,
        );

        // W_eff for refine_nuisance: base x factor (all stars get a
        // fresh attempt; not !ok-masked, so ok can recover).
        let weight_refine: Option<Array3<f64>> =
            effective_weight(weight_base.as_ref(), &factor, params.residual_reweight);

        // W_acc / R_acc for the psi accumulate: !ok stars zeroed, every
        // non-finite weight/residual zeroed so the scatter stays finite.
        let mut weight_acc =
            Array3::<f64>::zeros((batch_size, stamp_size, stamp_size));
        let mut residual_acc =
            Array3::<f64>::zeros((batch_size, stamp_size, stamp_size));
        for n in 0..batch_size {
            let star_ok = ok[n];
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    let r = residual[(n, i, j)];
                    let w = match &weight_refine {
                        Some(w_eff) => w_eff[(n, i, j)],
                        None => 1.0,
                    };
                    if star_ok && r.is_finite() && w.is_finite() && w > 0.0 {
                        weight_acc[(n, i, j)] = w;
                        residual_acc[(n, i, j)] = r;
                    }
                }
            }
        }

        // Operator-norm-adaptive Landweber step.
        let operator_norm = power_iteration_norm(
            oversample,
            stamp_size,
            &delta_safe,
            &flux_safe,
            &zeros_background,
            &weight_acc,
            side,
        );
        let update = accumulate(
            residual_acc.view(),
            Some(weight_acc.view()),
            oversample,
            stamp_size,
            delta_safe.view(),
            flux_safe.view(),
        )
        .expect(
            "accumulate preconditions are guaranteed by build_epsf validation",
        );
        let lambda = if operator_norm.is_finite() && operator_norm > 0.0 {
            params.step / operator_norm
        } else {
            0.0
        };

        // psi <- max(0, psi + lambda * update); non-negative projection
        // (decision 5). Negative *data* pixels were already kept.
        let mut psi_next = Array2::<f64>::zeros((side, side));
        for p in 0..side {
            for q in 0..side {
                let value = psi[(p, q)] + lambda * update[(p, q)];
                psi_next[(p, q)] = if value > 0.0 { value } else { 0.0 };
            }
        }

        // Unit-volume gauge (fork 4): after the projection, before this
        // round's refine, so refine sees the gauged model.
        gauge_unit_volume(&mut psi_next, &mut flux, oversample);

        let relative_change = relative_l2_change(&psi, &psi_next);
        psi = psi_next;
        iterations = iter + 1;

        // refine_nuisance for the next round, with the same W_eff
        // (decision 7 / fork 3: fed to both accumulate and refine).
        run_refine(
            &psi,
            oversample,
            &data_f,
            weight_refine.as_ref(),
            &delta.clone(),
            params.nuisance_max_iter,
            params.nuisance_tol,
            &mut flux,
            &mut background,
            &mut delta,
            &mut ok,
        );

        if relative_change < params.tol {
            converged = iter + 1 < params.max_iter;
            break;
        }
    }

    Ok(BuildEpsf {
        epsf: psi,
        flux,
        background,
        delta,
        ok,
        iterations,
        converged,
    })
}

/// Replace non-finite entries of a `(N,)` array with `0.0` so the
/// operator calls cannot see a sentinel `NaN` (the `!ok` stars are
/// excluded from the scatter by a zero weight).
fn sanitize_1d(values: &Array1<f64>) -> Array1<f64> {
    values.mapv(|x| if x.is_finite() { x } else { 0.0 })
}

/// Replace non-finite rows' entries of a `(N, 2)` array with `0.0`.
fn sanitize_2d(values: &Array2<f64>) -> Array2<f64> {
    values.mapv(|x| if x.is_finite() { x } else { 0.0 })
}

/// Euclidean (Frobenius) `‖psi_new - psi‖ / ‖psi‖`, guarded against a
/// zero denominator.
fn relative_l2_change(psi: &Array2<f64>, psi_next: &Array2<f64>) -> f64 {
    let mut diff_sq = 0.0;
    let mut base_sq = 0.0;
    for (a, b) in psi.iter().zip(psi_next.iter()) {
        let d = b - a;
        diff_sq += d * d;
        base_sq += a * a;
    }
    let base = base_sq.sqrt();
    if base > 1e-300 {
        diff_sq.sqrt() / base
    } else {
        diff_sq.sqrt()
    }
}

/// In-place unit-volume gauge (fork 4): scale `psi` so
/// `sum(psi) / os^2 = 1` and fold the reciprocal scale into `flux`
/// (the `flux <-> psi` product is invariant). A non-finite /
/// non-positive volume is left untouched (degenerate psi).
fn gauge_unit_volume(
    psi: &mut Array2<f64>,
    flux: &mut Array1<f64>,
    oversample: usize,
) {
    let volume = psi.sum() / (oversample * oversample) as f64;
    if volume.is_finite() && volume > 0.0 {
        psi.mapv_inplace(|x| x / volume);
        flux.mapv_inplace(|f| f * volume);
    }
}

/// Robust per-pixel reweight factor (fork 3). `factor[n,i,j]` is in
/// `[0, 1]`; a non-finite residual or a missing scale yields `1.0`
/// (cannot judge outlierness). Sign-agnostic: depends only on `|z|`.
fn reweight_factor(
    residual: &Array3<f64>,
    weight_base: Option<&Array3<f64>>,
    ok: &Array1<bool>,
    method: ResidualReweight,
) -> Array3<f64> {
    let shape = residual.dim();
    if let ResidualReweight::None = method {
        return Array3::<f64>::ones(shape);
    }
    // Global MAD scale used only when no base weight is supplied.
    let global_sigma = if weight_base.is_none() {
        global_mad_sigma(residual, ok)
    } else {
        f64::NAN
    };
    let mut factor = Array3::<f64>::ones(shape);
    let (batch, height, width) = shape;
    for n in 0..batch {
        if !ok[n] {
            continue;
        }
        for i in 0..height {
            for j in 0..width {
                let r = residual[(n, i, j)];
                if !r.is_finite() {
                    continue;
                }
                let sigma = match weight_base {
                    Some(w) => {
                        let wv = w[(n, i, j)];
                        if wv.is_finite() && wv > 0.0 {
                            (1.0 / wv).sqrt()
                        } else {
                            continue;
                        }
                    }
                    None => global_sigma,
                };
                if !(sigma.is_finite() && sigma > 0.0) {
                    continue;
                }
                let z = (r / sigma).abs();
                factor[(n, i, j)] = match method {
                    ResidualReweight::None => 1.0,
                    ResidualReweight::Huber { c } => {
                        if z <= c { 1.0 } else { c / z }
                    }
                    ResidualReweight::Tukey { c } => {
                        if z <= c {
                            let t = 1.0 - (z / c) * (z / c);
                            t * t
                        } else {
                            0.0
                        }
                    }
                };
            }
        }
    }
    factor
}

/// `1.4826 * median(|r - median(r)|)` over the finite residuals of the
/// `ok` stars (the no-base-weight global scale).
fn global_mad_sigma(residual: &Array3<f64>, ok: &Array1<bool>) -> f64 {
    let (batch, height, width) = residual.dim();
    let mut samples: Vec<f64> = Vec::new();
    for n in 0..batch {
        if !ok[n] {
            continue;
        }
        for i in 0..height {
            for j in 0..width {
                let r = residual[(n, i, j)];
                if r.is_finite() {
                    samples.push(r);
                }
            }
        }
    }
    if samples.is_empty() {
        return f64::NAN;
    }
    let center = median(&mut samples.clone());
    let mut deviations: Vec<f64> =
        samples.iter().map(|x| (x - center).abs()).collect();
    let mad = median(&mut deviations);
    MAD_TO_SIGMA * mad
}

/// Median of a slice (sorts in place; NaN-free by construction here).
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    }
}

/// `W_eff = W_base x factor` (fork 3). `None` only when there is no
/// base weight *and* no reweighting (true unit weight).
fn effective_weight(
    weight_base: Option<&Array3<f64>>,
    factor: &Array3<f64>,
    method: ResidualReweight,
) -> Option<Array3<f64>> {
    match (weight_base, method) {
        (None, ResidualReweight::None) => None,
        (None, _) => Some(factor.clone()),
        (Some(base), ResidualReweight::None) => Some(base.clone()),
        (Some(base), _) => {
            let mut w = base.clone();
            for (we, fa) in w.iter_mut().zip(factor.iter()) {
                *we *= *fa;
            }
            Some(w)
        }
    }
}

/// Power-iteration estimate of `‖A^T W A‖` at the current
/// `(psi-grid, flux, delta, W_acc)`, used to make `params.step` a
/// dimensionless relaxation. `A psi = render(psi; f, 0, delta)`;
/// `A^T W y = accumulate(y, W, ..., f)`. Returns `0.0` (=> zero step)
/// when the operator is degenerate.
#[allow(clippy::too_many_arguments)]
fn power_iteration_norm(
    oversample: usize,
    stamp_size: usize,
    delta_safe: &Array2<f64>,
    flux_safe: &Array1<f64>,
    zeros_background: &Array1<f64>,
    weight_acc: &Array3<f64>,
    side: usize,
) -> f64 {
    let mut probe = Array2::<f64>::ones((side, side));
    let mut estimate = 0.0;
    for _ in 0..POWER_ITERATIONS {
        let probe_norm = l2_norm(probe.iter());
        if !(probe_norm.is_finite() && probe_norm > 0.0) {
            return 0.0;
        }
        let a_probe = render(
            probe.view(),
            oversample,
            delta_safe.view(),
            flux_safe.view(),
            zeros_background.view(),
        )
        .expect("render preconditions are guaranteed by build_epsf validation");
        let at_w_a_probe = accumulate(
            a_probe.view(),
            Some(weight_acc.view()),
            oversample,
            stamp_size,
            delta_safe.view(),
            flux_safe.view(),
        )
        .expect(
            "accumulate preconditions are guaranteed by build_epsf validation",
        );
        let next_norm = l2_norm(at_w_a_probe.iter());
        if !next_norm.is_finite() {
            return 0.0;
        }
        estimate = next_norm / probe_norm;
        if next_norm == 0.0 {
            return 0.0;
        }
        probe = at_w_a_probe.mapv(|x| x / next_norm);
    }
    estimate
}

/// Euclidean norm of an iterator of `f64`.
fn l2_norm<'a, I: Iterator<Item = &'a f64>>(iter: I) -> f64 {
    iter.map(|&x| x * x).sum::<f64>().sqrt()
}

/// Run `refine_nuisance` and copy the per-star results back into the
/// running `(flux, background, delta, ok)` (a `!ok` star carries the
/// atomic `(NaN, NaN, delta_init)` sentinel from `nuisance`).
#[allow(clippy::too_many_arguments)]
fn run_refine(
    psi: &Array2<f64>,
    oversample: usize,
    data_f: &Array3<f64>,
    weight: Option<&Array3<f64>>,
    delta_init: &Array2<f64>,
    nuisance_max_iter: usize,
    nuisance_tol: f64,
    flux: &mut Array1<f64>,
    background: &mut Array1<f64>,
    delta: &mut Array2<f64>,
    ok: &mut Array1<bool>,
) {
    let refined = refine_nuisance::<f64>(
        psi.view(),
        oversample,
        data_f.view(),
        weight.map(|w| w.view()),
        delta_init.view(),
        nuisance_max_iter,
        nuisance_tol,
    )
    .expect(
        "refine_nuisance preconditions are guaranteed by build_epsf validation",
    );
    flux.assign(&refined.flux);
    background.assign(&refined.background);
    delta.assign(&refined.delta);
    ok.assign(&refined.ok);
}

/// Internal seed (fork 5): the `delta ~ 0` integer-aligned stack's
/// robust mean, lifted to the oversampled grid. Per-stamp background =
/// border-ring median; per-stamp flux = sum of the background-subtracted
/// finite pixels; the normalized stamps are `robust_combine`d
/// (`ClippedMean`, the only seed reuse of Phase 4) and the native
/// `(s, s)` coadd is `os x os` block-replicated to `(os*s, os*s)`. A
/// degenerate seed falls back to a flat positive psi.
fn seed_psi(
    data_f: &Array3<f64>,
    oversample: usize,
    stamp_size: usize,
    side: usize,
) -> Array2<f64> {
    let batch_size = data_f.shape()[0];
    if batch_size == 0 {
        return Array2::<f64>::from_elem((side, side), 1.0);
    }
    let mut normalized =
        Array3::<f64>::from_elem((batch_size, stamp_size, stamp_size), f64::NAN);
    for n in 0..batch_size {
        let stamp = data_f.index_axis(Axis(0), n);
        let background = border_ring_median(&stamp, stamp_size);
        let mut flux_estimate = 0.0;
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                let v = stamp[(i, j)];
                if v.is_finite() {
                    flux_estimate += v - background;
                }
            }
        }
        if !(flux_estimate.is_finite() && flux_estimate > 0.0) {
            continue; // dropped from the seed coadd
        }
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                let v = stamp[(i, j)];
                if v.is_finite() {
                    normalized[(n, i, j)] = (v - background) / flux_estimate;
                }
            }
        }
    }

    let coadd = robust_combine::<f64>(
        normalized.view(),
        None,
        CombineMethod::ClippedMean {
            kappa: SEED_CLIP_KAPPA,
            max_iter: SEED_CLIP_MAX_ITER,
        },
    )
    .expect("robust_combine preconditions hold for the seed coadd")
    .combined;

    let mut psi = Array2::<f64>::zeros((side, side));
    let mut any_positive = false;
    for p in 0..side {
        let native_row = p / oversample;
        for q in 0..side {
            let native_col = q / oversample;
            let value = coadd[(native_row, native_col)];
            if value.is_finite() && value > 0.0 {
                psi[(p, q)] = value;
                any_positive = true;
            }
        }
    }
    if !any_positive {
        return Array2::<f64>::from_elem((side, side), 1.0);
    }
    psi
}

/// Robust median of a stamp's border ring (first/last row and column);
/// `0.0` when the ring has no finite pixel.
fn border_ring_median(
    stamp: &ArrayView2<f64>,
    stamp_size: usize,
) -> f64 {
    if stamp_size == 0 {
        return 0.0;
    }
    let mut ring: Vec<f64> = Vec::new();
    for j in 0..stamp_size {
        let top = stamp[(0, j)];
        if top.is_finite() {
            ring.push(top);
        }
        let bottom = stamp[(stamp_size - 1, j)];
        if bottom.is_finite() {
            ring.push(bottom);
        }
    }
    for i in 1..stamp_size.saturating_sub(1) {
        let left = stamp[(i, 0)];
        if left.is_finite() {
            ring.push(left);
        }
        let right = stamp[(i, stamp_size - 1)];
        if right.is_finite() {
            ring.push(right);
        }
    }
    if ring.is_empty() {
        return 0.0;
    }
    median(&mut ring)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::psf::render::render;
    use ndarray::{Array1, Array2, Array3};

    /// SplitMix64: a tiny, dependency-free, fully deterministic PRNG so
    /// the synthetic scatter is reproducible (mirrors the
    /// accumulate/nuisance test RNG).
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

    /// Isotropic Gaussian model on the oversampled grid.
    fn gaussian_epsf(side: usize, sigma: f64, amplitude: f64) -> Array2<f64> {
        let center = (side as f64 - 1.0) / 2.0;
        Array2::from_shape_fn((side, side), |(p, q)| {
            let dp = p as f64 - center;
            let dq = q as f64 - center;
            amplitude * (-0.5 * (dp * dp + dq * dq) / (sigma * sigma)).exp()
        })
    }

    /// A genuinely asymmetric, strictly positive model (Gaussian times a
    /// positive linear ramp) so any row/col confusion shows up.
    fn asymmetric_epsf(side: usize, sigma: f64) -> Array2<f64> {
        let center = (side as f64 - 1.0) / 2.0;
        Array2::from_shape_fn((side, side), |(p, q)| {
            let dp = p as f64 - center;
            let dq = q as f64 - center;
            let g = (-0.5 * (dp * dp + dq * dq) / (sigma * sigma)).exp();
            g * (1.0 + 0.20 * (p as f64 / side as f64))
                * (1.0 + 0.12 * (q as f64 / side as f64))
        })
    }

    /// Normalize a model to unit volume (`sum / os^2 = 1`) for a
    /// gauge-invariant comparison against `build_epsf`'s output.
    fn unit_volume(psi: &Array2<f64>, oversample: usize) -> Array2<f64> {
        let volume = psi.sum() / (oversample * oversample) as f64;
        psi.mapv(|x| x / volume)
    }

    /// Synthesize a noise-free stamp stack via the forward operator.
    fn synth(
        epsf: &Array2<f64>,
        oversample: usize,
        delta: &Array2<f64>,
        flux: &Array1<f64>,
        background: &Array1<f64>,
    ) -> Array3<f64> {
        render(
            epsf.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap()
    }

    /// Relative L2 reconstruction error `‖render(out) - data‖ / ‖data‖`
    /// (gauge-invariant: independent of the psi/flux split).
    fn recon_rel_error(
        result: &BuildEpsf,
        oversample: usize,
        data: &Array3<f64>,
    ) -> f64 {
        let model = render(
            result.epsf.view(),
            oversample,
            result.delta.view(),
            result.flux.view(),
            result.background.view(),
        )
        .unwrap();
        let mut diff = 0.0;
        let mut base = 0.0;
        for (m, d) in model.iter().zip(data.iter()) {
            diff += (m - d) * (m - d);
            base += d * d;
        }
        (diff / base).sqrt()
    }

    fn rel_l2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut diff = 0.0;
        let mut base = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            diff += (x - y) * (x - y);
            base += y * y;
        }
        (diff / base).sqrt()
    }

    fn scattered_deltas(rng: &mut SplitMix64, n: usize) -> Array2<f64> {
        // Spread over [-0.45, 0.45)^2 -- the sub-pixel phase diversity
        // that makes the super-resolution well posed.
        Array2::from_shape_fn((n, 2), |_| rng.range(-0.45, 0.45))
    }

    // --- Core end-to-end pin: warm start recovers psi* exactly. ---

    #[test]
    fn warm_start_recovers_known_psi_and_star_table() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = asymmetric_epsf(side, oversample as f64 * 1.4);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0001);
        let n = 8;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(50.0, 150.0));
        let background = Array1::from_shape_fn(n, |_| rng.range(2.0, 9.0));
        let data = synth(&psi_true, oversample, &delta, &flux, &background);

        let params = BuildEpsfParams {
            max_iter: 30,
            tol: 1e-9,
            step: 1.0,
            residual_reweight: ResidualReweight::None,
            nuisance_max_iter: 10,
            nuisance_tol: 1e-12,
        };
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            params,
        )
        .unwrap();

        // Shape, unit-volume gauge, non-negativity.
        assert_eq!(result.epsf.dim(), (side, side));
        assert!(
            (result.epsf.sum() / (oversample * oversample) as f64 - 1.0).abs()
                < 1e-9
        );
        assert!(result.epsf.iter().all(|&x| x >= 0.0 && x.is_finite()));
        assert!(result.ok.iter().all(|&o| o));
        // ~1 round for a clean warm start (fork 5 / decision 4).
        assert!(
            result.iterations <= 3,
            "iterations = {}",
            result.iterations
        );
        assert!(result.converged);
        // psi recovered up to the unit-volume gauge.
        assert!(
            rel_l2(&result.epsf, &unit_volume(&psi_true, oversample)) < 1e-4
        );
        // Star table + reconstruction (gauge-invariant pin).
        assert!(recon_rel_error(&result, oversample, &data) < 1e-6);
        for n in 0..n {
            assert!((result.delta[(n, 0)] - delta[(n, 0)]).abs() < 1e-4);
            assert!((result.delta[(n, 1)] - delta[(n, 1)]).abs() < 1e-4);
        }
    }

    // --- Seed path (psi_init = None) runs through and converges. ---

    #[test]
    fn seed_path_runs_and_converges() {
        let oversample = 3;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.5, 100.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0002);
        let n = 14;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(60.0, 140.0));
        let background = Array1::from_shape_fn(n, |_| rng.range(1.0, 5.0));
        let data = synth(&psi_true, oversample, &delta, &flux, &background);

        // Production-default step (1.0); the cold seed path is pinned
        // on the gauge-invariant reconstruction, the meaningful
        // end-to-end measure. Plain projected Landweber with no
        // explicit regularizer (v1, decision 10) has semi-convergence:
        // the data fit converges fast while the ill-posed high-frequency
        // psi detail plateaus -- the tight psi* recovery pin is the
        // warm-start test, not this one.
        let params = BuildEpsfParams {
            max_iter: 250,
            tol: 1e-7,
            step: 1.0,
            residual_reweight: ResidualReweight::None,
            nuisance_max_iter: 3,
            nuisance_tol: 1e-8,
        };
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            None,
            params,
        )
        .unwrap();

        assert_eq!(result.epsf.dim(), (side, side));
        assert!(result.epsf.iter().all(|&x| x >= 0.0 && x.is_finite()));
        assert!(result.ok.iter().all(|&o| o));
        assert!(
            (result.epsf.sum() / (oversample * oversample) as f64 - 1.0).abs()
                < 1e-6
        );
        // The crude block-replicated seed has been sharpened well past
        // any blurred starting point.
        // Gauge-invariant reconstruction is the meaningful pin.
        let err = recon_rel_error(&result, oversample, &data);
        assert!(err < 0.01, "seed-path reconstruction error = {err}");
        // psi shape is in the right ballpark (not garbage / divergent);
        // exact recovery is the warm-start test's job.
        let shape = rel_l2(&result.epsf, &unit_volume(&psi_true, oversample));
        assert!(shape < 0.25, "seed-path psi shape error = {shape}");
        assert!(result.iterations <= 250);
    }

    // --- Convergence flag / max_iter discipline. ---

    #[test]
    fn max_iter_is_respected_and_converged_semantics() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.4, 100.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0003);
        let n = 6;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(50.0, 120.0));
        let background = Array1::from_elem(n, 3.0);
        let data = synth(&psi_true, oversample, &delta, &flux, &background);

        // max_iter = 1: exactly one outer iteration, and `converged`
        // is false by the "before the budget was exhausted" rule
        // (mirrors refine_nuisance: iter + 1 < max_iter).
        let capped = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(capped.iterations, 1);
        assert!(!capped.converged);

        // A generous budget on a clean warm start converges strictly
        // before max_iter.
        let free = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 50,
                tol: 1e-8,
                nuisance_max_iter: 8,
                nuisance_tol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(free.converged);
        assert!(free.iterations < 50);
    }

    // --- Non-negativity holds even with negative data pixels. ---

    #[test]
    fn output_nonnegative_with_negative_data_pixels() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.3, 80.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0004);
        let n = 6;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(40.0, 90.0));
        // A negative background pushes many detector pixels negative;
        // they are kept (decision 5: never dropped by sign).
        let background = Array1::from_elem(n, -7.0);
        let data = synth(&psi_true, oversample, &delta, &flux, &background);
        assert!(data.iter().any(|&v| v < 0.0));

        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 30,
                tol: 1e-8,
                nuisance_max_iter: 8,
                nuisance_tol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.epsf.iter().all(|&x| x >= 0.0 && x.is_finite()));
        assert!(result.ok.iter().all(|&o| o));
        assert!(recon_rel_error(&result, oversample, &data) < 1e-5);
    }

    // --- Unit-volume gauge helper: sum and flux*psi invariance. ---

    #[test]
    fn gauge_unit_volume_invariance() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let mut psi = gaussian_epsf(side, oversample as f64 * 1.5, 42.0);
        let mut flux = Array1::from_vec(vec![3.0, 7.5]);
        let delta = Array2::from_shape_vec((2, 2), vec![0.1, -0.2, -0.3, 0.05])
            .unwrap();
        let background = Array1::from_vec(vec![1.0, 2.0]);
        let before = render(
            psi.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();

        gauge_unit_volume(&mut psi, &mut flux, oversample);

        assert!(
            (psi.sum() / (oversample * oversample) as f64 - 1.0).abs() < 1e-12
        );
        let after = render(
            psi.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();
        // f * psi product is invariant under the gauge.
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-9 * a.abs().max(1.0));
        }
    }

    // --- Residual reweight reduces and symmetrizes outlier bias. ---

    fn build_with_outlier(
        sign: f64,
        method: ResidualReweight,
    ) -> (BuildEpsf, Array2<f64>, usize) {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.4, 100.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0005);
        let n = 8;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(60.0, 130.0));
        let background = Array1::from_elem(n, 4.0);
        let mut data = synth(&psi_true, oversample, &delta, &flux, &background);
        // Corrupt a few interior pixels of star 0 with a gross,
        // symmetric outlier.
        for (i, j) in [(2, 2), (2, 4), (4, 3)] {
            data[(0, i, j)] += sign * 5000.0;
        }
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 25,
                tol: 1e-9,
                step: 1.0,
                residual_reweight: method,
                nuisance_max_iter: 6,
                nuisance_tol: 1e-10,
            },
        )
        .unwrap();
        (result, unit_volume(&psi_true, oversample), oversample)
    }

    #[test]
    fn residual_reweight_reduces_outlier_bias() {
        let (none_res, truth, os) =
            build_with_outlier(1.0, ResidualReweight::None);
        let (huber_res, _, _) =
            build_with_outlier(1.0, ResidualReweight::Huber { c: 3.0 });
        let (tukey_res, _, _) =
            build_with_outlier(1.0, ResidualReweight::Tukey { c: 4.0 });

        let err_none = rel_l2(&none_res.epsf, &truth);
        let err_huber = rel_l2(&huber_res.epsf, &truth);
        let err_tukey = rel_l2(&tukey_res.epsf, &truth);
        assert!(
            err_huber < err_none,
            "huber {err_huber} vs none {err_none}"
        );
        assert!(
            err_tukey < err_none,
            "tukey {err_tukey} vs none {err_none}"
        );
        let _ = os;
    }

    #[test]
    fn residual_reweight_is_sign_agnostic() {
        // +outlier and -outlier must yield the same recovery error
        // (decision 5: rejection by magnitude, never by sign).
        let (plus, truth, _) =
            build_with_outlier(1.0, ResidualReweight::Huber { c: 3.0 });
        let (minus, _, _) =
            build_with_outlier(-1.0, ResidualReweight::Huber { c: 3.0 });
        let err_plus = rel_l2(&plus.epsf, &truth);
        let err_minus = rel_l2(&minus.epsf, &truth);
        assert!(
            (err_plus - err_minus).abs() < 1e-6 * err_plus.max(1e-6),
            "asymmetric: +{err_plus} vs -{err_minus}"
        );
    }

    // --- Per-star !ok exclusion: an all-NaN stamp drops out. ---

    #[test]
    fn per_star_not_ok_is_excluded_from_psi() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.4, 100.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0006);
        let n = 7;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(50.0, 120.0));
        let background = Array1::from_elem(n, 3.0);
        let mut data = synth(&psi_true, oversample, &delta, &flux, &background);
        // Star 3 is entirely NaN: refine_nuisance fails it (ok=false)
        // and it must not pollute the psi back-projection.
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                data[(3, i, j)] = f64::NAN;
            }
        }
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 30,
                tol: 1e-9,
                nuisance_max_iter: 8,
                nuisance_tol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!result.ok[3]);
        assert!(result.flux[3].is_nan());
        for n in 0..n {
            if n != 3 {
                assert!(result.ok[n], "star {n} should be ok");
            }
        }
        assert!(result.epsf.iter().all(|&x| x >= 0.0 && x.is_finite()));
        assert!(
            rel_l2(&result.epsf, &unit_volume(&psi_true, oversample)) < 1e-3
        );
    }

    // --- N = 1 and the empty batch. ---

    #[test]
    fn single_star_recovers() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.4, 100.0);
        let delta = Array2::from_shape_vec((1, 2), vec![0.12, -0.18]).unwrap();
        let flux = Array1::from_vec(vec![90.0]);
        let background = Array1::from_vec(vec![4.0]);
        let data = synth(&psi_true, oversample, &delta, &flux, &background);
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            BuildEpsfParams {
                max_iter: 30,
                tol: 1e-9,
                nuisance_max_iter: 8,
                nuisance_tol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.ok[0]);
        assert!(recon_rel_error(&result, oversample, &data) < 1e-5);
    }

    #[test]
    fn empty_batch_is_legal() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let data = Array3::<f64>::zeros((0, stamp_size, stamp_size));
        let delta = Array2::<f64>::zeros((0, 2));
        let result = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            oversample,
            None,
            BuildEpsfParams::default(),
        )
        .unwrap();
        assert_eq!(result.epsf.dim(), (side, side));
        assert_eq!(result.flux.len(), 0);
        assert_eq!(result.background.len(), 0);
        assert_eq!(result.delta.dim(), (0, 2));
        assert_eq!(result.ok.len(), 0);
        assert_eq!(result.iterations, 0);
        assert!(result.converged);
    }

    // --- f32 / f64 dual path. ---

    #[test]
    fn f32_and_f64_paths_agree() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;
        let psi_true = gaussian_epsf(side, oversample as f64 * 1.4, 100.0);
        let mut rng = SplitMix64::new(0x5EED_1234_ABCD_0007);
        let n = 6;
        let delta = scattered_deltas(&mut rng, n);
        let flux = Array1::from_shape_fn(n, |_| rng.range(50.0, 110.0));
        let background = Array1::from_elem(n, 3.0);
        let data64 = synth(&psi_true, oversample, &delta, &flux, &background);
        let data32 = data64.mapv(|x| x as f32);

        let params = BuildEpsfParams {
            max_iter: 25,
            tol: 1e-7,
            nuisance_max_iter: 6,
            nuisance_tol: 1e-9,
            ..Default::default()
        };
        let r64 = build_epsf::<f64>(
            data64.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            params.clone(),
        )
        .unwrap();
        let r32 = build_epsf::<f32>(
            data32.view(),
            None,
            delta.view(),
            oversample,
            Some(psi_true.view()),
            params,
        )
        .unwrap();
        assert!(r64.ok.iter().all(|&o| o));
        assert!(r32.ok.iter().all(|&o| o));
        // Both recover the truth; the only difference is the f32 input
        // rounding.
        assert!(rel_l2(&r64.epsf, &r32.epsf) < 1e-3);
        assert!(
            rel_l2(&r64.epsf, &unit_volume(&psi_true, oversample)) < 1e-3
        );
    }

    // --- Default params are sane. ---

    #[test]
    fn default_params_are_sane() {
        let p = BuildEpsfParams::default();
        assert_eq!(p.max_iter, DEFAULT_MAX_ITER);
        assert_eq!(p.tol, DEFAULT_TOL);
        assert_eq!(p.step, DEFAULT_STEP);
        assert_eq!(p.residual_reweight, ResidualReweight::None);
        assert_eq!(p.nuisance_max_iter, DEFAULT_NUISANCE_MAX_ITER);
        assert_eq!(p.nuisance_tol, DEFAULT_NUISANCE_TOL);
    }

    // --- Hard preconditions (Err; no Ok(None)). ---

    fn ok_params() -> BuildEpsfParams {
        BuildEpsfParams::default()
    }

    #[test]
    fn err_oversample_not_odd() {
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = Array2::<f64>::zeros((2, 2));
        let err = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            4,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(err, BuildEpsfError::OversampleNotOdd { oversample: 4 });
        // oversample == 0 is also rejected here.
        let err0 = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            0,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(err0, BuildEpsfError::OversampleNotOdd { oversample: 0 });
    }

    #[test]
    fn err_stamp_size_even() {
        let data = Array3::<f64>::zeros((2, 6, 6));
        let delta = Array2::<f64>::zeros((2, 2));
        let err = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            5,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(err, BuildEpsfError::StampSizeEven { stamp_size: 6 });
    }

    #[test]
    fn err_batch_length_mismatch() {
        // Non-square stamp.
        let data = Array3::<f64>::zeros((2, 7, 5));
        let delta = Array2::<f64>::zeros((2, 2));
        let err = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            5,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            BuildEpsfError::BatchLengthMismatch {
                data: (2, 7, 5),
                delta_init: (2, 2),
            }
        );
        // delta_init not (N, 2).
        let data2 = Array3::<f64>::zeros((2, 7, 7));
        let delta_bad = Array2::<f64>::zeros((3, 2));
        let err2 = build_epsf::<f64>(
            data2.view(),
            None,
            delta_bad.view(),
            5,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(
            err2,
            BuildEpsfError::BatchLengthMismatch {
                data: (2, 7, 7),
                delta_init: (3, 2),
            }
        );
    }

    #[test]
    fn err_weight_shape_mismatch() {
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = Array2::<f64>::zeros((2, 2));
        let weight = Array3::<f64>::ones((2, 7, 5));
        let err = build_epsf::<f64>(
            data.view(),
            Some(weight.view()),
            delta.view(),
            5,
            None,
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            BuildEpsfError::WeightShapeMismatch {
                weight: (2, 7, 5),
                data: (2, 7, 7),
            }
        );
    }

    #[test]
    fn err_psi_init_shape_mismatch() {
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = Array2::<f64>::zeros((2, 2));
        let psi_bad = Array2::<f64>::zeros((34, 35));
        let err = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            5,
            Some(psi_bad.view()),
            ok_params(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            BuildEpsfError::PsiInitShapeMismatch {
                psi_init: (34, 35),
                expected: (35, 35),
            }
        );
    }

    #[test]
    fn err_params_invalid() {
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = Array2::<f64>::zeros((2, 2));
        let cases = [
            BuildEpsfParams {
                max_iter: 0,
                ..Default::default()
            },
            BuildEpsfParams {
                tol: 0.0,
                ..Default::default()
            },
            BuildEpsfParams {
                tol: f64::NAN,
                ..Default::default()
            },
            BuildEpsfParams {
                step: -1.0,
                ..Default::default()
            },
            BuildEpsfParams {
                step: f64::INFINITY,
                ..Default::default()
            },
            BuildEpsfParams {
                nuisance_max_iter: 0,
                ..Default::default()
            },
            BuildEpsfParams {
                nuisance_tol: -3.0,
                ..Default::default()
            },
            BuildEpsfParams {
                residual_reweight: ResidualReweight::Huber { c: 0.0 },
                ..Default::default()
            },
            BuildEpsfParams {
                residual_reweight: ResidualReweight::Tukey { c: f64::NAN },
                ..Default::default()
            },
        ];
        for params in cases {
            let err = build_epsf::<f64>(
                data.view(),
                None,
                delta.view(),
                5,
                None,
                params,
            )
            .unwrap_err();
            assert!(matches!(err, BuildEpsfError::ParamsInvalid { .. }));
        }
        // A valid Huber c is accepted.
        let okp = BuildEpsfParams {
            residual_reweight: ResidualReweight::Huber { c: 3.0 },
            ..Default::default()
        };
        assert!(
            build_epsf::<f64>(
                data.view(),
                None,
                delta.view(),
                5,
                None,
                okp
            )
            .is_ok()
        );
    }

    #[test]
    fn shape_preconditions_precede_params_invalid() {
        // oversample even AND params invalid -> the shape precondition
        // wins (ParamsInvalid is checked last, after every shape check).
        let data = Array3::<f64>::zeros((2, 7, 7));
        let delta = Array2::<f64>::zeros((2, 2));
        let err = build_epsf::<f64>(
            data.view(),
            None,
            delta.view(),
            4,
            None,
            BuildEpsfParams {
                max_iter: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(err, BuildEpsfError::OversampleNotOdd { oversample: 4 });
    }
}

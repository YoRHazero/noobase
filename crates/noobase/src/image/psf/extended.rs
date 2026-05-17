//! Extended-PSF construction: stitch the Phase 6 oversampled core with a
//! native-resolution wing/spike robustly stacked from bright stars.
//!
//! Two entry points (one file, like `nuisance` holds
//! `solve_flux_background` + `refine_nuisance`):
//!
//! - [`stitch_psf`] is the independent leaf: a Phase 6 oversampled core
//!   plus an already-built native wing -> a hybrid, encircled-energy
//!   normalized [`ExtendedPsf`]. It is *non-generic* `f64` (it operates
//!   on already-built model arrays, like `render`/`accumulate`).
//! - [`build_extended_psf`] is the *orchestrator*: a bright-star
//!   large-support native stamp stack plus the Phase 6 core ->
//!   per-stamp robust background + per-stamp scale -> a cross-`M` robust
//!   wing via [`super::robust::robust_combine`] -> [`stitch_psf`]. It
//!   owns no operator of its own: the wing stack combine is delegated to
//!   the Phase 4 reducer and the per-stamp scale solve to the Phase 5
//!   leaf [`super::nuisance::solve_flux_background`]. Nothing here
//!   re-implements the robust combine, the 2x2 weighted solve or
//!   Catmull-Rom sampling -- that coupling to the Phase 4/5/6 stack is
//!   deliberate and is what keeps the orchestrator an orchestrator.
//!
//! `robust_combine` is consumed here as the *proper* decision-3 consumer
//! (the cross-`N` **stack combine** that hard-rejects whole bright-star
//! samples), distinct from the Phase 6 seed reuse. The prose verb here
//! is always "combine"; it is never mixed with Phase 6's per-pixel
//! `ResidualReweight` (a smooth per-pixel *reweight*, a different
//! mechanism).
//!
//! Design notes:
//!
//! - **Four load-bearing forks (decided, authoritative).**
//!   1. Consume the Phase 6 psi; the core is *not* re-solved. `core` is
//!      an input to both entry points; Phase 7 only builds the wing and
//!      stitches.
//!   2. Hybrid / multi-resolution [`ExtendedPsf`] (oversampled core +
//!      native wing + the linking meta-info), so decision 3 ("only the
//!      core region is oversampled") is not lost. The documented
//!      sampling contract: a consumer uses the core for
//!      `r < match_radius - feather_width/2`, the raised-cosine feather
//!      blend across the ring, and the wing for
//!      `r > match_radius + feather_width/2`; concretely
//!      `recon(r) = f_core(r) * core_sample(r) + wing_sample(r)`, where
//!      the stored `wing` already has the feather `f_wing` and the
//!      scalar match baked in (so it is ~0 in the core region) while the
//!      stored `core` is the whole oversampled core (the consumer
//!      applies `f_core = 1 - f_wing`).
//!   3. Wing stack: integer alignment + per-stamp scalar robust
//!      background + per-stamp scale. The scale is *preferentially*
//!      solved against the Phase 6 core psi with
//!      `solve_flux_background` on the bright star's central `s x s`
//!      region; the *fallback* is aperture photometry (used when the
//!      core solve is `!ok`, i.e. the core is saturated).
//!      `star_scale_from_core` records which path was taken; a star is
//!      dropped only if *both* fail.
//!   4. Stitch math: scalar-match the wing to the core's azimuthal
//!      average at `match_radius` (so the seam is continuous), then a
//!      raised-cosine (Hann) feather across the ring; one global
//!      encircled-energy normalization so the total integral within
//!      `ee_aperture_radius` (a native-pixel circular aperture) is 1.
//!
//! - **Six sub-decisions (decided; labelled A-F for the ROADMAP).**
//!   1. (A) A saturated core falls back to aperture-photometry scale
//!      rather than dropping the star (otherwise the bright-star sample
//!      is too small); the saturated core pixels are excluded by their
//!      weight/mask upstream.
//!   2. (B) Stitch is scalar-match-first, then raised-cosine feather, so
//!      the reconstructed radial profile is C1-continuous at both ring
//!      edges (no seam derivative jump); the output keeps both grids.
//!   3. (C) The encircled-energy unit is a native-pixel circular
//!      aperture; the aperture integral is v1 pixel-center counted, with
//!      an exact area-fraction hook left for later (the decision-6
//!      "scalar bg v1 + plane hook" rhythm).
//!   4. (D) `wing_delta` is taken but only as integer-recentering
//!      bookkeeping; the sub-pixel part is *not* applied (the wing is
//!      sky-dominated; resampling would correlate its noise, decision
//!      3).
//!   5. (E) Errors are hard preconditions only; there is no `Ok(None)`.
//!      A per-star algorithmic failure (the core solve *and* the
//!      aperture fallback are both unusable) sets `star_ok = false` and
//!      excludes that star from the wing stack -- a per-star sentinel,
//!      never a global `Err` (the `nuisance`/`build_epsf` per-star
//!      spirit plus `robust_combine`'s per-pixel NaN sentinel).
//!   6. (F) The decomposition is hybrid: a `stitch_psf` leaf plus a
//!      `build_extended_psf` orchestrator (mirroring Phase 5's
//!      `solve_flux_background` + `refine_nuisance`), with the dual
//!      return structs [`ExtendedPsf`] / [`ExtendedPsfBuilt`] splitting
//!      the leaf and orchestrator outputs (mirroring
//!      `FluxBackground` / `NuisanceRefined`).
//!
//! - **Conventions inherited, not re-decided.** `build_extended_psf` is
//!   data-path generic over `T: Float` (`wing_data`/`wing_weight`,
//!   bright-star detector data, like `robust`/`nuisance`/`build_stamp`);
//!   `core` is always `f64` (the model is always `f64`, like `render`).
//!   `stitch_psf` is non-generic `f64`. Internally everything is `f64`:
//!   `wing_data`/`wing_weight` are upcast to owned `f64` arrays on entry
//!   and `T` never threads through the flow (the `robust`/`nuisance`/
//!   `build_epsf` "internal f64" convention). `weight` is the complete
//!   per-pixel weight (`Option`; `None` = unit), mask polarity folded
//!   into `weight = 0` by the caller (decision 7). The core native side
//!   `s` is recovered from `core.shape()` / `oversample` (like
//!   `render`/`nuisance`, never passed separately); the wing `(H, W)` is
//!   its own native large support (independent of `s`, larger than the
//!   core stamp). The core center `(os*s - 1)/2` follows `render`
//!   (decision 12); the wing is centered. `robust_combine` already
//!   parallelizes over pixels and `solve_flux_background` over `M`; the
//!   orchestrator only orchestrates. Preconditions are checked in
//!   declaration order with `ParamsInvalid` / `StitchParamsInvalid`
//!   after the shape preconditions (mirroring
//!   `build_epsf`/`nuisance`/`robust`). `M = 0` is legal (no bright
//!   stars -> an all-sentinel wing -> a pure-core EE-normalized
//!   extended PSF).
//!
//! Landing decisions (the Phase-7 "implementation details decided
//! during development", recorded here as Phase 5/6 recorded
//! `MIN_VALID_PIXELS` / the operator norm):
//!
//! - **Seam scalar match = exact-`match_radius` azimuthal average.**
//!   The core and wing are sampled on the circle of native radius
//!   `match_radius` at [`NUM_AZIMUTH`] equally spaced angles (via the
//!   shared [`super::kernel::catmull_rom_sample`], the project's locked
//!   interpolation kernel, reused -- never re-implemented). The wing
//!   average is `wing_confidence`-weighted when a confidence plane is
//!   given (low-count edge samples do not bias the match). The scale is
//!   `core_az / wing_az`; if that is not finite (e.g. an all-sentinel
//!   wing, the `M = 0` / all-`!ok` degenerate case) the scale is `0`
//!   (the wing contributes nothing -> a pure-core EE-normalized
//!   result).
//! - **Core native profile is sampled, not os-binned.** The core's
//!   value at a native offset is taken by Catmull-Rom sampling the
//!   oversampled grid (a native pixel center lands exactly on an
//!   oversampled grid point, so this is interpolation-free, the
//!   decision-12 `delta = 0` property), keeping the oversampled
//!   precision rather than averaging it away with an `os x os` bin.
//! - **Raised-cosine feather (exact form).** With
//!   `inner = match_radius - feather_width/2`,
//!   `outer = match_radius + feather_width/2`,
//!   `f_wing(r) = 0` for `r <= inner`, `1` for `r >= outer`, and
//!   `0.5 * (1 - cos(pi * (r - inner) / feather_width))` between;
//!   `f_core = 1 - f_wing`. Both have zero derivative at `inner` and
//!   `outer`, so the blended profile is C1 (no seam kink).
//! - **EE v1 pixel-center.** The encircled energy is summed over the
//!   native wing-grid pixels whose center is within
//!   `ee_aperture_radius`, each pixel contributing
//!   `f_core(r) * core_native + wing_plane_raw` (the same recon the
//!   consumer evaluates). The global factor `g = 1 / EE_raw` (or `1`
//!   when `EE_raw` is non-finite / non-positive, the degenerate guard)
//!   scales *both* the stored core and the stored wing.
//! - **Per-stamp background = robust annulus median.** A scalar sky from
//!   the median of the finite (and, when a weight is given,
//!   positive-weight) pixels in the `scale_background_annulus`
//!   `(r_in, r_out)` ring; `0.0` when the ring has no usable pixel
//!   (neutral, like `build_stamp`'s empty border ring). It is used both
//!   for the wing normalization subtraction and for the aperture
//!   photometry fallback. The core-solve's own fitted background is
//!   *not* reused for the wing (the wing's large support wants its own
//!   sky, decision 3).
//! - **Aperture photometry fallback (v1 recipe).** The scale is the sum
//!   of `(data - bg)` over the finite, positive-weight pixels within
//!   `scale_aperture_radius` (a plain aperture sum; a growth-curve
//!   refinement is a later hook). A non-finite / non-positive sum fails
//!   the star.
//! - **Per-star scale precondition.** The core solve is attempted only
//!   when the wing support covers the core stamp (`H >= s` and
//!   `W >= s`, so the central `s x s` window exists); otherwise every
//!   star goes straight to the aperture fallback.
//! - **Wing weighting through the normalization (v1).** The robust
//!   combine uses the caller's base weight directly (with `!ok` stars
//!   and invalid pixels excluded); the `scale^2` inverse-variance
//!   refinement of a per-stamp-scaled sample is a documented hook left
//!   for later (the decision-6 "v1 simple + hook" rhythm). A `!ok` star
//!   contributes an all-`NaN` normalized stamp (excluded by
//!   `robust_combine`'s value-finiteness gate, the `build_epsf` seed
//!   pattern); its base weight stamp is also zeroed when a weight is
//!   given.
//! - **`wing_confidence` is not carried in `ExtendedPsf`.** The
//!   provisional struct is authoritative; the confidence plane only
//!   weights the seam scalar-match azimuthal average (and gates
//!   sampling at zero-confidence wing pixels). It does not enter the
//!   stored wing plane or the EE.
//! - **No minimum-star precondition.** `M = 0` is legal (pure core) and
//!   a small `M` only yields a noisier wing (`robust_combine` handles
//!   the stack); there is no saturation-threshold constant -- saturation
//!   is handled entirely through the weight/mask gate plus the core
//!   solve `!ok` -> aperture fallback.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use thiserror::Error;

use crate::float::Float;

use super::kernel::catmull_rom_sample;
use super::nuisance::solve_flux_background;
use super::robust::{CombineMethod, robust_combine};

/// Number of equally spaced azimuthal samples for the seam scalar-match
/// (and the radial profiles the tests evaluate). A scale factor, not a
/// precise integral, so a fixed budget is enough.
const NUM_AZIMUTH: usize = 720;

/// Default core<->wing crossover radius (native px).
pub const DEFAULT_MATCH_RADIUS: f64 = 8.0;
/// Default raised-cosine feather full width (native px).
pub const DEFAULT_FEATHER_WIDTH: f64 = 4.0;
/// Default encircled-energy aperture radius (native px).
pub const DEFAULT_EE_APERTURE_RADIUS: f64 = 15.0;
/// Default aperture-photometry fallback scale radius (native px).
pub const DEFAULT_SCALE_APERTURE_RADIUS: f64 = 6.0;
/// Default fallback per-stamp robust sky annulus `(r_in, r_out)` (native
/// px).
pub const DEFAULT_SCALE_BACKGROUND_ANNULUS: (f64, f64) = (18.0, 24.0);

/// The core<->wing stitch geometry, shared by [`stitch_psf`] and
/// [`build_extended_psf`]. `impl Default` gives Rust-only callers a sane
/// configuration (decision 9).
#[derive(Debug, Clone, PartialEq)]
pub struct StitchParams {
    /// Native-pixel radius at which the wing is scalar-matched to the
    /// core (the feather ring center). Finite, `> 0`.
    pub match_radius: f64,
    /// Native-pixel full width of the raised-cosine feather ring.
    /// Finite, `> 0`.
    pub feather_width: f64,
    /// Native-pixel radius of the circular aperture the extended PSF is
    /// encircled-energy normalized to (integral within == 1). Finite,
    /// `> 0`.
    pub ee_aperture_radius: f64,
}

impl Default for StitchParams {
    fn default() -> Self {
        Self {
            match_radius: DEFAULT_MATCH_RADIUS,
            feather_width: DEFAULT_FEATHER_WIDTH,
            ee_aperture_radius: DEFAULT_EE_APERTURE_RADIUS,
        }
    }
}

/// Packed control parameters for [`build_extended_psf`]. `impl Default`
/// gives Rust-only callers a sane configuration (decision 9).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedPsfParams {
    /// Core<->wing stitch geometry.
    pub stitch: StitchParams,
    /// Phase 4 `robust_combine` method for the cross-`M` wing stack (the
    /// decision-3 proper consumer).
    pub combine: CombineMethod,
    /// Native-pixel aperture radius for the aperture-photometry fallback
    /// scale (used when the core solve is `!ok`; sub-decision A).
    /// Finite, `> 0`.
    pub scale_aperture_radius: f64,
    /// Native-pixel `(r_in, r_out)` ring for the per-stamp robust sky.
    /// Finite, `0 <= r_in < r_out`.
    pub scale_background_annulus: (f64, f64),
}

impl Default for ExtendedPsfParams {
    fn default() -> Self {
        Self {
            stitch: StitchParams::default(),
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: DEFAULT_SCALE_APERTURE_RADIUS,
            scale_background_annulus: DEFAULT_SCALE_BACKGROUND_ANNULUS,
        }
    }
}

/// Output of [`stitch_psf`]: the hybrid / multi-resolution extended PSF
/// (fork 2).
///
/// `recon(r) = f_core(r) * core_sample(r) + wing_sample(r)` with
/// `f_core = 1 - f_wing` and the raised-cosine `f_wing`. The stored
/// `wing` already has `f_wing` and the scalar match baked in (so it is
/// ~0 in the core region); the stored `core` is the whole oversampled
/// core (the consumer applies `f_core`). Both planes are scaled by the
/// single global encircled-energy factor.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedPsf {
    /// `(os*s, os*s)` oversampled core, EE-normalized (the Phase 6 psi
    /// times the global EE factor).
    pub core: Array2<f64>,
    /// The oversample factor `os` (the meta-info linking the two grids).
    pub oversample: usize,
    /// `(H, W)` native wing: scalar-matched to the core at
    /// `match_radius`, raised-cosine feathered (so ~0 in the core
    /// region), and EE-normalized.
    pub wing: Array2<f64>,
    /// The native-pixel core<->wing crossover radius.
    pub match_radius: f64,
    /// The native-pixel feather full width.
    pub feather_width: f64,
    /// The native-pixel encircled-energy aperture radius.
    pub ee_aperture_radius: f64,
}

/// Output of [`build_extended_psf`]: the stitched [`ExtendedPsf`] plus
/// the per-bright-star table (mirroring Phase 5's
/// `FluxBackground`/`NuisanceRefined` dual-struct split).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedPsfBuilt {
    /// The stitched hybrid extended PSF.
    pub extended: ExtendedPsf,
    /// `(M,)` per-stamp scale actually used (the core-solve flux or the
    /// aperture-photometry fallback). `NaN` for a `!star_ok` star.
    pub star_flux: Array1<f64>,
    /// `(M,)` per-stamp robust sky actually subtracted. `NaN` for a
    /// `!star_ok` star.
    pub star_background: Array1<f64>,
    /// `(M,)` per-star usability: it entered the wing combine. `false`
    /// means it was excluded (saturated *and* the aperture fallback was
    /// also unusable).
    pub star_ok: Array1<bool>,
    /// `(M,)` scale provenance: `true` = the core
    /// `solve_flux_background`, `false` = the aperture-photometry
    /// fallback (sub-decision A). Meaningless (`false`) for a
    /// `!star_ok` star.
    pub star_scale_from_core: Array1<bool>,
}

/// Errors returned by [`stitch_psf`] for ill-shaped / ill-parameterized
/// inputs.
///
/// Every variant is a hard precondition. Like the
/// `render`/`accumulate`/`robust`/`nuisance`/`build_epsf` family (and
/// unlike `build_stamp`), there is no algorithmic skip path: a
/// well-formed call always yields an [`ExtendedPsf`] (a degenerate
/// all-sentinel wing yields a pure-core EE-normalized result, not an
/// `Err`). `StitchParamsInvalid` is checked after the shape
/// preconditions.
#[derive(Debug, Error, PartialEq)]
pub enum StitchError {
    #[error("oversample must be odd; got {oversample}")]
    OversampleNotOdd { oversample: usize },
    #[error("core must be square; got ({rows}, {cols})")]
    CoreNotSquare { rows: usize, cols: usize },
    #[error(
        "core side ({core_side}) must be an integer multiple of oversample ({oversample})"
    )]
    CoreSizeNotMultiple { core_side: usize, oversample: usize },
    #[error(
        "derived stamp_size (core_side / oversample) must be odd; got {stamp_size}"
    )]
    DerivedStampSizeEven { stamp_size: usize },
    #[error("wing must have odd dimensions to have a defined center; got ({rows}, {cols})")]
    WingNotOdd { rows: usize, cols: usize },
    #[error("wing_confidence shape {confidence:?} must equal wing shape {wing:?}")]
    WingConfidenceShapeMismatch {
        confidence: (usize, usize),
        wing: (usize, usize),
    },
    #[error(
        "stitch params must be finite and > 0 with 0 <= match_radius - feather_width/2 and match_radius + feather_width/2 within both the core and wing native extent and ee_aperture_radius within the wing native extent; got match_radius = {match_radius}, feather_width = {feather_width}, ee_aperture_radius = {ee_aperture_radius}"
    )]
    StitchParamsInvalid {
        match_radius: f64,
        feather_width: f64,
        ee_aperture_radius: f64,
    },
}

/// Errors returned by [`build_extended_psf`] for ill-shaped /
/// ill-parameterized inputs.
///
/// Every variant is a hard precondition; there is no `Ok(None)` path. A
/// bright star that cannot be calibrated is handled per-star with
/// `star_ok = false`, not an `Err`. `ParamsInvalid` is checked after the
/// shape preconditions.
#[derive(Debug, Error, PartialEq)]
pub enum ExtendedPsfError {
    #[error("oversample must be odd; got {oversample}")]
    OversampleNotOdd { oversample: usize },
    #[error("core must be square; got ({rows}, {cols})")]
    CoreNotSquare { rows: usize, cols: usize },
    #[error(
        "core side ({core_side}) must be an integer multiple of oversample ({oversample})"
    )]
    CoreSizeNotMultiple { core_side: usize, oversample: usize },
    #[error(
        "derived stamp_size (core_side / oversample) must be odd; got {stamp_size}"
    )]
    DerivedStampSizeEven { stamp_size: usize },
    #[error(
        "wing_data must have odd spatial dimensions to have a defined center; got ({rows}, {cols})"
    )]
    WingNotOdd { rows: usize, cols: usize },
    #[error(
        "batch dimensions disagree: wing_data shape {wing_data:?} must be (M, H, W) and wing_delta shape {wing_delta:?} must be (M, 2)"
    )]
    BatchLengthMismatch {
        wing_data: (usize, usize, usize),
        wing_delta: (usize, usize),
    },
    #[error("wing_weight shape {weight:?} must equal wing_data shape {wing_data:?}")]
    WeightShapeMismatch {
        weight: (usize, usize, usize),
        wing_data: (usize, usize, usize),
    },
    #[error(
        "params invalid: stitch params (see StitchParamsInvalid), the combine method (ClippedMean kappa > 0 and max_iter > 0), scale_aperture_radius finite and > 0 and within the wing extent, and scale_background_annulus (r_in, r_out) finite with 0 <= r_in < r_out within the wing extent; got match_radius = {match_radius}, feather_width = {feather_width}, ee_aperture_radius = {ee_aperture_radius}, scale_aperture_radius = {scale_aperture_radius}"
    )]
    ParamsInvalid {
        match_radius: f64,
        feather_width: f64,
        ee_aperture_radius: f64,
        scale_aperture_radius: f64,
    },
}

/// Resolved, validated core geometry shared by both entry points.
struct CoreGeometry {
    /// Core native side `s = core_side / oversample` (odd).
    stamp_size: usize,
    /// Oversampled-grid center `(os*s - 1) / 2` (exact integer in f64).
    core_center: f64,
    /// `oversample` as f64 (the native -> oversampled scale).
    oversample_f: f64,
    /// Core native half-extent `(s - 1) / 2` (the fully covered radius).
    core_native_half: f64,
}

/// Validate the core shape/parity preconditions shared by both entry
/// points and resolve the core geometry. Variant constructors are passed
/// in so the same logic serves both error enums (checked in declaration
/// order, first failure returned).
fn validate_core(
    core: &ArrayView2<f64>,
    oversample: usize,
) -> Result<CoreGeometry, CoreValidationError> {
    let rows = core.shape()[0];
    let cols = core.shape()[1];
    if rows != cols {
        return Err(CoreValidationError::NotSquare { rows, cols });
    }
    if oversample % 2 == 0 {
        // Also rejects oversample == 0 (0 is even, hence not odd).
        return Err(CoreValidationError::OversampleNotOdd { oversample });
    }
    let core_side = rows;
    if core_side % oversample != 0 {
        return Err(CoreValidationError::SizeNotMultiple {
            core_side,
            oversample,
        });
    }
    let stamp_size = core_side / oversample;
    if stamp_size % 2 == 0 {
        // Catches the degenerate empty model (stamp_size == 0) too.
        return Err(CoreValidationError::StampSizeEven { stamp_size });
    }
    Ok(CoreGeometry {
        stamp_size,
        core_center: (core_side as f64 - 1.0) / 2.0,
        oversample_f: oversample as f64,
        core_native_half: (stamp_size as f64 - 1.0) / 2.0,
    })
}

/// Internal core-validation outcome, mapped to whichever public error
/// enum the caller uses (the variants are identical in both).
#[derive(Debug)]
enum CoreValidationError {
    NotSquare { rows: usize, cols: usize },
    OversampleNotOdd { oversample: usize },
    SizeNotMultiple { core_side: usize, oversample: usize },
    StampSizeEven { stamp_size: usize },
}

impl From<CoreValidationError> for StitchError {
    fn from(error: CoreValidationError) -> Self {
        match error {
            CoreValidationError::NotSquare { rows, cols } => {
                StitchError::CoreNotSquare { rows, cols }
            }
            CoreValidationError::OversampleNotOdd { oversample } => {
                StitchError::OversampleNotOdd { oversample }
            }
            CoreValidationError::SizeNotMultiple {
                core_side,
                oversample,
            } => StitchError::CoreSizeNotMultiple {
                core_side,
                oversample,
            },
            CoreValidationError::StampSizeEven { stamp_size } => {
                StitchError::DerivedStampSizeEven { stamp_size }
            }
        }
    }
}

impl From<CoreValidationError> for ExtendedPsfError {
    fn from(error: CoreValidationError) -> Self {
        match error {
            CoreValidationError::NotSquare { rows, cols } => {
                ExtendedPsfError::CoreNotSquare { rows, cols }
            }
            CoreValidationError::OversampleNotOdd { oversample } => {
                ExtendedPsfError::OversampleNotOdd { oversample }
            }
            CoreValidationError::SizeNotMultiple {
                core_side,
                oversample,
            } => ExtendedPsfError::CoreSizeNotMultiple {
                core_side,
                oversample,
            },
            CoreValidationError::StampSizeEven { stamp_size } => {
                ExtendedPsfError::DerivedStampSizeEven { stamp_size }
            }
        }
    }
}

/// `true` when the stitch geometry is infeasible for the given core /
/// wing native extents. Used by both entry points (mapped to
/// `StitchParamsInvalid` / folded into `ParamsInvalid`).
fn stitch_params_infeasible(
    params: &StitchParams,
    core_native_half: f64,
    wing_half: f64,
) -> bool {
    let positive_finite =
        |x: f64| x.is_finite() && x > 0.0;
    if !positive_finite(params.match_radius)
        || !positive_finite(params.feather_width)
        || !positive_finite(params.ee_aperture_radius)
    {
        return true;
    }
    let inner = params.match_radius - 0.5 * params.feather_width;
    let outer = params.match_radius + 0.5 * params.feather_width;
    // The ring must not underflow the center, the core must cover the
    // ring (its native half-extent), and the wing must cover both the
    // ring and the EE aperture.
    inner < 0.0
        || outer > core_native_half
        || outer > wing_half
        || params.ee_aperture_radius > wing_half
}

/// `true` when the `robust_combine` method is ill-parameterized
/// (mirrors `robust_combine`'s own `ClippedMeanInvalidParams`
/// precondition).
fn combine_method_invalid(method: CombineMethod) -> bool {
    match method {
        CombineMethod::ClippedMean { kappa, max_iter } => {
            kappa <= 0.0 || max_iter == 0
        }
        CombineMethod::Median => false,
    }
}

/// The raised-cosine (Hann) wing feather weight at native radius `r`.
/// `0` for `r <= inner`, `1` for `r >= outer`,
/// `0.5 * (1 - cos(pi * (r - inner)/feather_width))` between (C1 at both
/// edges). `f_core = 1 - f_wing`.
fn feather_wing_weight(r: f64, match_radius: f64, feather_width: f64) -> f64 {
    let inner = match_radius - 0.5 * feather_width;
    let outer = match_radius + 0.5 * feather_width;
    if r <= inner {
        0.0
    } else if r >= outer {
        1.0
    } else {
        0.5 * (1.0 - (std::f64::consts::PI * (r - inner) / feather_width).cos())
    }
}

/// Confidence-weighted azimuthal average of `image` on the circle of
/// native radius `r` about `(center_row, center_col)`, sampled at
/// [`NUM_AZIMUTH`] angles via the shared Catmull-Rom kernel. Optional
/// `confidence` (same grid as `image`) weights each angular sample and
/// gates out zero/non-finite-confidence samples. Returns `NaN` when no
/// finite sample contributes.
fn azimuthal_average(
    image: &ArrayView2<f64>,
    center_row: f64,
    center_col: f64,
    radius: f64,
    sample_scale: f64,
    confidence: Option<&ArrayView2<f64>>,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for angle_index in 0..NUM_AZIMUTH {
        let angle =
            2.0 * std::f64::consts::PI * (angle_index as f64) / (NUM_AZIMUTH as f64);
        let offset_row = radius * angle.sin();
        let offset_col = radius * angle.cos();
        let row_coord = center_row + sample_scale * offset_row;
        let col_coord = center_col + sample_scale * offset_col;
        let value = catmull_rom_sample(image, row_coord, col_coord);
        if !value.is_finite() {
            continue;
        }
        let sample_weight = match confidence {
            Some(confidence_view) => {
                let c = catmull_rom_sample(
                    confidence_view,
                    row_coord,
                    col_coord,
                );
                if !(c.is_finite() && c > 0.0) {
                    continue;
                }
                c
            }
            None => 1.0,
        };
        weighted_sum += sample_weight * value;
        weight_sum += sample_weight;
    }
    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        f64::NAN
    }
}

/// Robust median of a slice (sorts in place; NaN-free by construction at
/// every call site here).
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

/// Robust per-stamp sky: the median of the finite (and, when a weight is
/// given, positive-weight) pixels whose center falls in the
/// `(r_in, r_out)` native annulus about the stamp center. `0.0` when the
/// ring has no usable pixel (neutral, like `build_stamp`'s empty border
/// ring).
fn annulus_background(
    stamp: &ArrayView2<f64>,
    weight: Option<&ArrayView2<f64>>,
    center_row: f64,
    center_col: f64,
    r_in: f64,
    r_out: f64,
) -> f64 {
    let (rows, cols) = (stamp.shape()[0], stamp.shape()[1]);
    let mut ring: Vec<f64> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let dr = row as f64 - center_row;
            let dc = col as f64 - center_col;
            let radius = (dr * dr + dc * dc).sqrt();
            if radius < r_in || radius > r_out {
                continue;
            }
            let value = stamp[(row, col)];
            if !value.is_finite() {
                continue;
            }
            if let Some(weight_view) = weight {
                let w = weight_view[(row, col)];
                if !(w.is_finite() && w > 0.0) {
                    continue;
                }
            }
            ring.push(value);
        }
    }
    if ring.is_empty() {
        return 0.0;
    }
    median(&mut ring)
}

/// Aperture-photometry fallback scale (sub-decision A): the sum of
/// `(data - background)` over the finite, positive-weight pixels within
/// `aperture_radius` of the stamp center. Returns `NaN` when the sum is
/// non-finite or non-positive (the star then fails).
fn aperture_photometry_scale(
    stamp: &ArrayView2<f64>,
    weight: Option<&ArrayView2<f64>>,
    center_row: f64,
    center_col: f64,
    aperture_radius: f64,
    background: f64,
) -> f64 {
    let (rows, cols) = (stamp.shape()[0], stamp.shape()[1]);
    let mut sum = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let dr = row as f64 - center_row;
            let dc = col as f64 - center_col;
            if (dr * dr + dc * dc).sqrt() > aperture_radius {
                continue;
            }
            let value = stamp[(row, col)];
            if !value.is_finite() {
                continue;
            }
            if let Some(weight_view) = weight {
                let w = weight_view[(row, col)];
                if !(w.is_finite() && w > 0.0) {
                    continue;
                }
            }
            sum += value - background;
        }
    }
    if sum.is_finite() && sum > 0.0 {
        sum
    } else {
        f64::NAN
    }
}

/// Stitch a Phase 6 oversampled core and an already-built native wing
/// into a hybrid, encircled-energy normalized extended PSF.
///
/// See the module-level documentation for the sampling contract, the
/// scalar-match-then-feather math and the landing decisions. This is the
/// independent leaf; it is *non-generic* `f64` (it operates on
/// already-built model arrays, like `render`/`accumulate`).
///
/// # Parameters
///
/// - `core`: `(os*s, os*s)` Phase 6 oversampled effective PSF
///   (unit-volume gauged). `s` is recovered from `core.shape()` /
///   `oversample`.
/// - `oversample`: `os`, odd (decision 1/12).
/// - `wing`: `(H, W)` native large-support wing, centered; `H`, `W`
///   odd. An all-`NaN` (sentinel) wing yields a pure-core result.
/// - `wing_confidence`: optional `(H, W)` per-pixel certainty (e.g.
///   `robust_combine`'s combined weight); `None` is uniform. It only
///   weights the seam scalar-match.
/// - `params`: the stitch geometry.
///
/// # Returns
///
/// `Ok(ExtendedPsf)` with the EE-normalized core and feathered,
/// scalar-matched, EE-normalized wing plus the linking meta-info.
///
/// # Errors
///
/// Returns a [`StitchError`] for the shape/parity preconditions (odd
/// `oversample`, square `core`, `oversample` dividing the core side, odd
/// derived `stamp_size`, odd `wing` dims, a `Some` `wing_confidence`
/// matching `wing`) checked first, then `StitchParamsInvalid` when the
/// geometry is infeasible.
pub fn stitch_psf(
    core: ArrayView2<f64>,
    oversample: usize,
    wing: ArrayView2<f64>,
    wing_confidence: Option<ArrayView2<f64>>,
    params: StitchParams,
) -> Result<ExtendedPsf, StitchError> {
    let geometry = validate_core(&core, oversample)?;

    let wing_rows = wing.shape()[0];
    let wing_cols = wing.shape()[1];
    if wing_rows % 2 == 0 || wing_cols % 2 == 0 {
        return Err(StitchError::WingNotOdd {
            rows: wing_rows,
            cols: wing_cols,
        });
    }
    if let Some(confidence_view) = wing_confidence.as_ref() {
        let confidence_shape = confidence_view.shape();
        if confidence_shape[0] != wing_rows || confidence_shape[1] != wing_cols {
            return Err(StitchError::WingConfidenceShapeMismatch {
                confidence: (confidence_shape[0], confidence_shape[1]),
                wing: (wing_rows, wing_cols),
            });
        }
    }
    let wing_half = ((wing_rows as f64 - 1.0) / 2.0)
        .min((wing_cols as f64 - 1.0) / 2.0);
    if stitch_params_infeasible(&params, geometry.core_native_half, wing_half) {
        return Err(StitchError::StitchParamsInvalid {
            match_radius: params.match_radius,
            feather_width: params.feather_width,
            ee_aperture_radius: params.ee_aperture_radius,
        });
    }

    Ok(stitch_core_and_wing(&core, &geometry, &wing, wing_confidence.as_ref(), &params))
}

/// The stitch body, shared by `stitch_psf` and the orchestrator (which
/// has already validated every precondition). Pure `f64`, no error
/// path.
fn stitch_core_and_wing(
    core: &ArrayView2<f64>,
    geometry: &CoreGeometry,
    wing: &ArrayView2<f64>,
    wing_confidence: Option<&ArrayView2<f64>>,
    params: &StitchParams,
) -> ExtendedPsf {
    let wing_rows = wing.shape()[0];
    let wing_cols = wing.shape()[1];
    let wing_center_row = (wing_rows as f64 - 1.0) / 2.0;
    let wing_center_col = (wing_cols as f64 - 1.0) / 2.0;

    // --- Seam scalar match (fork 4 / sub-decision B). Azimuthal average
    // of the core and the wing exactly at `match_radius`. A native
    // radial offset of `r` maps to `os * r` on the oversampled core
    // grid (the decision-12 sampling scale). ---
    let core_at_match = azimuthal_average(
        core,
        geometry.core_center,
        geometry.core_center,
        params.match_radius,
        geometry.oversample_f,
        None,
    );
    let wing_at_match = azimuthal_average(
        wing,
        wing_center_row,
        wing_center_col,
        params.match_radius,
        1.0,
        wing_confidence,
    );
    // `scale * wing` matches the core at the seam. A non-finite ratio
    // (e.g. an all-sentinel wing, the M = 0 / all-!ok degenerate case)
    // drops the wing entirely -> a pure-core EE-normalized result.
    let scale = {
        let candidate = core_at_match / wing_at_match;
        if candidate.is_finite() {
            candidate
        } else {
            0.0
        }
    };

    // --- Feathered, scalar-matched wing plane (pre-EE). `f_wing` is
    // baked in (so it is ~0 in the core region); a non-finite raw wing
    // pixel contributes nothing. ---
    let mut wing_plane = Array2::<f64>::zeros((wing_rows, wing_cols));
    for row in 0..wing_rows {
        for col in 0..wing_cols {
            let dr = row as f64 - wing_center_row;
            let dc = col as f64 - wing_center_col;
            let radius = (dr * dr + dc * dc).sqrt();
            let f_wing = feather_wing_weight(
                radius,
                params.match_radius,
                params.feather_width,
            );
            let raw = wing[(row, col)];
            if f_wing > 0.0 && scale != 0.0 && raw.is_finite() {
                wing_plane[(row, col)] = f_wing * scale * raw;
            }
        }
    }

    // --- One global encircled-energy normalization (fork 4 /
    // sub-decision C). EE_raw = sum over native wing-grid pixel centers
    // within `ee_aperture_radius` of the same recon the consumer
    // evaluates: `f_core * core_native + wing_plane_raw`. ---
    let mut ee_raw = 0.0;
    for row in 0..wing_rows {
        for col in 0..wing_cols {
            let dr = row as f64 - wing_center_row;
            let dc = col as f64 - wing_center_col;
            let radius = (dr * dr + dc * dc).sqrt();
            if radius > params.ee_aperture_radius {
                continue;
            }
            let f_wing = feather_wing_weight(
                radius,
                params.match_radius,
                params.feather_width,
            );
            let core_native = catmull_rom_sample(
                core,
                geometry.core_center + geometry.oversample_f * dr,
                geometry.core_center + geometry.oversample_f * dc,
            );
            let core_term = if core_native.is_finite() {
                (1.0 - f_wing) * core_native
            } else {
                0.0
            };
            ee_raw += core_term + wing_plane[(row, col)];
        }
    }
    let global_factor = if ee_raw.is_finite() && ee_raw > 0.0 {
        1.0 / ee_raw
    } else {
        1.0
    };

    ExtendedPsf {
        core: core.mapv(|v| v * global_factor),
        oversample: geometry.oversample_f as usize,
        wing: wing_plane.mapv(|v| v * global_factor),
        match_radius: params.match_radius,
        feather_width: params.feather_width,
        ee_aperture_radius: params.ee_aperture_radius,
    }
}

/// Build the extended PSF from a bright-star large-support native stamp
/// stack and the Phase 6 core: per-stamp robust background + per-stamp
/// scale -> a cross-`M` robust wing -> [`stitch_psf`].
///
/// See the module-level documentation for the four forks, the six
/// sub-decisions and the landing decisions. This is the orchestrator; it
/// delegates the cross-`M` combine to [`super::robust::robust_combine`]
/// and the per-stamp scale solve to
/// [`super::nuisance::solve_flux_background`] -- it re-implements
/// neither, nor the Catmull-Rom kernel.
///
/// # Parameters
///
/// - `wing_data`: `(M, H, W)` caller-aligned bright-star native stamps.
///   May be `f32` or `f64`; upcast to `f64` internally.
/// - `wing_weight`: optional `(M, H, W)` complete per-pixel base weight;
///   `None` is unit weight (decision 7). A non-finite / non-positive
///   pixel weight excludes that pixel (folds in the `true = invalid`
///   mask).
/// - `wing_delta`: `(M, 2)` bright-star `build_stamp` `delta`. Only the
///   integer recentering is applied (sub-decision D); the sub-pixel part
///   is deliberately dropped.
/// - `core`: `(os*s, os*s)` Phase 6 effective PSF (fork 1: consumed, not
///   re-solved). `s` is recovered from `core.shape()` / `oversample`.
/// - `oversample`: `os`, odd.
/// - `params`: see [`ExtendedPsfParams`].
///
/// # Returns
///
/// `Ok(ExtendedPsfBuilt)` with the stitched [`ExtendedPsf`] and the
/// `(M,)` per-star table. `M = 0` yields a pure-core EE-normalized
/// extended PSF and empty star arrays.
///
/// # Errors
///
/// Returns an [`ExtendedPsfError`] for the shape/parity preconditions
/// (odd `oversample`, square `core`, `oversample` dividing the core
/// side, odd derived `stamp_size`, odd `wing_data` dims, `wing_data`
/// being `(M, H, W)` with `wing_delta` `(M, 2)`, a `Some` `wing_weight`
/// matching `wing_data`) checked first, then `ParamsInvalid`.
pub fn build_extended_psf<T: Float>(
    wing_data: ArrayView3<T>,
    wing_weight: Option<ArrayView3<T>>,
    wing_delta: ArrayView2<f64>,
    core: ArrayView2<f64>,
    oversample: usize,
    params: ExtendedPsfParams,
) -> Result<ExtendedPsfBuilt, ExtendedPsfError> {
    // --- Hard preconditions (Err only; no Ok(None)). Declaration
    // order: core shape/parity, then wing dims, batch, weight shape,
    // then ParamsInvalid. ---
    let geometry = validate_core(&core, oversample)?;

    let star_count = wing_data.shape()[0];
    let wing_rows = wing_data.shape()[1];
    let wing_cols = wing_data.shape()[2];
    if wing_rows % 2 == 0 || wing_cols % 2 == 0 {
        return Err(ExtendedPsfError::WingNotOdd {
            rows: wing_rows,
            cols: wing_cols,
        });
    }
    if wing_delta.shape()[0] != star_count || wing_delta.shape()[1] != 2 {
        return Err(ExtendedPsfError::BatchLengthMismatch {
            wing_data: (star_count, wing_rows, wing_cols),
            wing_delta: (wing_delta.shape()[0], wing_delta.shape()[1]),
        });
    }
    if let Some(weight_view) = wing_weight.as_ref() {
        let weight_shape = weight_view.shape();
        if weight_shape[0] != star_count
            || weight_shape[1] != wing_rows
            || weight_shape[2] != wing_cols
        {
            return Err(ExtendedPsfError::WeightShapeMismatch {
                weight: (weight_shape[0], weight_shape[1], weight_shape[2]),
                wing_data: (star_count, wing_rows, wing_cols),
            });
        }
    }
    let wing_half = ((wing_rows as f64 - 1.0) / 2.0)
        .min((wing_cols as f64 - 1.0) / 2.0);
    let (annulus_in, annulus_out) = params.scale_background_annulus;
    let annulus_invalid = !annulus_in.is_finite()
        || !annulus_out.is_finite()
        || annulus_in < 0.0
        || annulus_out <= annulus_in
        || annulus_out > wing_half;
    let scale_aperture_invalid = !(params.scale_aperture_radius.is_finite()
        && params.scale_aperture_radius > 0.0)
        || params.scale_aperture_radius > wing_half;
    let params_invalid = stitch_params_infeasible(
        &params.stitch,
        geometry.core_native_half,
        wing_half,
    ) || combine_method_invalid(params.combine)
        || scale_aperture_invalid
        || annulus_invalid;
    if params_invalid {
        return Err(ExtendedPsfError::ParamsInvalid {
            match_radius: params.stitch.match_radius,
            feather_width: params.stitch.feather_width,
            ee_aperture_radius: params.stitch.ee_aperture_radius,
            scale_aperture_radius: params.scale_aperture_radius,
        });
    }

    // --- Internal f64: upcast once at the boundary; T never threads
    // through the flow (mirrors robust/nuisance/build_epsf). ---
    let wing_data_f: Array3<f64> =
        wing_data.mapv(|value| value.to_f64().unwrap_or(f64::NAN));
    let wing_weight_f: Option<Array3<f64>> = wing_weight
        .as_ref()
        .map(|w| w.mapv(|value| value.to_f64().unwrap_or(f64::NAN)));

    let wing_center_row = (wing_rows as f64 - 1.0) / 2.0;
    let wing_center_col = (wing_cols as f64 - 1.0) / 2.0;
    let stamp_size = geometry.stamp_size;

    // --- Per-stamp scale, preferentially via the Phase 6 core solve
    // (fork 3 / sub-decision A). The core path needs the central
    // `s x s` window, which exists only when the wing support covers
    // the core stamp. One batched `solve_flux_background::<f64>` call
    // (it already parallelizes over M). ---
    let core_path_available =
        star_count > 0 && wing_rows >= stamp_size && wing_cols >= stamp_size;
    let core_solution = if core_path_available {
        let row_offset = (wing_rows - stamp_size) / 2;
        let col_offset = (wing_cols - stamp_size) / 2;
        let mut central = Array3::<f64>::zeros((
            star_count,
            stamp_size,
            stamp_size,
        ));
        let mut central_weight = wing_weight_f
            .as_ref()
            .map(|_| Array3::<f64>::zeros((star_count, stamp_size, stamp_size)));
        for star in 0..star_count {
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    central[(star, i, j)] =
                        wing_data_f[(star, row_offset + i, col_offset + j)];
                    if let (Some(dst), Some(src)) =
                        (central_weight.as_mut(), wing_weight_f.as_ref())
                    {
                        dst[(star, i, j)] =
                            src[(star, row_offset + i, col_offset + j)];
                    }
                }
            }
        }
        Some(
            solve_flux_background::<f64>(
                core,
                oversample,
                central.view(),
                central_weight.as_ref().map(|w| w.view()),
                wing_delta,
            )
            .expect(
                "solve_flux_background preconditions are guaranteed by build_extended_psf validation",
            ),
        )
    } else {
        None
    };

    // --- Per-star background + scale + provenance + the normalized,
    // integer-recentered wing stamp. ---
    let mut normalized =
        Array3::<f64>::from_elem((star_count, wing_rows, wing_cols), f64::NAN);
    let mut normalized_weight = wing_weight_f
        .as_ref()
        .map(|_| Array3::<f64>::zeros((star_count, wing_rows, wing_cols)));
    let mut star_flux = vec![f64::NAN; star_count];
    let mut star_background = vec![f64::NAN; star_count];
    let mut star_ok = vec![false; star_count];
    let mut star_scale_from_core = vec![false; star_count];

    for star in 0..star_count {
        let stamp = wing_data_f.index_axis(Axis(0), star);
        let weight_stamp = wing_weight_f
            .as_ref()
            .map(|w| w.index_axis(Axis(0), star));

        let background = annulus_background(
            &stamp,
            weight_stamp.as_ref(),
            wing_center_row,
            wing_center_col,
            annulus_in,
            annulus_out,
        );

        // Preferred: the core solve flux. Fallback (sub-decision A):
        // aperture photometry. Both unusable -> the star is excluded.
        let core_flux = core_solution.as_ref().and_then(|solution| {
            if solution.ok[star] && solution.flux[star].is_finite()
                && solution.flux[star] > 0.0
            {
                Some(solution.flux[star])
            } else {
                None
            }
        });
        let (scale, from_core) = match core_flux {
            Some(flux) => (flux, true),
            None => {
                let aperture = aperture_photometry_scale(
                    &stamp,
                    weight_stamp.as_ref(),
                    wing_center_row,
                    wing_center_col,
                    params.scale_aperture_radius,
                    background,
                );
                (aperture, false)
            }
        };
        if !(scale.is_finite() && scale > 0.0) {
            // Saturated core AND the aperture fallback unusable: the
            // per-star sentinel (sub-decision E), not a global Err.
            continue;
        }

        // Integer recentering bookkeeping only (sub-decision D): round
        // the sub-pixel delta to the nearest integer (~0 for the
        // build_stamp delta in [-0.5, 0.5)); the sub-pixel part is
        // deliberately not applied (the wing is sky-dominated).
        let shift_row = wing_delta[(star, 0)].round() as i64;
        let shift_col = wing_delta[(star, 1)].round() as i64;
        for row in 0..wing_rows {
            for col in 0..wing_cols {
                let source_row = row as i64 + shift_row;
                let source_col = col as i64 + shift_col;
                if source_row < 0
                    || source_row >= wing_rows as i64
                    || source_col < 0
                    || source_col >= wing_cols as i64
                {
                    continue; // out of the recentered window: missing
                }
                let value =
                    wing_data_f[(star, source_row as usize, source_col as usize)];
                if value.is_finite() {
                    normalized[(star, row, col)] = (value - background) / scale;
                }
                if let (Some(dst), Some(src)) =
                    (normalized_weight.as_mut(), wing_weight_f.as_ref())
                {
                    dst[(star, row, col)] =
                        src[(star, source_row as usize, source_col as usize)];
                }
            }
        }

        star_flux[star] = scale;
        star_background[star] = background;
        star_ok[star] = true;
        star_scale_from_core[star] = from_core;
    }

    // --- Cross-M robust stack combine (decision-3 proper consumer of
    // robust_combine). A !ok star is an all-NaN normalized stamp,
    // excluded by robust_combine's value-finiteness gate (the build_epsf
    // seed pattern); its base weight stamp stays zero too. ---
    let combined = robust_combine::<f64>(
        normalized.view(),
        normalized_weight.as_ref().map(|w| w.view()),
        params.combine,
    )
    .expect("robust_combine preconditions are guaranteed by build_extended_psf validation");

    // The combined weight plane is the per-pixel certainty used to
    // weight the seam scalar match (it does not enter ExtendedPsf).
    let extended = stitch_core_and_wing(
        &core,
        &geometry,
        &combined.combined.view(),
        Some(&combined.weight.view()),
        &params.stitch,
    );

    Ok(ExtendedPsfBuilt {
        extended,
        star_flux: Array1::from(star_flux),
        star_background: Array1::from(star_background),
        star_ok: Array1::from(star_ok),
        star_scale_from_core: Array1::from(star_scale_from_core),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// SplitMix64: a tiny, dependency-free, fully deterministic PRNG so
    /// the random comparisons are reproducible (the same generator as
    /// the `accumulate`/`nuisance`/`build_epsf` tests).
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

    /// Unit-volume oversampled Gaussian core, the Phase 6 psi convention
    /// (`sum(core) / os^2 = 1`). `sigma` is in native pixels; the
    /// oversampled sigma is `os * sigma`.
    fn gaussian_core(oversample: usize, stamp_size: usize, sigma: f64) -> Array2<f64> {
        let side = oversample * stamp_size;
        let center = (side as f64 - 1.0) / 2.0;
        let sigma_os = oversample as f64 * sigma;
        let mut core = Array2::<f64>::zeros((side, side));
        for r in 0..side {
            for c in 0..side {
                let dr = r as f64 - center;
                let dc = c as f64 - center;
                core[(r, c)] =
                    (-(dr * dr + dc * dc) / (2.0 * sigma_os * sigma_os)).exp();
            }
        }
        let volume: f64 = core.iter().sum::<f64>() / (oversample * oversample) as f64;
        core.mapv(|v| v / volume)
    }

    /// The core's native-resolution value at native offset `(dr, dc)`
    /// from center (the decision-12 interpolation-free grid pick, zero
    /// padded beyond the oversampled grid).
    fn core_native_value(
        core: &Array2<f64>,
        oversample: usize,
        dr: f64,
        dc: f64,
    ) -> f64 {
        let side = core.shape()[0];
        let center = (side as f64 - 1.0) / 2.0;
        catmull_rom_sample(
            &core.view(),
            center + oversample as f64 * dr,
            center + oversample as f64 * dc,
        )
    }

    /// The unit-volume Gaussian core's peak (its central value), `= A`
    /// in `core_native(dr, dc) = A * exp(-(dr^2+dc^2)/(2 sigma^2))`.
    fn core_peak(core: &Array2<f64>) -> f64 {
        let mid = (core.shape()[0] - 1) / 2;
        core[(mid, mid)]
    }

    /// The analytic truth profile `peak * exp(-r^2 / 2 sigma^2)`. At
    /// integer native offsets within the oversampled grid this equals
    /// `core_native_value` exactly (decision 12), so a synthetic star's
    /// central window is exactly `flux * core_native + bg` (the Phase 5
    /// solve recovers `flux`/`bg`); unlike the zero-padded
    /// `core_native_value` it extends analytically into the wing, giving
    /// the wing real signal.
    fn truth_native(peak: f64, sigma: f64, dr: f64, dc: f64) -> f64 {
        peak * (-(dr * dr + dc * dc) / (2.0 * sigma * sigma)).exp()
    }

    // --- stitch_psf ---

    #[test]
    fn stitch_seam_is_c0_c1_continuous_and_ee_normalized() {
        // A unit-volume Gaussian core plus a wider Gaussian "wing" with a
        // known scalar offset. After stitching, the reconstructed radial
        // profile must be continuous (C0) and slope-continuous (C1)
        // across the feather ring, and the encircled energy within the
        // aperture must be exactly 1.
        let oversample = 5;
        let stamp_size = 21;
        let core = gaussian_core(oversample, stamp_size, 1.6);

        let wing_side = 81usize;
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        // A broad Gaussian, scaled and offset so it is NOT already
        // matched to the core (the stitch must rescale it).
        let wing = Array2::<f64>::from_shape_fn((wing_side, wing_side), |(r, c)| {
            let dr = r as f64 - wing_center;
            let dc = c as f64 - wing_center;
            17.0 * (-(dr * dr + dc * dc) / (2.0 * 3.4 * 3.4)).exp()
        });

        let params = StitchParams {
            match_radius: 5.0,
            feather_width: 3.0,
            ee_aperture_radius: 18.0,
        };
        let extended =
            stitch_psf(core.view(), oversample, wing.view(), None, params.clone())
                .unwrap();

        // Hybrid shape / meta-info preserved.
        assert_eq!(extended.core.shape(), &[oversample * stamp_size, oversample * stamp_size]);
        assert_eq!(extended.wing.shape(), &[wing_side, wing_side]);
        assert_eq!(extended.oversample, oversample);
        assert_eq!(extended.match_radius, params.match_radius);
        assert_eq!(extended.feather_width, params.feather_width);
        assert_eq!(extended.ee_aperture_radius, params.ee_aperture_radius);

        // The reconstructed radial profile sampled finely across the
        // ring.
        let ec = (extended.core.shape()[0] as f64 - 1.0) / 2.0;
        let recon = |radius: f64| -> f64 {
            let f_wing = feather_wing_weight(
                radius,
                params.match_radius,
                params.feather_width,
            );
            let core_az = azimuthal_average(
                &extended.core.view(),
                ec,
                ec,
                radius,
                oversample as f64,
                None,
            );
            let wing_az = azimuthal_average(
                &extended.wing.view(),
                wing_center,
                wing_center,
                radius,
                1.0,
                None,
            );
            (1.0 - f_wing) * core_az + wing_az
        };
        let step = 0.02;
        let mut radius = 1.0;
        while radius < 12.0 {
            let here = recon(radius);
            let ahead = recon(radius + step);
            // C0: the profile never jumps.
            assert!(
                (here - ahead).abs() < 5e-2 * here.abs().max(1e-3),
                "C0 break at r = {radius}: {here} vs {ahead}"
            );
            radius += step;
        }
        // C1 at the two ring edges specifically: the slope just inside
        // and just outside must agree (raised-cosine has zero derivative
        // there).
        for &edge in &[
            params.match_radius - 0.5 * params.feather_width,
            params.match_radius + 0.5 * params.feather_width,
        ] {
            let h = 0.05;
            let slope_in = (recon(edge) - recon(edge - h)) / h;
            let slope_out = (recon(edge + h) - recon(edge)) / h;
            assert!(
                (slope_in - slope_out).abs() < 5e-2 * slope_in.abs().max(1e-2),
                "C1 break at edge {edge}: {slope_in} vs {slope_out}"
            );
        }

        // EE normalization: the same recon integrated (v1 pixel-center)
        // over the native aperture is exactly 1.
        let mut ee = 0.0;
        for r in 0..wing_side {
            for c in 0..wing_side {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let radius = (dr * dr + dc * dc).sqrt();
                if radius > params.ee_aperture_radius {
                    continue;
                }
                let f_wing = feather_wing_weight(
                    radius,
                    params.match_radius,
                    params.feather_width,
                );
                let core_native = core_native_value(&extended.core, oversample, dr, dc);
                ee += (1.0 - f_wing) * core_native + extended.wing[(r, c)];
            }
        }
        assert!((ee - 1.0).abs() < 1e-9, "EE = {ee}, expected 1");
    }

    #[test]
    fn stitch_feather_partitions_and_wing_is_zero_in_core_region() {
        // f_core + f_wing == 1 everywhere, f_wing monotone, 0 below the
        // inner edge and 1 above the outer edge; the stored wing plane
        // is exactly 0 inside the core region.
        let match_radius = 6.0;
        let feather_width = 4.0;
        let inner = match_radius - 0.5 * feather_width;
        let outer = match_radius + 0.5 * feather_width;
        let mut previous = -1.0;
        let mut radius = 0.0;
        while radius < 12.0 {
            let f_wing = feather_wing_weight(radius, match_radius, feather_width);
            assert!((0.0..=1.0).contains(&f_wing));
            // Partition of unity with f_core.
            assert!(((1.0 - f_wing) + f_wing - 1.0).abs() < 1e-15);
            if radius <= inner {
                assert_eq!(f_wing, 0.0);
            }
            if radius >= outer {
                assert_eq!(f_wing, 1.0);
            }
            assert!(f_wing + 1e-12 >= previous, "f_wing must be monotone");
            previous = f_wing;
            radius += 0.05;
        }

        let oversample = 3;
        // Core native half-extent (s - 1)/2 = 10 must cover the ring
        // outer edge (match_radius + feather_width/2 = 8).
        let stamp_size = 21;
        let core = gaussian_core(oversample, stamp_size, 1.3);
        let wing_side = 61usize;
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        let wing = Array2::<f64>::from_shape_fn((wing_side, wing_side), |(r, c)| {
            let dr = r as f64 - wing_center;
            let dc = c as f64 - wing_center;
            5.0 * (-(dr * dr + dc * dc) / (2.0 * 3.0 * 3.0)).exp()
        });
        let extended = stitch_psf(
            core.view(),
            oversample,
            wing.view(),
            None,
            StitchParams {
                match_radius,
                feather_width,
                ee_aperture_radius: 16.0,
            },
        )
        .unwrap();
        for r in 0..wing_side {
            for c in 0..wing_side {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let radius = (dr * dr + dc * dc).sqrt();
                if radius <= inner {
                    assert_eq!(
                        extended.wing[(r, c)],
                        0.0,
                        "wing must be 0 in the core region at r = {radius}"
                    );
                }
            }
        }
    }

    #[test]
    fn stitch_all_sentinel_wing_yields_pure_core() {
        // The M = 0 / all-!ok degenerate case: an all-NaN wing -> the
        // scale is 0, the wing plane is 0, and the result is the
        // pure-core EE-normalized PSF (no panic, finite everywhere).
        let oversample = 5;
        let stamp_size = 17;
        let core = gaussian_core(oversample, stamp_size, 1.5);
        let wing_side = 71usize;
        let wing = Array2::<f64>::from_elem((wing_side, wing_side), f64::NAN);
        let params = StitchParams {
            match_radius: 6.0,
            feather_width: 3.0,
            ee_aperture_radius: 14.0,
        };
        let extended =
            stitch_psf(core.view(), oversample, wing.view(), None, params.clone())
                .unwrap();
        assert!(extended.wing.iter().all(|&v| v == 0.0));
        assert!(extended.core.iter().all(|v| v.is_finite()));

        // Pure-core EE within the aperture is 1.
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        let mut ee = 0.0;
        for r in 0..wing_side {
            for c in 0..wing_side {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let radius = (dr * dr + dc * dc).sqrt();
                if radius > params.ee_aperture_radius {
                    continue;
                }
                let f_wing = feather_wing_weight(
                    radius,
                    params.match_radius,
                    params.feather_width,
                );
                ee += (1.0 - f_wing)
                    * core_native_value(&extended.core, oversample, dr, dc);
            }
        }
        assert!((ee - 1.0).abs() < 1e-9, "pure-core EE = {ee}");
    }

    #[test]
    fn stitch_params_default_fields() {
        let p = StitchParams::default();
        assert_eq!(p.match_radius, DEFAULT_MATCH_RADIUS);
        assert_eq!(p.feather_width, DEFAULT_FEATHER_WIDTH);
        assert_eq!(p.ee_aperture_radius, DEFAULT_EE_APERTURE_RADIUS);
        let ep = ExtendedPsfParams::default();
        assert_eq!(ep.stitch, StitchParams::default());
        assert_eq!(
            ep.combine,
            CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5
            }
        );
        assert_eq!(ep.scale_aperture_radius, DEFAULT_SCALE_APERTURE_RADIUS);
        assert_eq!(
            ep.scale_background_annulus,
            DEFAULT_SCALE_BACKGROUND_ANNULUS
        );
    }

    #[test]
    fn stitch_preconditions() {
        let oversample = 5;
        let stamp_size = 15;
        let good_core = gaussian_core(oversample, stamp_size, 1.4);
        let good_wing = Array2::<f64>::from_elem((61, 61), 0.01);
        let good = StitchParams {
            match_radius: 6.0,
            feather_width: 3.0,
            ee_aperture_radius: 14.0,
        };

        // OversampleNotOdd (including 0).
        assert_eq!(
            stitch_psf(good_core.view(), 4, good_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::OversampleNotOdd { oversample: 4 }
        );
        assert_eq!(
            stitch_psf(good_core.view(), 0, good_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::OversampleNotOdd { oversample: 0 }
        );
        // CoreNotSquare.
        let rect = Array2::<f64>::zeros((10, 12));
        assert_eq!(
            stitch_psf(rect.view(), 5, good_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::CoreNotSquare { rows: 10, cols: 12 }
        );
        // CoreSizeNotMultiple.
        let not_mult = Array2::<f64>::zeros((76, 76));
        assert_eq!(
            stitch_psf(not_mult.view(), 5, good_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::CoreSizeNotMultiple {
                core_side: 76,
                oversample: 5
            }
        );
        // DerivedStampSizeEven (5 * 4 = 20, derived s = 4).
        let even_s = Array2::<f64>::zeros((20, 20));
        assert_eq!(
            stitch_psf(even_s.view(), 5, good_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::DerivedStampSizeEven { stamp_size: 4 }
        );
        // WingNotOdd.
        let even_wing = Array2::<f64>::zeros((60, 61));
        assert_eq!(
            stitch_psf(good_core.view(), 5, even_wing.view(), None, good.clone())
                .unwrap_err(),
            StitchError::WingNotOdd { rows: 60, cols: 61 }
        );
        // WingConfidenceShapeMismatch.
        let bad_conf = Array2::<f64>::zeros((59, 61));
        assert_eq!(
            stitch_psf(
                good_core.view(),
                5,
                good_wing.view(),
                Some(bad_conf.view()),
                good.clone()
            )
            .unwrap_err(),
            StitchError::WingConfidenceShapeMismatch {
                confidence: (59, 61),
                wing: (61, 61)
            }
        );
        // StitchParamsInvalid: non-finite / non-positive, ring underflow,
        // ring beyond the core, aperture beyond the wing -- all one
        // variant, checked after the shape preconditions.
        for bad in [
            StitchParams {
                match_radius: 0.0,
                feather_width: 3.0,
                ee_aperture_radius: 14.0,
            },
            StitchParams {
                match_radius: 6.0,
                feather_width: f64::NAN,
                ee_aperture_radius: 14.0,
            },
            StitchParams {
                match_radius: 1.0,
                feather_width: 4.0,
                ee_aperture_radius: 14.0,
            }, // inner = -1 < 0
            StitchParams {
                match_radius: 6.0,
                feather_width: 3.0,
                ee_aperture_radius: 100.0,
            }, // aperture beyond the wing half (30)
        ] {
            let err = stitch_psf(
                good_core.view(),
                5,
                good_wing.view(),
                None,
                bad.clone(),
            )
            .unwrap_err();
            // matches! (not assert_eq!): a NaN field makes the derived
            // PartialEq false even for an otherwise-identical struct.
            assert!(
                matches!(err, StitchError::StitchParamsInvalid { .. }),
                "expected StitchParamsInvalid for {bad:?}, got {err:?}"
            );
        }
        // Shape precondition is checked before the params one.
        assert_eq!(
            stitch_psf(
                even_s.view(),
                5,
                good_wing.view(),
                None,
                StitchParams {
                    match_radius: 0.0,
                    feather_width: 0.0,
                    ee_aperture_radius: 0.0,
                }
            )
            .unwrap_err(),
            StitchError::DerivedStampSizeEven { stamp_size: 4 }
        );
    }

    // --- build_extended_psf ---

    /// Synthesize bright-star stamps from a known truth: the whole
    /// `(H, W)` stamp is `flux * truth_native(dr, dc) + background`. On
    /// the central window this is exactly `flux * core_native + bg` (the
    /// Phase 5 solve recovers `flux`/`bg`), while the analytic profile
    /// gives the wing region real signal (the zero-padded
    /// `core_native_value` would leave it flat).
    fn synth_bright_stars(
        core: &Array2<f64>,
        sigma: f64,
        wing_side: usize,
        fluxes: &[f64],
        backgrounds: &[f64],
    ) -> Array3<f64> {
        let m = fluxes.len();
        let peak = core_peak(core);
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        Array3::<f64>::from_shape_fn((m, wing_side, wing_side), |(star, r, c)| {
            let dr = r as f64 - wing_center;
            let dc = c as f64 - wing_center;
            fluxes[star] * truth_native(peak, sigma, dr, dc)
                + backgrounds[star]
        })
    }

    #[test]
    fn build_recovers_wing_vs_robust_combine_reference() {
        // Clean synthetic stars from a known core. The recovered wing
        // must match a direct robust_combine of the hand-normalized
        // stamps (the reference), the per-star table must report the
        // truth, and every star must use the core-solve scale path.
        let oversample = 5;
        let stamp_size = 15;
        // A narrow core so the annulus (far out) sees ~pure background.
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let fluxes = [1000.0, 2500.0, 700.0, 1800.0, 3300.0];
        let backgrounds = [10.0, -4.0, 25.0, 0.0, 7.0];
        let sigma = 1.1;
        let data = synth_bright_stars(
            &core,
            sigma,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let wing_delta = Array2::<f64>::zeros((fluxes.len(), 2));
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: 5.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let built = build_extended_psf(
            data.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();

        // Per-star table: truth recovered, every scale from the core
        // solve.
        for star in 0..fluxes.len() {
            assert!(built.star_ok[star]);
            assert!(built.star_scale_from_core[star]);
            assert!(
                (built.star_flux[star] - fluxes[star]).abs()
                    < 1e-5 * fluxes[star],
                "star {star} flux {} vs {}",
                built.star_flux[star],
                fluxes[star]
            );
            assert!(
                (built.star_background[star] - backgrounds[star]).abs() < 1e-3,
                "star {star} bg {} vs {}",
                built.star_background[star],
                backgrounds[star]
            );
        }

        // Reference: hand-normalize with the known truth and combine.
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        let peak = core_peak(&core);
        let reference_norm = Array3::<f64>::from_shape_fn(
            (fluxes.len(), wing_side, wing_side),
            |(star, r, c)| {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                truth_native(peak, sigma, dr, dc)
                    + (backgrounds[star]
                        - built.star_background[star])
                        / fluxes[star]
            },
        );
        let reference =
            robust_combine::<f64>(reference_norm.view(), None, params.combine)
                .unwrap()
                .combined;
        // Re-stitch the reference with the same geometry and compare the
        // final wing plane (the orchestrator path must equal the
        // hand-built path).
        let geometry = validate_core(&core.view(), oversample).unwrap();
        let reference_extended = stitch_core_and_wing(
            &core.view(),
            &geometry,
            &reference.view(),
            None,
            &params.stitch,
        );
        for (a, b) in built
            .extended
            .wing
            .iter()
            .zip(reference_extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-6 * a.abs().max(1.0),
                "wing mismatch vs reference: {a} vs {b}"
            );
        }
        // EE of the recovered extended PSF is 1.
        let mut ee = 0.0;
        for r in 0..wing_side {
            for c in 0..wing_side {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let radius = (dr * dr + dc * dc).sqrt();
                if radius > params.stitch.ee_aperture_radius {
                    continue;
                }
                let f_wing = feather_wing_weight(
                    radius,
                    params.stitch.match_radius,
                    params.stitch.feather_width,
                );
                ee += (1.0 - f_wing)
                    * core_native_value(&built.extended.core, oversample, dr, dc)
                    + built.extended.wing[(r, c)];
            }
        }
        assert!((ee - 1.0).abs() < 1e-9, "recovered EE = {ee}");
    }

    #[test]
    fn build_saturated_core_uses_aperture_fallback_still_in_stack() {
        // One star's central core pixels are saturated (weight 0) so its
        // core solve is !ok -> it falls back to aperture photometry
        // (star_scale_from_core = false) but STILL enters the stack
        // (star_ok = true); the wing is still recovered.
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let fluxes = [1500.0, 1500.0, 1500.0, 1500.0];
        let backgrounds = [5.0, 5.0, 5.0, 5.0];
        let data = synth_bright_stars(
            &core,
            1.1,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        // Saturate star 1's central s x s window (weight 0 there) so the
        // core solve has too few valid pixels.
        let row_off = (wing_side - stamp_size) / 2;
        let mut weight =
            Array3::<f64>::from_elem((4, wing_side, wing_side), 1.0);
        for i in 0..stamp_size {
            for j in 0..stamp_size {
                weight[(1, row_off + i, row_off + j)] = 0.0;
            }
        }
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::Median,
            scale_aperture_radius: 9.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let built = build_extended_psf(
            data.view(),
            Some(weight.view()),
            Array2::<f64>::zeros((4, 2)).view(),
            core.view(),
            oversample,
            params,
        )
        .unwrap();
        assert!(built.star_ok.iter().all(|&ok| ok));
        assert!(built.star_scale_from_core[0]);
        assert!(!built.star_scale_from_core[1], "saturated star -> fallback");
        assert!(built.star_scale_from_core[2]);
        assert!(built.star_scale_from_core[3]);
        // The fallback star's scale is still finite and positive.
        assert!(built.star_flux[1].is_finite() && built.star_flux[1] > 0.0);
        // The wing is still recovered (finite, positive near center).
        let wc = (wing_side - 1) / 2;
        assert!(built.extended.wing[(wc, wc + 8)].is_finite());
    }

    #[test]
    fn build_uncalibratable_star_excluded_wing_from_rest() {
        // A star with no usable pixel (all weight 0): the core solve is
        // !ok AND the aperture fallback is unusable -> star_ok = false,
        // excluded from the stack; the wing is recovered from the rest.
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let fluxes = [1200.0, 1200.0, 1200.0];
        let backgrounds = [3.0, 3.0, 3.0];
        let sigma = 1.1;
        let data = synth_bright_stars(
            &core,
            sigma,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let mut weight =
            Array3::<f64>::from_elem((3, wing_side, wing_side), 1.0);
        for r in 0..wing_side {
            for c in 0..wing_side {
                weight[(2, r, c)] = 0.0; // star 2 entirely unusable
            }
        }
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let built = build_extended_psf(
            data.view(),
            Some(weight.view()),
            Array2::<f64>::zeros((3, 2)).view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        assert!(built.star_ok[0]);
        assert!(built.star_ok[1]);
        assert!(!built.star_ok[2], "all-zero-weight star must be excluded");
        assert!(built.star_flux[2].is_nan());
        assert!(built.star_background[2].is_nan());
        // The wing is still recovered from the two good stars.
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        let peak = core_peak(&core);
        let reference_norm = Array3::<f64>::from_shape_fn(
            (2, wing_side, wing_side),
            |(star, r, c)| {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let s = if star == 0 { 0 } else { 1 };
                truth_native(peak, sigma, dr, dc)
                    + (backgrounds[s] - built.star_background[s]) / fluxes[s]
            },
        );
        let reference =
            robust_combine::<f64>(reference_norm.view(), None, params.combine)
                .unwrap()
                .combined;
        let geometry = validate_core(&core.view(), oversample).unwrap();
        let reference_extended = stitch_core_and_wing(
            &core.view(),
            &geometry,
            &reference.view(),
            None,
            &params.stitch,
        );
        for (a, b) in built
            .extended
            .wing
            .iter()
            .zip(reference_extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-6 * a.abs().max(1.0),
                "wing must be recovered from the surviving stars"
            );
        }
    }

    #[test]
    fn build_integer_recenter_drops_subpixel_delta() {
        // sub-decision D: only the integer part of wing_delta is applied
        // (~0 for build_stamp delta in [-0.5, 0.5)); the sub-pixel part
        // is deliberately NOT applied. Force the aperture path (zero
        // central weight) so the result is delta-independent, then a
        // sub-pixel delta must give exactly the same wing as delta = 0.
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let fluxes = [900.0, 1400.0, 2000.0];
        let backgrounds = [2.0, 6.0, -3.0];
        let data = synth_bright_stars(
            &core,
            1.1,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let row_off = (wing_side - stamp_size) / 2;
        let mut weight =
            Array3::<f64>::from_elem((3, wing_side, wing_side), 1.0);
        for star in 0..3 {
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    weight[(star, row_off + i, row_off + j)] = 0.0;
                }
            }
        }
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::Median,
            scale_aperture_radius: 9.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let zero_delta = Array2::<f64>::zeros((3, 2));
        let subpixel_delta = Array2::<f64>::from_shape_vec(
            (3, 2),
            vec![0.3, -0.4, 0.49, 0.1, -0.49, 0.2],
        )
        .unwrap();
        let from_zero = build_extended_psf(
            data.view(),
            Some(weight.view()),
            zero_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        let from_subpixel = build_extended_psf(
            data.view(),
            Some(weight.view()),
            subpixel_delta.view(),
            core.view(),
            oversample,
            params,
        )
        .unwrap();
        for (a, b) in from_zero
            .extended
            .wing
            .iter()
            .zip(from_subpixel.extended.wing.iter())
        {
            assert_eq!(
                a, b,
                "sub-pixel delta must not move the wing (integer recenter only)"
            );
        }
    }

    #[test]
    fn build_m1_and_m0() {
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.2);
        let wing_side = 61usize;
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
        };

        // M = 1: a single bright star still produces a wing.
        let data1 =
            synth_bright_stars(&core, 1.2, wing_side, &[1700.0], &[4.0]);
        let built1 = build_extended_psf(
            data1.view(),
            None,
            Array2::<f64>::zeros((1, 2)).view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        assert_eq!(built1.star_ok.len(), 1);
        assert!(built1.star_ok[0]);
        assert!((built1.star_flux[0] - 1700.0).abs() < 1e-2);

        // M = 0: legal -> pure-core EE-normalized, empty star arrays.
        let data0 = Array3::<f64>::zeros((0, wing_side, wing_side));
        let built0 = build_extended_psf(
            data0.view(),
            None,
            Array2::<f64>::zeros((0, 2)).view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        assert_eq!(built0.star_ok.len(), 0);
        assert!(built0.extended.wing.iter().all(|&v| v == 0.0));
        let wing_center = (wing_side as f64 - 1.0) / 2.0;
        let mut ee = 0.0;
        for r in 0..wing_side {
            for c in 0..wing_side {
                let dr = r as f64 - wing_center;
                let dc = c as f64 - wing_center;
                let radius = (dr * dr + dc * dc).sqrt();
                if radius > params.stitch.ee_aperture_radius {
                    continue;
                }
                let f_wing = feather_wing_weight(
                    radius,
                    params.stitch.match_radius,
                    params.stitch.feather_width,
                );
                ee += (1.0 - f_wing)
                    * core_native_value(&built0.extended.core, oversample, dr, dc);
            }
        }
        assert!((ee - 1.0).abs() < 1e-9, "M=0 pure-core EE = {ee}");
    }

    #[test]
    fn build_f32_and_f64_dual_path_agree() {
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let fluxes = [1000.0, 2000.0, 1500.0];
        let backgrounds = [5.0, -2.0, 8.0];
        let data64 = synth_bright_stars(
            &core,
            1.1,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let data32: Array3<f32> = data64.mapv(|v| v as f32);
        let wing_delta = Array2::<f64>::zeros((3, 2));
        let params = ExtendedPsfParams::default();
        // Override the geometry to fit this small synthetic wing.
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
            ..params
        };
        let from64 = build_extended_psf(
            data64.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        let from32 = build_extended_psf(
            data32.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params,
        )
        .unwrap();
        for (a, b) in from64
            .extended
            .wing
            .iter()
            .zip(from32.extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-3 * a.abs().max(1.0),
                "f32/f64 wing mismatch: {a} vs {b}"
            );
        }
        for star in 0..3 {
            assert!(
                (from64.star_flux[star] - from32.star_flux[star]).abs()
                    < 1e-2 * from64.star_flux[star].abs().max(1.0)
            );
        }
    }

    #[test]
    fn build_weight_none_vs_some_consistent() {
        // An all-equal positive weight must give the same result as
        // `None` (unit weight) for the same data.
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.2);
        let wing_side = 61usize;
        let fluxes = [1100.0, 1700.0, 2300.0, 800.0];
        let backgrounds = [4.0, -1.0, 9.0, 2.0];
        let data = synth_bright_stars(
            &core,
            1.2,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let wing_delta = Array2::<f64>::zeros((4, 2));
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let none = build_extended_psf(
            data.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        let ones = Array3::<f64>::from_elem((4, wing_side, wing_side), 1.0);
        let some = build_extended_psf(
            data.view(),
            Some(ones.view()),
            wing_delta.view(),
            core.view(),
            oversample,
            params,
        )
        .unwrap();
        for (a, b) in none
            .extended
            .wing
            .iter()
            .zip(some.extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-9 * a.abs().max(1.0),
                "None vs all-ones weight must agree: {a} vs {b}"
            );
        }
    }

    #[test]
    fn build_robust_recovery_noise_and_symmetric_outlier() {
        // Noisy bright stars plus a single large outlier injected in the
        // wing region of one star (outside the central window and the
        // sky annulus, so neither the core-solve scale nor the annulus
        // background sees it -- only the cross-M combine does). The
        // orchestrator's recovered wing must (a) equal a hand-built
        // (data - bg)/scale -> robust_combine -> stitch reference (the
        // orchestration is exactly that), and (b) be identical whether
        // the outlier is +O or -O (the decision-5 sign-agnostic
        // rejection propagated through the orchestrator).
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.1);
        let wing_side = 61usize;
        let sigma = 1.1;
        let fluxes = [
            1000.0, 1400.0, 900.0, 2100.0, 1700.0, 1200.0, 2600.0, 800.0,
            1500.0,
        ];
        let backgrounds = [5.0, -3.0, 9.0, 1.0, 7.0, 0.0, 12.0, -2.0, 4.0];
        let clean = synth_bright_stars(
            &core,
            sigma,
            wing_side,
            &fluxes,
            &backgrounds,
        );
        let mut rng = SplitMix64::new(0x7151_3771_2DEA_D17F);
        let mut noisy = clean.clone();
        for value in noisy.iter_mut() {
            *value += rng.range(-0.4, 0.4);
        }
        let wc = (wing_side - 1) / 2;
        let outlier_pixel = (0usize, wc, wc + 12); // r = 12: wing region
        let params = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            // kappa < sqrt(M) so a lone per-pixel outlier among the M
            // stars is actually rejected (with kappa == sqrt(M) the
            // RMS-about-median scale is exactly outlier/sqrt(M) and the
            // sample sits on the clip boundary -- robust_combine's
            // documented behavior, pinned in its own tests).
            combine: CombineMethod::ClippedMean {
                kappa: 2.5,
                max_iter: 5,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
        };
        let wing_delta = Array2::<f64>::zeros((fluxes.len(), 2));

        let mut plus = noisy.clone();
        plus[outlier_pixel] += 5.0e4;
        let mut minus = noisy.clone();
        minus[outlier_pixel] -= 5.0e4;

        let built_plus = build_extended_psf(
            plus.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();
        let built_minus = build_extended_psf(
            minus.view(),
            None,
            wing_delta.view(),
            core.view(),
            oversample,
            params.clone(),
        )
        .unwrap();

        // (b) Sign-agnostic: +O and -O give the same recovered wing.
        for (a, b) in built_plus
            .extended
            .wing
            .iter()
            .zip(built_minus.extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-9 * a.abs().max(1.0),
                "sign-agnostic outlier rejection must propagate: {a} vs {b}"
            );
        }

        // (a) Orchestration correctness: the recovered wing equals a
        // hand-built (data - bg)/scale -> robust_combine -> stitch
        // reference using the orchestrator's own per-star bg/scale.
        let geometry = validate_core(&core.view(), oversample).unwrap();
        let reference_norm = Array3::<f64>::from_shape_fn(
            (fluxes.len(), wing_side, wing_side),
            |(star, r, c)| {
                (plus[(star, r, c)] - built_plus.star_background[star])
                    / built_plus.star_flux[star]
            },
        );
        let reference = robust_combine::<f64>(
            reference_norm.view(),
            None,
            params.combine,
        )
        .unwrap();
        // The orchestrator feeds robust_combine's weight plane as the
        // seam-match confidence; the reference must do the same.
        let reference_extended = stitch_core_and_wing(
            &core.view(),
            &geometry,
            &reference.combined.view(),
            Some(&reference.weight.view()),
            &params.stitch,
        );
        for (a, b) in built_plus
            .extended
            .wing
            .iter()
            .zip(reference_extended.wing.iter())
        {
            assert!(
                (a - b).abs() < 1e-6 * a.abs().max(1.0),
                "orchestrator wing must equal the hand-built reference: {a} vs {b}"
            );
        }
        // Every star calibrated via the core solve (clean central
        // windows); the table reports finite scales/backgrounds.
        assert!(built_plus.star_ok.iter().all(|&ok| ok));
        assert!(built_plus.star_scale_from_core.iter().all(|&v| v));
        assert!(built_plus.star_flux.iter().all(|f| f.is_finite() && *f > 0.0));
    }

    #[test]
    fn build_preconditions() {
        let oversample = 5;
        let stamp_size = 15;
        let core = gaussian_core(oversample, stamp_size, 1.2);
        let wing_side = 61usize;
        let data = Array3::<f64>::zeros((3, wing_side, wing_side));
        let delta = Array2::<f64>::zeros((3, 2));
        let good = ExtendedPsfParams {
            stitch: StitchParams {
                match_radius: 5.0,
                feather_width: 3.0,
                ee_aperture_radius: 18.0,
            },
            combine: CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
            scale_aperture_radius: 6.0,
            scale_background_annulus: (22.0, 28.0),
        };

        // OversampleNotOdd (incl. 0).
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                delta.view(),
                core.view(),
                4,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::OversampleNotOdd { oversample: 4 }
        );
        // CoreNotSquare.
        let rect = Array2::<f64>::zeros((10, 12));
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                delta.view(),
                rect.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::CoreNotSquare { rows: 10, cols: 12 }
        );
        // CoreSizeNotMultiple.
        let nm = Array2::<f64>::zeros((76, 76));
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                delta.view(),
                nm.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::CoreSizeNotMultiple {
                core_side: 76,
                oversample: 5
            }
        );
        // DerivedStampSizeEven.
        let es = Array2::<f64>::zeros((20, 20));
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                delta.view(),
                es.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::DerivedStampSizeEven { stamp_size: 4 }
        );
        // WingNotOdd.
        let even_wing = Array3::<f64>::zeros((3, 60, 61));
        assert_eq!(
            build_extended_psf(
                even_wing.view(),
                None,
                delta.view(),
                core.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::WingNotOdd { rows: 60, cols: 61 }
        );
        // BatchLengthMismatch: delta not (M, 2).
        let bad_delta = Array2::<f64>::zeros((2, 2));
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                bad_delta.view(),
                core.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::BatchLengthMismatch {
                wing_data: (3, wing_side, wing_side),
                wing_delta: (2, 2)
            }
        );
        // WeightShapeMismatch.
        let bad_weight = Array3::<f64>::zeros((3, wing_side, wing_side - 2));
        assert_eq!(
            build_extended_psf(
                data.view(),
                Some(bad_weight.view()),
                delta.view(),
                core.view(),
                5,
                good.clone()
            )
            .unwrap_err(),
            ExtendedPsfError::WeightShapeMismatch {
                weight: (3, wing_side, wing_side - 2),
                wing_data: (3, wing_side, wing_side)
            }
        );
        // ParamsInvalid: bad combine, bad aperture, bad annulus, bad
        // stitch geometry -- all one variant, after the shape ones.
        for bad in [
            ExtendedPsfParams {
                combine: CombineMethod::ClippedMean {
                    kappa: 0.0,
                    max_iter: 5,
                },
                ..good.clone()
            },
            ExtendedPsfParams {
                scale_aperture_radius: -1.0,
                ..good.clone()
            },
            ExtendedPsfParams {
                scale_background_annulus: (10.0, 5.0),
                ..good.clone()
            },
            ExtendedPsfParams {
                stitch: StitchParams {
                    match_radius: f64::NAN,
                    feather_width: 3.0,
                    ee_aperture_radius: 18.0,
                },
                ..good.clone()
            },
        ] {
            let err = build_extended_psf(
                data.view(),
                None,
                delta.view(),
                core.view(),
                5,
                bad.clone(),
            )
            .unwrap_err();
            // matches! (not assert_eq!): a NaN field makes the derived
            // PartialEq false even for an otherwise-identical struct.
            assert!(
                matches!(err, ExtendedPsfError::ParamsInvalid { .. }),
                "expected ParamsInvalid for {bad:?}, got {err:?}"
            );
        }
        // Shape precondition is checked before ParamsInvalid.
        assert_eq!(
            build_extended_psf(
                data.view(),
                None,
                delta.view(),
                es.view(),
                5,
                ExtendedPsfParams {
                    scale_aperture_radius: -1.0,
                    ..good.clone()
                }
            )
            .unwrap_err(),
            ExtendedPsfError::DerivedStampSizeEven { stamp_size: 4 }
        );
    }
}

//! Cross-stamp robust combination of an aligned native-resolution stack.
//!
//! [`robust_combine`] reduces a caller-aligned `(N, h, w)` stack of native
//! (un-oversampled) stamps along the stamp axis into a single `(h, w)`
//! image, rejecting outliers symmetrically by residual *magnitude*. It is
//! the workhorse for the extended-PSF wing/spike segment, which is built
//! by robustly stacking bright stars at native resolution (decision 3)
//! with sign-agnostic outlier rejection (decision 5).
//!
//! Design notes:
//!
//! - **Pure cross-N reducer (boundary).** This leaf takes a stack the
//!   caller has *already aligned* plus the *complete* per-pixel weight;
//!   alignment, per-stamp background subtraction and per-stamp scaling are
//!   not done here (they belong to the Phase 5/7 layers). This mirrors the
//!   `accumulate` boundary of "take the complete weight, compute no
//!   robust factor internally" and the decision-3 decomposition.
//!
//! - **Methods.** v1 ships [`CombineMethod::ClippedMean`] (symmetric
//!   sigma-clipped inverse-variance weighted mean) and
//!   [`CombineMethod::Median`]. A mode estimator is intentionally deferred
//!   to a later sub-step (estimator choice is a project of its own); the
//!   `CombineMethod` enum is the extension point reserved for it.
//!
//! - **Weight semantics.** For `Median` the weight is an *inclusion gate
//!   only* (`> 0` counts, `<= 0`/non-finite is excluded) -- there is no
//!   weighted quantile (decision 3 branch). Full inverse-variance
//!   weighting applies *only* in `ClippedMean`'s final mean (decision 7).
//!   `None` means equal weight (unit weight everywhere), matching the
//!   `Option` polarity of `build_stamp`/`render`/`accumulate`. mask
//!   polarity is the caller's job: a `true = invalid` pixel is folded into
//!   `weight = 0` upstream (decision 7); this leaf takes no mask of its
//!   own.
//!
//! - **Output.** A struct of three `(h, w)` planes: the combined value,
//!   the effective combined weight (sum of surviving sample weights;
//!   equals the survivor count when equal-weight), and the survivor count.
//!   Phase 7 (core/wing stitching, encircled-energy normalization) needs
//!   the per-pixel certainty and counts, not just the combined value.
//!
//! - **Data path, generic.** Unlike `render`/`accumulate` (whose `psi`
//!   model is always `f64`), `robust_combine` consumes native detector
//!   *data*, so it is generic over `T: Float` and dispatches f32/f64 the
//!   same way `build_stamp` does for its cutout. `weight` shares `T` with
//!   `stack` (the `error <-> cutout` pairing of `build_stamp`).
//!   Computation is in `f64`; outputs are `f64`/`u32`.
//!
//! - **Validity and per-pixel sentinel.** A sample is valid iff its value
//!   is finite AND (when a weight array is given) its weight is finite and
//!   strictly positive. A pixel whose every sample is invalid -- or whose
//!   survivors are all clipped away -- yields the per-pixel sentinel
//!   `combined = NaN, weight = 0, count = 0`. That is an *algorithmic*
//!   per-pixel outcome, not a global `Err` (the same spirit as the
//!   `reproject` NaN family).
//!
//! - **Empty shapes are legal.** `N = 0`, `h = 0` or `w = 0` are valid:
//!   an empty `N` gives an all-sentinel `(h, w)`, an empty spatial axis
//!   gives an empty output. None of these is an error (mirrors the
//!   `render`/`accumulate` empty-batch convention).
//!
//! - **Parallel over output pixels.** Each output pixel independently
//!   reduces over the `N` samples at that location and writes only its own
//!   cell, so there is no write contention (contrast `accumulate`, which
//!   shares one model buffer and needs a map-reduce).
//!
//! - **Errors are hard preconditions only.** Both [`RobustError`] variants
//!   are shape/parameter preconditions; there is no algorithmic `Ok(None)`
//!   path (mirroring `render`/`accumulate`, contrasting `build_stamp`).
//!   They are checked in declaration order: weight shape first, then the
//!   `ClippedMean` parameters.
//!
//! - **Landing decisions (sigma-clip internals; decision 5 compliant).**
//!   The clip center is the *unweighted* median of the current survivors
//!   and the clip scale is the root-mean-square deviation about that same
//!   median (std-vs-MAD resolved as RMS-about-median: the standard
//!   sigma-clip dispersion, exactly reproducible by a naive reference,
//!   with robustness coming from the median center plus iteration). A
//!   sample is rejected iff `|x - center| > kappa * scale` -- symmetric,
//!   by magnitude, *never* by sign (decision 5). The clip statistics are
//!   deliberately unweighted (rejection is about residual magnitude) while
//!   the final `ClippedMean` estimate is the inverse-variance weighted
//!   mean of the survivors (decision 7). The iteration stops when a pass
//!   removes nothing (survivors only ever shrink, so "removed nothing" is
//!   exactly convergence) or `max_iter` is reached. `sigma = 0` (all
//!   survivors equal, including the single-survivor / `N = 1` case) gives
//!   threshold `0` with `|x - center| = 0`, so nothing is rejected and the
//!   pass is stable. For an odd survivor count the median sample always
//!   has `|x - center| = 0`, so a pass can only empty the set for a
//!   pathologically small `kappa` with an even count; that yields the same
//!   sentinel as the all-invalid case. `count` is accumulated as `usize`
//!   and returned as `u32`. The stack/weight are upcast per sample via
//!   `T::to_f64()` rather than copied into full `f64` arrays, since the
//!   stack is the large 3-D input. Output planes are assembled row-major.

use ndarray::{Array2, ArrayView3};
use rayon::prelude::*;
use thiserror::Error;

use crate::float::Float;
use crate::image::stats::median_in_place;

/// How [`robust_combine`] reduces the stamp axis.
///
/// `Median` ignores the weight magnitude (it is only an inclusion gate);
/// `ClippedMean` uses the full inverse-variance weight in its final mean.
/// A mode estimator is deliberately left to a later sub-step -- this enum
/// is the reserved extension point for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombineMethod {
    /// Symmetric sigma-clipped, inverse-variance weighted mean. `kappa` is
    /// the clip threshold in robust sigmas; `max_iter` caps the clip
    /// iteration. Both must be positive (see [`RobustError`]).
    ClippedMean { kappa: f64, max_iter: usize },
    /// Unweighted median of the samples that pass the inclusion gate
    /// (`weight > 0`, or all samples when `weight` is `None`).
    Median,
}

/// Output of [`robust_combine`]; three `(h, w)` planes.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustCombined {
    /// Combined value per pixel. A pixel with no survivors carries the
    /// sentinel `NaN`.
    pub combined: Array2<f64>,
    /// Effective combined weight: the sum of the surviving samples'
    /// weights. Equals `count` (as `f64`) when `weight` is `None`. `0.0`
    /// for a no-survivor pixel.
    pub weight: Array2<f64>,
    /// Number of surviving (valid and, for `ClippedMean`, un-clipped)
    /// samples per pixel. `0` for a no-survivor pixel.
    pub count: Array2<u32>,
}

/// Errors returned by [`robust_combine`] for ill-shaped /
/// ill-parameterized inputs.
///
/// Both variants are hard preconditions. Like `render`/`accumulate` (and
/// unlike `build_stamp`), `robust_combine` has no algorithmic skip path: a
/// well-formed call always yields `(h, w)` planes (a no-survivor pixel is
/// handled per-pixel with the `NaN`/`0`/`0` sentinel, not an `Err`).
#[derive(Debug, Error, PartialEq)]
pub enum RobustError {
    #[error("weight shape {weight:?} must equal stack shape {stack:?}")]
    WeightShapeMismatch {
        weight: (usize, usize, usize),
        stack: (usize, usize, usize),
    },
    #[error(
        "ClippedMean requires kappa > 0 and max_iter > 0; got kappa = {kappa}, max_iter = {max_iter}"
    )]
    ClippedMeanInvalidParams { kappa: f64, max_iter: usize },
}

/// Robustly combine an aligned `(N, h, w)` stack along the stamp axis.
///
/// See the module-level documentation for the reducer boundary, the
/// weight/validity semantics, the per-pixel sentinel, and the sigma-clip
/// landing decisions.
///
/// # Parameters
///
/// - `stack`: `(N, h, w)` caller-aligned native stamp stack. May be `f32`
///   or `f64`; upcast to `f64` internally. Non-finite samples are
///   excluded.
/// - `weight`: optional `(N, h, w)` complete per-pixel weight (`valid *
///   1/sigma^2 * ...`, formed by the caller). `None` is equal weight. A
///   sample with non-finite or non-positive weight is excluded.
/// - `method`: see [`CombineMethod`].
///
/// # Returns
///
/// `Ok(RobustCombined)` with three `(h, w)` planes. A pixel with no
/// surviving sample carries `combined = NaN`, `weight = 0`, `count = 0`.
/// `N = 0` yields an all-sentinel `(h, w)`; `h = 0` or `w = 0` yields an
/// empty output.
///
/// # Errors
///
/// Returns a [`RobustError`] if `weight` is `Some` with a shape different
/// from `stack`, or if `method` is `ClippedMean` with `kappa <= 0` or
/// `max_iter == 0`.
pub fn robust_combine<T: Float>(
    stack: ArrayView3<T>,
    weight: Option<ArrayView3<T>>,
    method: CombineMethod,
) -> Result<RobustCombined, RobustError> {
    let sample_count = stack.shape()[0];
    let height = stack.shape()[1];
    let width = stack.shape()[2];

    // --- Hard preconditions (Err only; no Ok(None)). The check order is
    // the documented order: weight shape, then ClippedMean parameters. ---
    if let Some(weight_view) = weight {
        let weight_shape = weight_view.shape();
        if weight_shape[0] != sample_count || weight_shape[1] != height || weight_shape[2] != width
        {
            return Err(RobustError::WeightShapeMismatch {
                weight: (weight_shape[0], weight_shape[1], weight_shape[2]),
                stack: (sample_count, height, width),
            });
        }
    }
    if let CombineMethod::ClippedMean { kappa, max_iter } = method {
        let params_invalid = kappa <= 0.0 || max_iter == 0;
        if params_invalid {
            return Err(RobustError::ClippedMeanInvalidParams { kappa, max_iter });
        }
    }

    // Parallel over output pixels: each (row, column) independently
    // reduces over the N samples at that location and produces its own
    // (combined, weight, count) triple. Distinct output cells => no write
    // contention (contrast accumulate's shared-buffer map-reduce). Pixels
    // are walked row-major, so the flat index is `row * width + column`.
    // `pixel_count == 0` (h == 0 or w == 0) makes the range empty, so the
    // `% width` / `/ width` below never divide by zero.
    let pixel_count = height * width;
    let per_pixel: Vec<(f64, f64, u32)> = (0..pixel_count)
        .into_par_iter()
        .map(|flat_index| {
            let row = flat_index / width;
            let column = flat_index % width;
            combine_one_pixel(&stack, weight.as_ref(), row, column, sample_count, method)
        })
        .collect();

    let mut combined_values = Vec::with_capacity(pixel_count);
    let mut weight_values = Vec::with_capacity(pixel_count);
    let mut count_values = Vec::with_capacity(pixel_count);
    for (combined, combined_weight, survivors) in per_pixel {
        combined_values.push(combined);
        weight_values.push(combined_weight);
        count_values.push(survivors);
    }

    Ok(RobustCombined {
        combined: Array2::from_shape_vec((height, width), combined_values)
            .expect("combined length == height * width"),
        weight: Array2::from_shape_vec((height, width), weight_values)
            .expect("weight length == height * width"),
        count: Array2::from_shape_vec((height, width), count_values)
            .expect("count length == height * width"),
    })
}

/// Reduce the `N` samples at one output pixel into `(combined, weight,
/// count)`.
///
/// Gathers the valid samples for this `(row, column)` across the stamp
/// axis, then dispatches to the chosen estimator. A pixel with no valid
/// sample short-circuits to the `NaN`/`0`/`0` sentinel.
fn combine_one_pixel<T: Float>(
    stack: &ArrayView3<T>,
    weight: Option<&ArrayView3<T>>,
    row: usize,
    column: usize,
    sample_count: usize,
    method: CombineMethod,
) -> (f64, f64, u32) {
    // Validity (spec-derived): the value is finite AND, when a weight
    // array is given, its weight is finite and strictly positive. The
    // caller folds mask (`true = invalid`) into `weight = 0` upstream
    // (decision 7); this leaf takes no mask of its own. Equal weight
    // (`weight == None`) uses unit weights, so the gate is finiteness
    // alone.
    let mut sample_values: Vec<f64> = Vec::with_capacity(sample_count);
    let mut sample_weights: Vec<f64> = Vec::with_capacity(sample_count);
    for n in 0..sample_count {
        let value = stack[(n, row, column)].to_f64().unwrap_or(f64::NAN);
        if !value.is_finite() {
            continue;
        }
        let pixel_weight = match weight {
            Some(weight_view) => {
                let w = weight_view[(n, row, column)].to_f64().unwrap_or(f64::NAN);
                if !(w.is_finite() && w > 0.0) {
                    continue;
                }
                w
            }
            None => 1.0,
        };
        sample_values.push(value);
        sample_weights.push(pixel_weight);
    }

    if sample_values.is_empty() {
        // Every sample invalid for this pixel: per-pixel sentinel
        // (algorithmic, NOT a global Err).
        return (f64::NAN, 0.0, 0);
    }

    match method {
        CombineMethod::Median => combine_median(&sample_values, &sample_weights),
        CombineMethod::ClippedMean { kappa, max_iter } => {
            combine_clipped_mean(&sample_values, &sample_weights, kappa, max_iter)
        }
    }
}

/// Unweighted median of the valid samples. Weight is an inclusion gate
/// only (decision 3 branch): every valid sample is a survivor, the
/// combined value is the *unweighted* median, and the reported weight is
/// the sum of the valid samples' weights (= count when equal-weight).
fn combine_median(values: &[f64], weights: &[f64]) -> (f64, f64, u32) {
    let mut sorted = values.to_vec();
    let combined = median_in_place(&mut sorted).unwrap_or(f64::NAN);
    let weight_sum: f64 = weights.iter().sum();
    (combined, weight_sum, values.len() as u32)
}

/// Symmetric sigma-clipped, inverse-variance weighted mean.
///
/// Iterates: take the unweighted median `center` of the current
/// survivors, the RMS deviation `scale` about that median, and reject any
/// survivor with `|x - center| > kappa * scale` (symmetric, by magnitude,
/// never by sign -- decision 5). Survivors only shrink, so a pass that
/// removes nothing is exactly convergence. The final estimate is the
/// inverse-variance weighted mean of the survivors (decision 7).
fn combine_clipped_mean(
    values: &[f64],
    weights: &[f64],
    kappa: f64,
    max_iter: usize,
) -> (f64, f64, u32) {
    let mut alive = vec![true; values.len()];

    // `max_iter >= 1` is guaranteed by the ClippedMeanInvalidParams
    // precondition.
    for _iteration in 0..max_iter {
        let survivors: Vec<f64> = (0..values.len())
            .filter(|&index| alive[index])
            .map(|index| values[index])
            .collect();
        if survivors.is_empty() {
            break;
        }

        // Clip center: unweighted median of the survivors. A robust
        // center stops the rejection from chasing the very outliers it
        // should remove; the clip statistics are deliberately unweighted
        // because rejection is by residual MAGNITUDE (decision 5), while
        // the final estimate below is inverse-variance weighted
        // (decision 7).
        let mut sorted = survivors.clone();
        let center = median_in_place(&mut sorted).unwrap_or(f64::NAN);

        // Clip scale: root-mean-square deviation about that same median
        // (self-consistent with the `|x - center|` test below).
        let mean_square: f64 = survivors
            .iter()
            .map(|&x| (x - center) * (x - center))
            .sum::<f64>()
            / survivors.len() as f64;
        let scale = mean_square.sqrt();
        let threshold = kappa * scale;

        // Symmetric, sign-agnostic rejection: reject iff the ABSOLUTE
        // deviation exceeds the threshold. `scale == 0` (all survivors
        // equal, incl. the single-survivor / N = 1 case) gives
        // `threshold == 0` and `|x - center| == 0`, so nothing is removed
        // and the pass is stable. Survivors only ever shrink, so "removed
        // nothing this pass" is exactly convergence.
        let mut removed_this_pass = false;
        for index in 0..values.len() {
            if alive[index] && (values[index] - center).abs() > threshold {
                alive[index] = false;
                removed_this_pass = true;
            }
        }
        if !removed_this_pass {
            break;
        }
    }

    // Inverse-variance weighted mean of the survivors. Every surviving
    // weight is strictly positive (inclusion gate; unit when
    // `weight == None`), so `weight_sum > 0` whenever there is a survivor.
    let mut weighted_value_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut survivor_count: usize = 0;
    for index in 0..values.len() {
        if alive[index] {
            weighted_value_sum += weights[index] * values[index];
            weight_sum += weights[index];
            survivor_count += 1;
        }
    }
    if survivor_count == 0 {
        // Pathologically small kappa with an even count can clip every
        // sample; same sentinel as the all-invalid case.
        return (f64::NAN, 0.0, 0);
    }
    (
        weighted_value_sum / weight_sum,
        weight_sum,
        survivor_count as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// SplitMix64: a tiny, dependency-free, fully deterministic PRNG so
    /// the random production-vs-naive comparisons are reproducible (same
    /// generator as the `accumulate` tests).
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

    // --- Naive reference implementations (the correctness oracle). ---

    /// Naive per-pixel valid-sample gather: returns `(values, weights)`
    /// with the exact validity gate of the production code.
    fn naive_gather(
        stack: &Array3<f64>,
        weight: Option<&Array3<f64>>,
        row: usize,
        column: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut values = Vec::new();
        let mut weights = Vec::new();
        for n in 0..stack.shape()[0] {
            let value = stack[(n, row, column)];
            if !value.is_finite() {
                continue;
            }
            let w = match weight {
                Some(weight_array) => {
                    let w = weight_array[(n, row, column)];
                    if !(w.is_finite() && w > 0.0) {
                        continue;
                    }
                    w
                }
                None => 1.0,
            };
            values.push(value);
            weights.push(w);
        }
        (values, weights)
    }

    fn naive_median(values: &mut [f64]) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let length = values.len();
        if length % 2 == 1 {
            values[length / 2]
        } else {
            0.5 * (values[length / 2 - 1] + values[length / 2])
        }
    }

    /// Naive `Median` combine for the whole stack.
    fn naive_median_combine(
        stack: &Array3<f64>,
        weight: Option<&Array3<f64>>,
    ) -> (Array2<f64>, Array2<f64>, Array2<u32>) {
        let (_, height, width) = (stack.shape()[0], stack.shape()[1], stack.shape()[2]);
        let mut combined = Array2::<f64>::zeros((height, width));
        let mut weight_out = Array2::<f64>::zeros((height, width));
        let mut count = Array2::<u32>::zeros((height, width));
        for row in 0..height {
            for column in 0..width {
                let (mut values, weights) = naive_gather(stack, weight, row, column);
                if values.is_empty() {
                    combined[(row, column)] = f64::NAN;
                    weight_out[(row, column)] = 0.0;
                    count[(row, column)] = 0;
                    continue;
                }
                count[(row, column)] = values.len() as u32;
                weight_out[(row, column)] = weights.iter().sum();
                combined[(row, column)] = naive_median(&mut values);
            }
        }
        (combined, weight_out, count)
    }

    /// Naive symmetric sigma-clip + inverse-variance weighted mean,
    /// transcribing the exact algorithm the production code implements.
    fn naive_clipped_mean_combine(
        stack: &Array3<f64>,
        weight: Option<&Array3<f64>>,
        kappa: f64,
        max_iter: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<u32>) {
        let (_, height, width) = (stack.shape()[0], stack.shape()[1], stack.shape()[2]);
        let mut combined = Array2::<f64>::zeros((height, width));
        let mut weight_out = Array2::<f64>::zeros((height, width));
        let mut count = Array2::<u32>::zeros((height, width));
        for row in 0..height {
            for column in 0..width {
                let (values, weights) = naive_gather(stack, weight, row, column);
                if values.is_empty() {
                    combined[(row, column)] = f64::NAN;
                    continue;
                }
                let mut alive = vec![true; values.len()];
                for _ in 0..max_iter {
                    let survivors: Vec<f64> = (0..values.len())
                        .filter(|&i| alive[i])
                        .map(|i| values[i])
                        .collect();
                    if survivors.is_empty() {
                        break;
                    }
                    let mut sorted = survivors.clone();
                    let center = naive_median(&mut sorted);
                    let mean_square: f64 = survivors
                        .iter()
                        .map(|&x| (x - center) * (x - center))
                        .sum::<f64>()
                        / survivors.len() as f64;
                    let threshold = kappa * mean_square.sqrt();
                    let mut removed = false;
                    for i in 0..values.len() {
                        if alive[i] && (values[i] - center).abs() > threshold {
                            alive[i] = false;
                            removed = true;
                        }
                    }
                    if !removed {
                        break;
                    }
                }
                let mut wv = 0.0;
                let mut ws = 0.0;
                let mut c: u32 = 0;
                for i in 0..values.len() {
                    if alive[i] {
                        wv += weights[i] * values[i];
                        ws += weights[i];
                        c += 1;
                    }
                }
                if c == 0 {
                    combined[(row, column)] = f64::NAN;
                    weight_out[(row, column)] = 0.0;
                    count[(row, column)] = 0;
                } else {
                    combined[(row, column)] = wv / ws;
                    weight_out[(row, column)] = ws;
                    count[(row, column)] = c;
                }
            }
        }
        (combined, weight_out, count)
    }

    fn assert_planes_close(
        got: &RobustCombined,
        want: &(Array2<f64>, Array2<f64>, Array2<u32>),
        tol: f64,
    ) {
        let (want_combined, want_weight, want_count) = want;
        for ((g, w), _) in got.combined.iter().zip(want_combined.iter()).zip(0..) {
            if w.is_nan() {
                assert!(g.is_nan(), "expected NaN sentinel, got {g}");
            } else {
                assert!(
                    (g - w).abs() <= tol * w.abs().max(1.0),
                    "combined {g} != {w}"
                );
            }
        }
        for (g, w) in got.weight.iter().zip(want_weight.iter()) {
            assert!((g - w).abs() <= tol * w.abs().max(1.0), "weight {g} != {w}");
        }
        for (g, w) in got.count.iter().zip(want_count.iter()) {
            assert_eq!(g, w, "count {g} != {w}");
        }
    }

    fn random_stack(rng: &mut SplitMix64, n: usize, h: usize, w: usize) -> Array3<f64> {
        // Includes negative samples on purpose (decision 5: negatives are
        // never dropped by sign).
        Array3::from_shape_fn((n, h, w), |_| rng.range(-5.0, 5.0))
    }

    // --- Random production-vs-naive comparisons. ---

    #[test]
    fn median_matches_naive_reference_equal_weight() {
        let mut rng = SplitMix64::new(0xC0FF_EE00_1234_5678);
        let stack = random_stack(&mut rng, 9, 6, 7);
        let got = robust_combine(stack.view(), None, CombineMethod::Median).unwrap();
        let want = naive_median_combine(&stack, None);
        assert_planes_close(&got, &want, 1e-12);
    }

    #[test]
    fn median_matches_naive_reference_weighted_gate() {
        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        let stack = random_stack(&mut rng, 11, 5, 4);
        // Weights in [-0.5, 2): negatives/zeros exercise the inclusion
        // gate (excluded), positives are kept.
        let weight = Array3::from_shape_fn((11, 5, 4), |_| rng.range(-0.5, 2.0));
        let got = robust_combine(stack.view(), Some(weight.view()), CombineMethod::Median).unwrap();
        let want = naive_median_combine(&stack, Some(&weight));
        assert_planes_close(&got, &want, 1e-12);
    }

    #[test]
    fn clipped_mean_matches_naive_reference_equal_weight() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF_F00D_BABE);
        let stack = random_stack(&mut rng, 15, 5, 6);
        let method = CombineMethod::ClippedMean {
            kappa: 2.5,
            max_iter: 5,
        };
        let got = robust_combine(stack.view(), None, method).unwrap();
        let want = naive_clipped_mean_combine(&stack, None, 2.5, 5);
        assert_planes_close(&got, &want, 1e-9);
    }

    #[test]
    fn clipped_mean_matches_naive_reference_weighted() {
        let mut rng = SplitMix64::new(0x0BAD_C0DE_1234_5678);
        let stack = random_stack(&mut rng, 13, 4, 5);
        let weight = Array3::from_shape_fn((13, 4, 5), |_| rng.range(0.1, 3.0));
        let method = CombineMethod::ClippedMean {
            kappa: 3.0,
            max_iter: 8,
        };
        let got = robust_combine(stack.view(), Some(weight.view()), method).unwrap();
        let want = naive_clipped_mean_combine(&stack, Some(&weight), 3.0, 8);
        assert_planes_close(&got, &want, 1e-9);
    }

    // --- Hand-computed small cases pinning the semantics. ---

    #[test]
    fn weight_zero_sample_excluded_by_inclusion_gate() {
        // One pixel, 3 samples. The middle sample has weight 0 and must be
        // excluded from BOTH the median and the count/weight.
        let stack = Array3::from_shape_vec((3, 1, 1), vec![1.0, 999.0, 3.0]).unwrap();
        let weight = Array3::from_shape_vec((3, 1, 1), vec![2.0, 0.0, 4.0]).unwrap();
        let got = robust_combine(stack.view(), Some(weight.view()), CombineMethod::Median).unwrap();
        // Only {1.0, 3.0} survive the gate -> median 2.0, count 2,
        // weight 2 + 4 = 6. The 999.0 weight-0 sample is gone.
        assert!((got.combined[(0, 0)] - 2.0).abs() < 1e-12);
        assert_eq!(got.count[(0, 0)], 2);
        assert!((got.weight[(0, 0)] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn all_invalid_pixel_is_nan_weight0_count0() {
        // Pixel (0,0): all NaN values. Pixel (0,1): all weight 0.
        let stack = Array3::from_shape_vec((2, 1, 2), vec![f64::NAN, 5.0, f64::NAN, 7.0]).unwrap();
        let weight = Array3::from_shape_vec((2, 1, 2), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        for method in [
            CombineMethod::Median,
            CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 5,
            },
        ] {
            let got = robust_combine(stack.view(), Some(weight.view()), method).unwrap();
            for column in 0..2 {
                assert!(
                    got.combined[(0, column)].is_nan(),
                    "expected NaN sentinel at column {column}"
                );
                assert_eq!(got.weight[(0, column)], 0.0);
                assert_eq!(got.count[(0, column)], 0);
            }
        }
    }

    #[test]
    fn clipped_mean_rejects_outlier_sign_agnostically() {
        // 20 symmetric inliers (median 0, modest spread) plus one large
        // outlier. Pixel 0 gets +50, pixel 1 gets -50. A sign-agnostic
        // clip must reject both identically, leaving the same inliers and
        // hence the same combined value and count.
        let inlier_pattern = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = inlier_pattern.len() * 4 + 1; // 21 samples
        let mut data = Vec::with_capacity(n * 2);
        // Layout is (N, h=1, w=2): for each n push pixel0 then pixel1.
        for sample_index in 0..n {
            let (v0, v1) = if sample_index < n - 1 {
                let v = inlier_pattern[sample_index % inlier_pattern.len()];
                (v, v)
            } else {
                (50.0, -50.0) // the single outlier, opposite signs
            };
            data.push(v0);
            data.push(v1);
        }
        let stack = Array3::from_shape_vec((n, 1, 2), data).unwrap();
        let method = CombineMethod::ClippedMean {
            kappa: 3.0,
            max_iter: 5,
        };
        let got = robust_combine(stack.view(), None, method).unwrap();

        // Both the +50 and the -50 outlier are rejected; the surviving
        // inliers are identical for the two pixels.
        assert_eq!(got.count[(0, 0)], 20);
        assert_eq!(got.count[(0, 1)], 20);
        assert!(
            (got.combined[(0, 0)] - got.combined[(0, 1)]).abs() < 1e-12,
            "sign-agnostic clip must give identical results: {} vs {}",
            got.combined[(0, 0)],
            got.combined[(0, 1)]
        );
        // The inlier mean is exactly 0.
        assert!(got.combined[(0, 0)].abs() < 1e-12);
        // Cross-check against the naive reference too.
        let want = naive_clipped_mean_combine(&stack, None, 3.0, 5);
        assert_planes_close(&got, &want, 1e-9);
    }

    #[test]
    fn clipped_mean_huge_kappa_is_weighted_mean() {
        // With kappa enormous nothing is ever clipped, so ClippedMean
        // degenerates to the plain inverse-variance weighted mean of all
        // valid samples.
        let mut rng = SplitMix64::new(0xABCD_1234_5678_9F00);
        let stack = random_stack(&mut rng, 7, 3, 3);
        let weight = Array3::from_shape_fn((7, 3, 3), |_| rng.range(0.2, 2.0));
        let got = robust_combine(
            stack.view(),
            Some(weight.view()),
            CombineMethod::ClippedMean {
                kappa: 1e30,
                max_iter: 10,
            },
        )
        .unwrap();
        for row in 0..3 {
            for column in 0..3 {
                let mut wv = 0.0;
                let mut ws = 0.0;
                for n in 0..7 {
                    wv += weight[(n, row, column)] * stack[(n, row, column)];
                    ws += weight[(n, row, column)];
                }
                assert!(
                    (got.combined[(row, column)] - wv / ws).abs() < 1e-9 * (wv / ws).abs().max(1.0),
                    "expected plain weighted mean"
                );
                assert_eq!(got.count[(row, column)], 7);
                assert!((got.weight[(row, column)] - ws).abs() < 1e-9 * ws.max(1.0));
            }
        }
    }

    #[test]
    fn single_stamp_passes_through() {
        // N = 1: every pixel has exactly one sample, which must pass
        // through unchanged for both methods.
        let stack = Array3::from_shape_vec((1, 2, 2), vec![3.0, -7.0, 0.5, 11.0]).unwrap();
        for method in [
            CombineMethod::Median,
            CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 4,
            },
        ] {
            let got = robust_combine(stack.view(), None, method).unwrap();
            for (g, s) in got.combined.iter().zip(stack.iter()) {
                assert!((g - s).abs() < 1e-12, "N=1 passthrough: {g} != {s}");
            }
            assert!(got.count.iter().all(|&c| c == 1));
            assert!(got.weight.iter().all(|&w| (w - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn empty_n_yields_all_sentinel_plane() {
        // N = 0 is legal: an all-sentinel (h, w) plane, not an error.
        let stack = Array3::<f64>::zeros((0, 3, 4));
        for method in [
            CombineMethod::Median,
            CombineMethod::ClippedMean {
                kappa: 2.0,
                max_iter: 3,
            },
        ] {
            let got = robust_combine(stack.view(), None, method).unwrap();
            assert_eq!(got.combined.shape(), &[3, 4]);
            assert!(got.combined.iter().all(|v| v.is_nan()));
            assert!(got.weight.iter().all(|&w| w == 0.0));
            assert!(got.count.iter().all(|&c| c == 0));
        }
    }

    #[test]
    fn empty_spatial_axis_yields_empty_output() {
        // h = 0 or w = 0 is legal: an empty output, not an error, and no
        // divide-by-zero in the row-major flatten.
        let stack = Array3::<f64>::zeros((4, 0, 5));
        let got = robust_combine(stack.view(), None, CombineMethod::Median).unwrap();
        assert_eq!(got.combined.shape(), &[0, 5]);

        let stack = Array3::<f64>::zeros((4, 5, 0));
        let got = robust_combine(
            stack.view(),
            None,
            CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 2,
            },
        )
        .unwrap();
        assert_eq!(got.combined.shape(), &[5, 0]);
    }

    #[test]
    fn count_and_weight_semantics() {
        // Equal weight -> reported weight == count (as f64). Weighted ->
        // reported weight == sum of survivor weights.
        let stack = Array3::from_shape_vec((4, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let equal = robust_combine(stack.view(), None, CombineMethod::Median).unwrap();
        assert_eq!(equal.count[(0, 0)], 4);
        assert!((equal.weight[(0, 0)] - 4.0).abs() < 1e-12);

        let weight = Array3::from_shape_vec((4, 1, 1), vec![0.5, 1.5, 2.0, 1.0]).unwrap();
        let weighted =
            robust_combine(stack.view(), Some(weight.view()), CombineMethod::Median).unwrap();
        assert_eq!(weighted.count[(0, 0)], 4);
        assert!((weighted.weight[(0, 0)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn f32_and_f64_dual_path_agree() {
        // Same data as f64 and f32 must give the same combined values
        // (within f32 precision); exercises the Float dispatch.
        let mut rng = SplitMix64::new(0xFEED_FACE_CAFE_0001);
        let stack_f64 = random_stack(&mut rng, 9, 4, 4);
        let stack_f32: Array3<f32> = stack_f64.mapv(|v| v as f32);

        let method = CombineMethod::ClippedMean {
            kappa: 2.5,
            max_iter: 5,
        };
        let from_f64 = robust_combine(stack_f64.view(), None, method).unwrap();
        let from_f32 = robust_combine(stack_f32.view(), None, method).unwrap();

        for (a, b) in from_f64.combined.iter().zip(from_f32.combined.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-4 * a.abs().max(1.0), "{a} vs {b}");
            }
        }
        assert_eq!(from_f64.count, from_f32.count);

        // Median path too.
        let mf64 = robust_combine(stack_f64.view(), None, CombineMethod::Median).unwrap();
        let mf32 = robust_combine(stack_f32.view(), None, CombineMethod::Median).unwrap();
        for (a, b) in mf64.combined.iter().zip(mf32.combined.iter()) {
            assert!((a - b).abs() < 1e-4 * a.abs().max(1.0), "{a} vs {b}");
        }
    }

    // --- Hard preconditions. ---

    #[test]
    fn error_weight_shape_mismatch() {
        let stack = Array3::<f64>::zeros((3, 4, 5));
        let weight = Array3::<f64>::zeros((3, 4, 6));
        let err =
            robust_combine(stack.view(), Some(weight.view()), CombineMethod::Median).unwrap_err();
        assert_eq!(
            err,
            RobustError::WeightShapeMismatch {
                weight: (3, 4, 6),
                stack: (3, 4, 5),
            }
        );
    }

    #[test]
    fn error_clipped_mean_invalid_kappa() {
        let stack = Array3::<f64>::zeros((3, 2, 2));
        for bad_kappa in [0.0, -1.0] {
            let err = robust_combine(
                stack.view(),
                None,
                CombineMethod::ClippedMean {
                    kappa: bad_kappa,
                    max_iter: 5,
                },
            )
            .unwrap_err();
            assert_eq!(
                err,
                RobustError::ClippedMeanInvalidParams {
                    kappa: bad_kappa,
                    max_iter: 5,
                }
            );
        }
    }

    #[test]
    fn error_clipped_mean_zero_max_iter() {
        let stack = Array3::<f64>::zeros((3, 2, 2));
        let err = robust_combine(
            stack.view(),
            None,
            CombineMethod::ClippedMean {
                kappa: 3.0,
                max_iter: 0,
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            RobustError::ClippedMeanInvalidParams {
                kappa: 3.0,
                max_iter: 0,
            }
        );
    }

    #[test]
    fn error_weight_shape_checked_before_clipped_mean_params() {
        // Both preconditions fail; the documented order checks the weight
        // shape first.
        let stack = Array3::<f64>::zeros((2, 2, 2));
        let weight = Array3::<f64>::zeros((2, 2, 3));
        let err = robust_combine(
            stack.view(),
            Some(weight.view()),
            CombineMethod::ClippedMean {
                kappa: -1.0,
                max_iter: 0,
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            RobustError::WeightShapeMismatch {
                weight: (2, 2, 3),
                stack: (2, 2, 2),
            }
        );
    }

    #[test]
    fn median_no_extra_precondition() {
        // Median has no parameters, so it never hits
        // ClippedMeanInvalidParams even with a degenerate stack.
        let stack = Array3::<f64>::zeros((1, 1, 1));
        let got = robust_combine(stack.view(), None, CombineMethod::Median).unwrap();
        assert_eq!(got.combined[(0, 0)], 0.0);
        assert_eq!(got.count[(0, 0)], 1);
    }
}

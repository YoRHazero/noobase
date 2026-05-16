//! Adjoint of the forward PSF model: (weighted) detector-grid residuals
//! back-projected onto the oversampled effective-PSF grid.
//!
//! [`accumulate`] is the *exact transpose* of the model-linear part of
//! [`super::render`]. Writing the forward stamp as
//! `model = flux * S(epsf) + background`, where `S` is the
//! Catmull-Rom point-sampling operator (linear in `epsf`), the linear map
//! pinned here is `A: epsf |-> flux * S(epsf)`. Its transpose `A^T`
//! scatters a detector-grid field back onto the model grid; with the
//! inner products taken over the full `(N, s, s)` and `(os*s, os*s)`
//! arrays the defining identity is
//!
//! ```text
//!   <render(epsf; delta, flux, b = 0), r>  ==  <epsf, accumulate(r; delta, flux)>
//! ```
//!
//! and, with a per-pixel weight `W`,
//!
//! ```text
//!   <render(epsf; delta, flux, b = 0), W . r>  ==  <epsf, accumulate(r, Some(W); delta, flux)>.
//! ```
//!
//! This is Irani-Peleg back-projection: the residual update direction the
//! super-resolution solver scatters into the model. Because the operator
//! is linear in `epsf`, the analytic adjoint *is* the gradient operator;
//! no autodiff is involved (decision 10).
//!
//! Design notes:
//!
//! - **Exact transpose by shared weights.** `render` gathers
//!   `value += epsf[ru, rv] * w_u * w_v`; `accumulate` scatters
//!   `epsf_acc[ru, rv] += coeff * w_u * w_v` using the *same*
//!   [`super::kernel::catmull_rom_weights`], the same `floor`-derived tap
//!   base, and the same zero-padding (taps outside `[0, os*s)` are
//!   dropped -- the transpose of zero padding is zero padding). Sharing
//!   one kernel module is what makes the identity hold to floating-point
//!   round-off rather than by coincidence.
//!
//! - **Operator composition.** `flux` is part of the operator, so its
//!   transpose carries `diag(flux)` (decision 10). `weight` is part of
//!   the operator too: it is the *complete* per-pixel weight
//!   (`valid * 1/sigma^2 * robust factor`) the solver has already formed;
//!   `accumulate` never computes a robust factor itself (decision 7).
//!   `None` means unit weight everywhere, matching the `Option` polarity
//!   of `build_stamp`/`render`. The constant `background` is *not* part of
//!   the model-linear map `A`, so it does not participate (the transpose
//!   identity is stated at `b = 0`); `residual` and `weight` are kept as
//!   two parameters so the scatter fuses into a single pass with no
//!   intermediate `W . r` array and `weight` stays optional.
//!
//! - **Grid is given, not derived.** `render` recovers `s` from
//!   `epsf.shape()` and `os`; here `epsf` is the *output*, so the model
//!   grid is specified by `(oversample, stamp_size)` and the result is a
//!   single `(os*s, os*s)` array summed over the whole batch.
//!
//! - **Errors are hard preconditions only.** Every [`AccumulateError`] is
//!   a shape/parity precondition; there is no algorithmic `Ok(None)` path
//!   (mirroring `render`, contrasting `build_stamp`). An empty batch
//!   (`N == 0`) is valid and yields an all-zero `(os*s, os*s)` array.
//!
//! - **Parallel reduction.** Unlike `render`, where each stamp owns a
//!   disjoint output slice, every stamp here scatters into the *same*
//!   model grid and overlapping 4x4 patches would race. The batch is
//!   therefore folded in parallel into per-worker private accumulators
//!   that are then summed (a map-reduce), which keeps the result
//!   independent of the worker count and bounds the extra memory by the
//!   thread count rather than by `N`.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use thiserror::Error;

use super::kernel::catmull_rom_weights;

/// Errors returned by [`accumulate`] for ill-shaped / ill-parameterized
/// inputs.
///
/// All variants are hard preconditions. Like `render` (and unlike
/// `build_stamp`), `accumulate` has no algorithmic skip path: a
/// well-formed call always yields an `(os*s, os*s)` array.
#[derive(Debug, Error, PartialEq)]
pub enum AccumulateError {
    #[error("oversample must be odd; got {oversample}")]
    OversampleNotOdd { oversample: usize },
    #[error("stamp_size must be odd; got {stamp_size}")]
    StampSizeEven { stamp_size: usize },
    #[error(
        "residual shape {residual:?} must be (N, stamp_size, stamp_size) = {expected:?}"
    )]
    ResidualShapeMismatch {
        residual: (usize, usize, usize),
        expected: (usize, usize, usize),
    },
    #[error("weight shape {weight:?} must equal residual shape {residual:?}")]
    WeightShapeMismatch {
        weight: (usize, usize, usize),
        residual: (usize, usize, usize),
    },
    #[error(
        "batch dimensions disagree: delta shape {delta:?} must be (N, 2) with N == residual N ({residual}) == flux len ({flux})"
    )]
    BatchLengthMismatch {
        delta: (usize, usize),
        residual: usize,
        flux: usize,
    },
}

/// Back-project a batch of (weighted) detector-grid residuals onto the
/// oversampled effective-PSF grid: the exact transpose of the
/// model-linear part of [`super::render`].
///
/// See the module-level documentation for the transpose identity, the
/// operator composition (`flux`/`weight` inside, `background` excluded),
/// and the shared-kernel / zero-padding rationale.
///
/// # Parameters
///
/// - `residual`: `(N, s, s)` detector-grid field, e.g. `data - model`.
/// - `weight`: optional `(N, s, s)` complete per-pixel weight the solver
///   has already formed (`valid * 1/sigma^2 * robust`); `None` is unit
///   weight everywhere.
/// - `oversample`: `os`, forced odd.
/// - `stamp_size`: `s`, forced odd (the detector stamp edge).
/// - `delta`: `(N, 2)` per-stamp sub-pixel offset `(row, col)`, with the
///   same sign convention as `render` / `build_stamp`.
/// - `flux`: `(N,)` per-stamp scale `f_i`; part of the operator, so it
///   multiplies into the scatter.
///
/// # Returns
///
/// `Ok(Array2<f64>)` of shape `(os*s, os*s)`, the sum over all `N` stamps
/// of each residual scattered through the Catmull-Rom weights. An empty
/// batch yields an all-zero array of that shape.
///
/// # Errors
///
/// Returns an [`AccumulateError`] if `oversample` is not odd, if
/// `stamp_size` is not odd, if `residual` is not `(N, s, s)`, if `weight`
/// is `Some` with a shape different from `residual`, or if `delta`
/// (`(N, 2)`) and `flux` do not share `residual`'s batch length.
pub fn accumulate(
    residual: ArrayView3<f64>,
    weight: Option<ArrayView3<f64>>,
    oversample: usize,
    stamp_size: usize,
    delta: ArrayView2<f64>,
    flux: ArrayView1<f64>,
) -> Result<Array2<f64>, AccumulateError> {
    // --- Hard preconditions (returned as Err; no Ok(None) path). The
    // check order is the documented order. ---
    if oversample % 2 == 0 {
        // Also rejects oversample == 0 (0 is even, hence not odd).
        return Err(AccumulateError::OversampleNotOdd { oversample });
    }
    if stamp_size % 2 == 0 {
        // Also rejects stamp_size == 0 (degenerate empty grid).
        return Err(AccumulateError::StampSizeEven { stamp_size });
    }

    let batch_size = residual.shape()[0];
    let residual_rows = residual.shape()[1];
    let residual_cols = residual.shape()[2];
    if residual_rows != stamp_size || residual_cols != stamp_size {
        return Err(AccumulateError::ResidualShapeMismatch {
            residual: (batch_size, residual_rows, residual_cols),
            expected: (batch_size, stamp_size, stamp_size),
        });
    }
    if let Some(weight_view) = weight {
        let weight_shape = weight_view.shape();
        if weight_shape[0] != batch_size
            || weight_shape[1] != stamp_size
            || weight_shape[2] != stamp_size
        {
            return Err(AccumulateError::WeightShapeMismatch {
                weight: (weight_shape[0], weight_shape[1], weight_shape[2]),
                residual: (batch_size, stamp_size, stamp_size),
            });
        }
    }
    if delta.shape()[0] != batch_size
        || delta.shape()[1] != 2
        || flux.len() != batch_size
    {
        return Err(AccumulateError::BatchLengthMismatch {
            delta: (delta.shape()[0], delta.shape()[1]),
            residual: batch_size,
            flux: flux.len(),
        });
    }

    // After the parity checks `oversample` and `stamp_size` are both odd
    // (>= 1), so `side` is odd and these centers are exact integers in
    // f64 -- identical to `render`, which is what keeps the two operators
    // exact transposes (a delta = 0 source scatters to a single cell).
    let side = oversample * stamp_size;
    let psf_center = (side as f64 - 1.0) / 2.0;
    let detector_center = (stamp_size as f64 - 1.0) / 2.0;
    let oversample_f = oversample as f64;

    // Each stamp scatters into the shared model grid, so a naive
    // parallel-over-N write would race on overlapping 4x4 patches.
    // Fold into per-worker private accumulators, then sum them. An empty
    // batch produces no folded accumulators and `reduce` returns its
    // identity: an all-zero `(side, side)` grid.
    let epsf_accumulated = (0..batch_size)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((side, side)),
            |mut partial, stamp_index| {
                let weight_stamp = weight
                    .as_ref()
                    .map(|w| w.index_axis(Axis(0), stamp_index));
                accumulate_stamp(
                    &mut partial,
                    residual.index_axis(Axis(0), stamp_index),
                    weight_stamp,
                    delta[(stamp_index, 0)],
                    delta[(stamp_index, 1)],
                    flux[stamp_index],
                    oversample_f,
                    stamp_size,
                    psf_center,
                    detector_center,
                );
                partial
            },
        )
        .reduce(
            || Array2::<f64>::zeros((side, side)),
            |mut left, right| {
                left += &right;
                left
            },
        );

    Ok(epsf_accumulated)
}

/// Scatter one detector stamp's (weighted) residual into `epsf_acc`.
/// Factored out of the driver so the parallel and any future sequential
/// driver share the exact same per-stamp work unit -- the structural
/// mirror of `render`'s `render_stamp`.
///
/// The gather/scatter correspondence is exact: where `render_stamp`
/// computes `pred = flux * sum epsf[ru, rv] * w_u * w_v + background`,
/// this adds `flux * w_pix * residual` distributed over the *same* taps
/// with the *same* weights, so the two are transposes by construction.
#[allow(clippy::too_many_arguments)]
fn accumulate_stamp(
    epsf_acc: &mut Array2<f64>,
    residual_stamp: ArrayView2<f64>,
    weight_stamp: Option<ArrayView2<f64>>,
    delta_row: f64,
    delta_column: f64,
    flux: f64,
    oversample: f64,
    stamp_size: usize,
    psf_center: f64,
    detector_center: f64,
) {
    let side = epsf_acc.shape()[0] as i64; // square: shape()[0] == shape()[1]

    for i in 0..stamp_size {
        let k_u = psf_center + oversample * ((i as f64 - detector_center) - delta_row);
        let u_floor = k_u.floor();
        let weights_u = catmull_rom_weights(k_u - u_floor);
        let base_u = u_floor as i64 - 1;

        for j in 0..stamp_size {
            let k_v =
                psf_center + oversample * ((j as f64 - detector_center) - delta_column);
            let v_floor = k_v.floor();
            let weights_v = catmull_rom_weights(k_v - v_floor);
            let base_v = v_floor as i64 - 1;

            let pixel_weight = match weight_stamp {
                Some(w) => w[(i, j)],
                None => 1.0,
            };
            // diag(flux) and diag(weight) are part of the operator; the
            // residual carries the cotangent. Scatter the same per-tap
            // weights `render` gathered (zero-pad outside the grid).
            let coefficient = flux * pixel_weight * residual_stamp[(i, j)];

            for tap_u in 0..4 {
                let row = base_u + tap_u as i64;
                if row < 0 || row >= side {
                    continue; // zero-pad: transpose of zero padding
                }
                let weighted_u = coefficient * weights_u[tap_u];
                for tap_v in 0..4 {
                    let column = base_v + tap_v as i64;
                    if column < 0 || column >= side {
                        continue; // zero-pad: transpose of zero padding
                    }
                    epsf_acc[(row as usize, column as usize)] +=
                        weighted_u * weights_v[tap_v];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::psf::render::render;
    use ndarray::{Array1, Array2, Array3, arr2};

    /// SplitMix64: a tiny, dependency-free, fully deterministic PRNG so
    /// the "random psi, r, delta, flux" transpose tests are reproducible.
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

        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Uniform in `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn random_epsf(rng: &mut SplitMix64, side: usize) -> Array2<f64> {
        // [-1, 1): genuinely asymmetric, so any (row, col) swap or
        // off-by-one between render's gather and the scatter shows up.
        Array2::from_shape_fn((side, side), |_| rng.range(-1.0, 1.0))
    }

    fn random_residual(
        rng: &mut SplitMix64,
        batch: usize,
        stamp_size: usize,
    ) -> Array3<f64> {
        // Includes negative pixels on purpose (decision 5: negatives are
        // never dropped by sign).
        Array3::from_shape_fn((batch, stamp_size, stamp_size), |_| {
            rng.range(-1.0, 1.0)
        })
    }

    fn random_delta(rng: &mut SplitMix64, batch: usize) -> Array2<f64> {
        // Sub-pixel offsets that force genuine interpolation (non-zero
        // Catmull-Rom fractions on both axes).
        Array2::from_shape_fn((batch, 2), |_| rng.range(-0.5, 0.5))
    }

    fn random_flux(rng: &mut SplitMix64, batch: usize) -> Array1<f64> {
        Array1::from_shape_fn(batch, |_| rng.range(0.5, 3.0))
    }

    fn dot3(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn dot2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn transpose_identity_unweighted() {
        // The defining property: <render(psi; delta, f, b = 0), r> ==
        // <psi, accumulate(r; delta, f)>. The identity is algebraically
        // exact, so the bound only absorbs floating-point round-off; any
        // structural transpose bug (swapped axis, dropped diag(f),
        // mismatched tap base) is O(1) and blows it by orders.
        let oversample = 5;
        let stamp_size = 7;
        let batch = 4;
        let side = oversample * stamp_size;

        let mut rng = SplitMix64::new(0xC0FF_EE12_3456_789A);
        let epsf = random_epsf(&mut rng, side);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let delta = random_delta(&mut rng, batch);
        let flux = random_flux(&mut rng, batch);
        let background = Array1::<f64>::zeros(batch); // b not in the linear map

        let rendered = render(
            epsf.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();
        let accumulated = accumulate(
            residual.view(),
            None,
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();

        let lhs = dot3(&rendered, &residual);
        let rhs = dot2(&epsf, &accumulated);
        assert!(
            (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
            "transpose identity broken: <Ax,r> = {lhs} vs <x,A^T r> = {rhs}"
        );
    }

    #[test]
    fn transpose_identity_weighted() {
        // <render(psi; delta, f, b = 0), W . r> ==
        // <psi, accumulate(r, Some(W); delta, f)>.
        let oversample = 5;
        let stamp_size = 9;
        let batch = 3;
        let side = oversample * stamp_size;

        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        let epsf = random_epsf(&mut rng, side);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let delta = random_delta(&mut rng, batch);
        let flux = random_flux(&mut rng, batch);
        let background = Array1::<f64>::zeros(batch);
        // Weights in [0, 2): includes near-zero (effectively masked)
        // pixels.
        let weight = Array3::from_shape_fn((batch, stamp_size, stamp_size), |_| {
            rng.range(0.0, 2.0)
        });

        let rendered = render(
            epsf.view(),
            oversample,
            delta.view(),
            flux.view(),
            background.view(),
        )
        .unwrap();
        // <A x, W . r>: fold the per-pixel weight into the inner product.
        let mut lhs = 0.0;
        for ((value, w), r) in rendered.iter().zip(weight.iter()).zip(residual.iter()) {
            lhs += value * w * r;
        }

        let accumulated = accumulate(
            residual.view(),
            Some(weight.view()),
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();
        let rhs = dot2(&epsf, &accumulated);

        assert!(
            (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
            "weighted transpose identity broken: {lhs} vs {rhs}"
        );
    }

    #[test]
    fn weight_none_equals_all_ones() {
        let oversample = 3;
        let stamp_size = 5;
        let batch = 3;

        let mut rng = SplitMix64::new(0xDEAD_BEEF_F00D_BABE);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let delta = random_delta(&mut rng, batch);
        let flux = random_flux(&mut rng, batch);
        let ones = Array3::<f64>::ones((batch, stamp_size, stamp_size));

        let from_none = accumulate(
            residual.view(),
            None,
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();
        let from_ones = accumulate(
            residual.view(),
            Some(ones.view()),
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();

        for (a, b) in from_none.iter().zip(from_ones.iter()) {
            assert!((a - b).abs() < 1e-12, "None vs all-ones differ: {a} vs {b}");
        }
    }

    #[test]
    fn delta_zero_is_exact_grid_scatter() {
        // delta = 0 makes render an exact grid pick; its transpose is an
        // exact grid scatter: every residual pixel lands, undivided, on a
        // single model cell, scaled by flux (and weight).
        let oversample = 5;
        let stamp_size = 7;
        let batch = 2;
        let side = oversample * stamp_size;

        let mut rng = SplitMix64::new(0x0BAD_C0DE_1234_5678);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let flux = random_flux(&mut rng, batch);
        let delta = Array2::<f64>::zeros((batch, 2));

        let accumulated = accumulate(
            residual.view(),
            None,
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();

        let psf_center = (side - 1) / 2; // 17
        let detector_center = (stamp_size - 1) / 2; // 3
        let mut expected = Array2::<f64>::zeros((side, side));
        for n in 0..batch {
            for i in 0..stamp_size {
                for j in 0..stamp_size {
                    let p = psf_center + oversample * i - oversample * detector_center;
                    let q = psf_center + oversample * j - oversample * detector_center;
                    expected[(p, q)] += flux[n] * residual[(n, i, j)];
                }
            }
        }
        for (got, want) in accumulated.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "grid scatter mismatch: {got} != {want}"
            );
        }
    }

    #[test]
    fn source_off_grid_scatters_zero() {
        // A source pushed entirely off the model grid: every tap is
        // clipped, so the back-projection is exactly zero (transpose of
        // render's flat-background / zero-pad behavior) and never panics.
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;

        let mut rng = SplitMix64::new(0xFEED_FACE_CAFE_0001);
        let residual = random_residual(&mut rng, 1, stamp_size);
        let delta = arr2(&[[1000.0, -1000.0]]);
        let flux = Array1::from_elem(1, 2.0);

        let accumulated = accumulate(
            residual.view(),
            None,
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();
        assert_eq!(accumulated.shape(), &[side, side]);
        for value in accumulated.iter() {
            assert!(value.is_finite(), "non-finite scatter {value}");
            assert!(value.abs() < 1e-12, "expected zero, got {value}");
        }
    }

    #[test]
    fn linear_in_residual_flux_and_weight() {
        let oversample = 5;
        let stamp_size = 7;
        let batch = 3;

        let mut rng = SplitMix64::new(0xABCD_1234_5678_9F00);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let delta = random_delta(&mut rng, batch);
        let flux = random_flux(&mut rng, batch);
        let weight = Array3::from_shape_fn((batch, stamp_size, stamp_size), |_| {
            rng.range(0.1, 2.0)
        });

        let base = accumulate(
            residual.view(),
            Some(weight.view()),
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();

        // Scaling the residual by alpha scales the back-projection.
        let alpha = 2.5;
        let scaled_residual = &residual * alpha;
        let from_scaled_residual = accumulate(
            scaled_residual.view(),
            Some(weight.view()),
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();
        for (got, b) in from_scaled_residual.iter().zip(base.iter()) {
            assert!(
                (got - alpha * b).abs() < 1e-9 * (alpha * b).abs().max(1.0),
                "residual linearity: {got} != {}",
                alpha * b
            );
        }

        // flux is part of the operator: doubling it doubles the output.
        let doubled_flux = &flux * 2.0;
        let from_doubled_flux = accumulate(
            residual.view(),
            Some(weight.view()),
            oversample,
            stamp_size,
            delta.view(),
            doubled_flux.view(),
        )
        .unwrap();
        for (got, b) in from_doubled_flux.iter().zip(base.iter()) {
            assert!(
                (got - 2.0 * b).abs() < 1e-9 * (2.0 * b).abs().max(1.0),
                "flux linearity: {got} != {}",
                2.0 * b
            );
        }

        // weight is part of the operator: scaling it scales the output.
        let scaled_weight = &weight * 3.0;
        let from_scaled_weight = accumulate(
            residual.view(),
            Some(scaled_weight.view()),
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();
        for (got, b) in from_scaled_weight.iter().zip(base.iter()) {
            assert!(
                (got - 3.0 * b).abs() < 1e-9 * (3.0 * b).abs().max(1.0),
                "weight linearity: {got} != {}",
                3.0 * b
            );
        }
    }

    #[test]
    fn batch_accumulation_is_sum_of_singles() {
        // The output is the sum over N of the per-stamp back-projections.
        let oversample = 5;
        let stamp_size = 7;
        let batch = 4;
        let side = oversample * stamp_size;

        let mut rng = SplitMix64::new(0x5151_5151_AAAA_BBBB);
        let residual = random_residual(&mut rng, batch, stamp_size);
        let delta = random_delta(&mut rng, batch);
        let flux = random_flux(&mut rng, batch);

        let full = accumulate(
            residual.view(),
            None,
            oversample,
            stamp_size,
            delta.view(),
            flux.view(),
        )
        .unwrap();

        let mut summed = Array2::<f64>::zeros((side, side));
        for n in 0..batch {
            let single = accumulate(
                residual.slice(ndarray::s![n..n + 1, .., ..]),
                None,
                oversample,
                stamp_size,
                delta.slice(ndarray::s![n..n + 1, ..]),
                flux.slice(ndarray::s![n..n + 1]),
            )
            .unwrap();
            summed += &single;
        }

        for (got, want) in full.iter().zip(summed.iter()) {
            assert!(
                (got - want).abs() < 1e-9 * want.abs().max(1.0),
                "batch sum mismatch: {got} != {want}"
            );
        }
    }

    #[test]
    fn empty_batch_yields_zero_grid() {
        let oversample = 5;
        let stamp_size = 7;
        let side = oversample * stamp_size;

        let accumulated = accumulate(
            Array3::<f64>::zeros((0, stamp_size, stamp_size)).view(),
            None,
            oversample,
            stamp_size,
            Array2::<f64>::zeros((0, 2)).view(),
            Array1::<f64>::zeros(0).view(),
        )
        .unwrap();
        assert_eq!(accumulated.shape(), &[side, side]);
        assert!(accumulated.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn error_oversample_not_odd() {
        let residual = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        let flux = Array1::from_elem(1, 1.0);
        for bad in [4usize, 0usize] {
            let err = accumulate(
                residual.view(),
                None,
                bad,
                7,
                delta.view(),
                flux.view(),
            )
            .unwrap_err();
            assert_eq!(err, AccumulateError::OversampleNotOdd { oversample: bad });
        }
    }

    #[test]
    fn error_stamp_size_even() {
        let residual = Array3::<f64>::zeros((1, 7, 7));
        let delta = arr2(&[[0.0, 0.0]]);
        let flux = Array1::from_elem(1, 1.0);
        for bad in [4usize, 0usize] {
            let err = accumulate(
                residual.view(),
                None,
                5,
                bad,
                delta.view(),
                flux.view(),
            )
            .unwrap_err();
            assert_eq!(err, AccumulateError::StampSizeEven { stamp_size: bad });
        }
    }

    #[test]
    fn error_residual_shape_mismatch() {
        // stamp_size = 7 but residual rows = 8.
        let residual = Array3::<f64>::zeros((2, 8, 7));
        let delta = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        let flux = Array1::from_elem(2, 1.0);
        let err = accumulate(
            residual.view(),
            None,
            5,
            7,
            delta.view(),
            flux.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            AccumulateError::ResidualShapeMismatch {
                residual: (2, 8, 7),
                expected: (2, 7, 7),
            }
        );
    }

    #[test]
    fn error_weight_shape_mismatch() {
        let residual = Array3::<f64>::zeros((2, 7, 7));
        let weight = Array3::<f64>::zeros((2, 7, 5));
        let delta = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        let flux = Array1::from_elem(2, 1.0);
        let err = accumulate(
            residual.view(),
            Some(weight.view()),
            5,
            7,
            delta.view(),
            flux.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            AccumulateError::WeightShapeMismatch {
                weight: (2, 7, 5),
                residual: (2, 7, 7),
            }
        );
    }

    #[test]
    fn error_batch_length_mismatch_delta_not_two_columns() {
        let residual = Array3::<f64>::zeros((2, 7, 7));
        let delta = arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]); // (2, 3)
        let flux = Array1::from_elem(2, 1.0);
        let err = accumulate(
            residual.view(),
            None,
            5,
            7,
            delta.view(),
            flux.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            AccumulateError::BatchLengthMismatch {
                delta: (2, 3),
                residual: 2,
                flux: 2,
            }
        );
    }

    #[test]
    fn error_batch_length_mismatch_flux_len() {
        let residual = Array3::<f64>::zeros((3, 7, 7));
        let delta = arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]); // (3, 2)
        let flux = Array1::from_elem(2, 1.0); // len 2 != N = 3
        let err = accumulate(
            residual.view(),
            None,
            5,
            7,
            delta.view(),
            flux.view(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            AccumulateError::BatchLengthMismatch {
                delta: (3, 2),
                residual: 3,
                flux: 2,
            }
        );
    }
}

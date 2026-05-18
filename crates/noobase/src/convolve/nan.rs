//! NaN-as-missing renormalized convolution (image side only).
//!
//! These wrappers turn the NaN-naive bare kernels into the
//! *normalized convolution* `out = N / D` (ROADMAP D5), where
//!
//! ```text
//! N = correlate(filled, kernel),  filled = if finite(v) { v } else { 0 }
//! D = correlate(valid,  kernel),  valid  = if finite(v) { 1 } else { 0 }
//! ```
//!
//! `finite(v) := v.is_finite()` is a *single* predicate: NaN and ±inf are
//! all treated as invalid (missing). This is exactly
//! `image::reproject_exact`'s `Σ(A·in) / Σ(A_valid)` and is the same
//! NaN-as-missing convention used across the image subsystem (ROADMAP
//! D9). The decomposition's two correlations have NaN-free inputs, so the
//! semantics are backend-uniform: a future FFT backend computes the same
//! `N` and `D` and divides identically.
//!
//! These take **no `Boundary`**. Out-of-bounds taps are always missing
//! (they contribute 0 to *both* `N` and `D` — zero-extension), which is
//! the renorm path's own, mutually exclusive boundary model (ROADMAP
//! D13): Reflect/Nearest would keep `D` saturated at the edge and defeat
//! the missing-data correction. Equivalently, OOB is resolved exactly as
//! the bare kernels resolve [`Boundary::Zero`] — the shared
//! [`resolve_index`] enforces this in one place.
//!
//! The implementation is the **single-pass direct** v1 backend: it fuses
//! the two correlations into one tap sweep, accumulating `N` and `D`
//! together. It is written to be *bitwise-identical* to running the two
//! bare correlations separately and dividing (pinned by tests), so the
//! "optimized" single pass never diverges from the semantic definition.
//!
//! LSF does not go through this path: an LSF template is noise-free by
//! construction (ROADMAP D8), so it has no NaN/error/mask to renormalize.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use rayon::prelude::*;

use crate::convolve::{Boundary, resolve_index};
use crate::float::Float;

/// Dense 2-D **renormalized correlation** (NaN-as-missing). Returns
/// `(value, weight)`, both the same shape as `image`:
///
/// - `value[i,j] = N / D` where `N`/`D` are the kernel-weighted sums of
///   the finite / validity-masked taps (see the module docs). Where no
///   valid tap contributed (`D` not strictly positive) the value is
///   `NaN`.
/// - `weight[i,j] = D` — the valid weight that fed the average. For a
///   `Sum`-normalized non-negative kernel this is the fraction of the
///   window that was valid and in-bounds (`<= 1`).
///
/// The correlation is **not** flipped (ROADMAP D3); `image::convolve_psf`
/// flips the PSF before calling this. Parallelized across output rows
/// with rayon; rows are independent and per-pixel taps accumulate in a
/// fixed row-major order, so the result is bitwise-identical to a
/// sequential run (cf. [`crate::image::reproject_exact`]).
///
/// `D` is gated with a strict `> 0` test (the `eps -> 0+` choice),
/// matching `reproject_exact`'s `denominator_valid > 0` convention. The
/// renorm assumes a non-negative kernel: a negative-lobe kernel can
/// drive `D` toward zero or change its sign and blow the division up —
/// out of scope here, and matched-filter callers must therefore not
/// renormalize (ROADMAP D5 caveat).
///
/// Panics if `kernel` has a zero dimension.
pub fn conv2d_renorm<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
) -> (Array2<T>, Array2<T>) {
    assert!(
        kernel.nrows() > 0 && kernel.ncols() > 0,
        "conv2d_renorm kernel must be non-empty"
    );
    let mut value = Array2::<T>::zeros(image.raw_dim());
    let mut weight = Array2::<T>::zeros(image.raw_dim());
    value
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(weight.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(row_index, (value_row, weight_row))| {
            renorm_row(row_index, value_row, weight_row, image, kernel);
        });
    (value, weight)
}

/// Sequential driver, retained for the parallel-vs-sequential
/// equivalence test (reuses the identical per-row work unit).
#[cfg(test)]
fn conv2d_renorm_sequential<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
) -> (Array2<T>, Array2<T>) {
    assert!(
        kernel.nrows() > 0 && kernel.ncols() > 0,
        "conv2d_renorm kernel must be non-empty"
    );
    let mut value = Array2::<T>::zeros(image.raw_dim());
    let mut weight = Array2::<T>::zeros(image.raw_dim());
    for (row_index, (value_row, weight_row)) in value
        .axis_iter_mut(Axis(0))
        .zip(weight.axis_iter_mut(Axis(0)))
        .enumerate()
    {
        renorm_row(row_index, value_row, weight_row, image, kernel);
    }
    (value, weight)
}

/// One output row of the 2-D renorm, in place. Shared by both drivers.
fn renorm_row<T: Float>(
    row_index: usize,
    mut value_row: ArrayViewMut1<T>,
    mut weight_row: ArrayViewMut1<T>,
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
) {
    let height = image.nrows();
    let width = image.ncols();
    let kernel_rows = kernel.nrows();
    let kernel_cols = kernel.ncols();
    let center_row = kernel_rows / 2;
    let center_col = kernel_cols / 2;
    for column_index in 0..width {
        let mut numerator = T::zero();
        let mut denominator = T::zero();
        for kernel_row in 0..kernel_rows {
            let source_row = row_index as isize + kernel_row as isize - center_row as isize;
            // OOB = missing (zero-extension); resolved exactly as the
            // bare kernels resolve Boundary::Zero so the two-pass
            // equivalence is bit-exact.
            let Some(image_row) = resolve_index(source_row, height, Boundary::Zero) else {
                continue;
            };
            for kernel_col in 0..kernel_cols {
                let source_col = column_index as isize + kernel_col as isize - center_col as isize;
                let Some(image_col) = resolve_index(source_col, width, Boundary::Zero) else {
                    continue;
                };
                let weight = kernel[(kernel_row, kernel_col)];
                let sample = image[(image_row, image_col)];
                let (filled, valid) = split_finite(sample);
                // Add unconditionally (even the 0/0 of an invalid tap):
                // this is correlate(filled)/correlate(valid) fused into
                // one sweep, op-for-op and in the same order.
                numerator = numerator + weight * filled;
                denominator = denominator + weight * valid;
            }
        }
        let (value, output_weight) = finalize(numerator, denominator);
        value_row[column_index] = value;
        weight_row[column_index] = output_weight;
    }
}

/// 1-D **renormalized correlation** applied independently along `axis`
/// of a 2-D image (NaN-as-missing). Returns `(value, weight)` shaped
/// like `image`, with the same `N / D` semantics, OOB-as-missing
/// boundary model, and `D > 0` gate as [`conv2d_renorm`].
///
/// This is the renorm companion of [`crate::convolve::conv_axis`] (req3
/// grism uses the non-renorm path; this exists for callers that opt into
/// missing-data correction along one axis). Lanes are independent and
/// evaluated sequentially, consistent with `conv_axis`.
///
/// Panics if `kernel` is empty.
pub fn conv_axis_renorm<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView1<T>,
    axis: Axis,
) -> (Array2<T>, Array2<T>) {
    assert!(
        !kernel.is_empty(),
        "conv_axis_renorm kernel must be non-empty"
    );
    let mut value = Array2::<T>::zeros(image.raw_dim());
    let mut weight = Array2::<T>::zeros(image.raw_dim());
    for ((input_lane, mut value_lane), mut weight_lane) in image
        .lanes(axis)
        .into_iter()
        .zip(value.lanes_mut(axis))
        .zip(weight.lanes_mut(axis))
    {
        renorm_lane(
            input_lane,
            kernel,
            value_lane.view_mut(),
            weight_lane.view_mut(),
        );
    }
    (value, weight)
}

/// One 1-D lane of the renorm, in place.
fn renorm_lane<T: Float>(
    signal: ArrayView1<T>,
    kernel: ArrayView1<T>,
    mut value_lane: ArrayViewMut1<T>,
    mut weight_lane: ArrayViewMut1<T>,
) {
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let center = kernel_len / 2;
    for output_index in 0..signal_len {
        let mut numerator = T::zero();
        let mut denominator = T::zero();
        for kernel_index in 0..kernel_len {
            let source = output_index as isize + kernel_index as isize - center as isize;
            let Some(sample_index) = resolve_index(source, signal_len, Boundary::Zero) else {
                continue;
            };
            let weight = kernel[kernel_index];
            let (filled, valid) = split_finite(signal[sample_index]);
            numerator = numerator + weight * filled;
            denominator = denominator + weight * valid;
        }
        let (value, output_weight) = finalize(numerator, denominator);
        value_lane[output_index] = value;
        weight_lane[output_index] = output_weight;
    }
}

/// Split a sample into its `(filled, valid)` contributions under the
/// single `is_finite` predicate: a finite sample contributes `(v, 1)`,
/// any non-finite sample (NaN or ±inf) contributes `(0, 0)`.
#[inline]
fn split_finite<T: Float>(sample: T) -> (T, T) {
    if sample.is_finite() {
        (sample, T::one())
    } else {
        (T::zero(), T::zero())
    }
}

/// `out = N / D`, with `D` strictly-positive gating: where no valid tap
/// contributed the value is `NaN`. Returns `(value, weight = D)`.
#[inline]
fn finalize<T: Float>(numerator: T, denominator: T) -> (T, T) {
    if denominator > T::zero() {
        (numerator / denominator, denominator)
    } else {
        (T::nan(), denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve::{conv_axis, conv2d};
    use ndarray::{Array1, Array2, array};

    const TOL_F64: f64 = 1e-12;

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL_F64 * a.abs().max(b.abs()).max(1.0)
    }

    /// Independent two-pass reference: the *semantic definition* of the
    /// renorm built from the bare kernels. The single-pass production
    /// path must match this bit for bit.
    fn two_pass_2d_reference(
        image: ArrayView2<f64>,
        kernel: ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let filled = image.mapv(|v| if v.is_finite() { v } else { 0.0 });
        let valid = image.mapv(|v| if v.is_finite() { 1.0 } else { 0.0 });
        let numerator = conv2d(filled.view(), kernel, Boundary::Zero);
        let denominator = conv2d(valid.view(), kernel, Boundary::Zero);
        let mut value = Array2::<f64>::zeros(image.raw_dim());
        for ((i, j), out) in value.indexed_iter_mut() {
            let d = denominator[(i, j)];
            *out = if d > 0.0 {
                numerator[(i, j)] / d
            } else {
                f64::NAN
            };
        }
        (value, denominator)
    }

    fn bits_eq(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            true
        } else {
            a.to_bits() == b.to_bits()
        }
    }

    #[test]
    fn nan_free_constant_image_recovers_constant_and_counts_weight() {
        // Constant image, unnormalized 3x3 ones kernel, no NaN.
        // value = N/D = c*D/D = c everywhere (even edges). weight = D =
        // the number of in-bounds taps (no Boundary extension).
        let image = Array2::<f64>::from_elem((4, 5), 2.5);
        let kernel = Array2::<f64>::from_elem((3, 3), 1.0);
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        for v in value.iter() {
            assert!(approx_eq_f64(*v, 2.5));
        }
        // Corner sees a 2x2 window, edge 2x3, interior 3x3.
        assert!(approx_eq_f64(weight[(0, 0)], 4.0));
        assert!(approx_eq_f64(weight[(0, 1)], 6.0));
        assert!(approx_eq_f64(weight[(1, 1)], 9.0));
    }

    #[test]
    fn nan_free_sum_kernel_interior_equals_bare_conv() {
        // ROADMAP §7: NaN-free, Sum-normalized kernel -> D = 1 at the
        // interior, so renorm value == bare conv there.
        let mut image = Array2::<f64>::zeros((6, 6));
        for ((i, j), v) in image.indexed_iter_mut() {
            *v = (i * 7 + j * 3) as f64 * 0.1 - 1.0;
        }
        let ninth = 1.0 / 9.0;
        let kernel = Array2::<f64>::from_elem((3, 3), ninth);
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        let bare = conv2d(image.view(), kernel.view(), Boundary::Zero);
        for i in 1..5 {
            for j in 1..5 {
                assert!(approx_eq_f64(weight[(i, j)], 1.0));
                assert!(approx_eq_f64(value[(i, j)], bare[(i, j)]));
            }
        }
    }

    #[test]
    fn nan_excluded_from_numerator_and_weight_handcalc() {
        // 3x3 image, center NaN, 3x3 ones kernel. Output at (1,1) sees
        // all 9 taps; the NaN one is dropped -> N = sum of the other 8,
        // D = 8, value = N/8.
        let image = array![[1.0_f64, 2.0, 3.0], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];
        let kernel = Array2::<f64>::from_elem((3, 3), 1.0);
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        let finite_sum = 1.0 + 2.0 + 3.0 + 4.0 + 6.0 + 7.0 + 8.0 + 9.0;
        assert!(approx_eq_f64(weight[(1, 1)], 8.0));
        assert!(approx_eq_f64(value[(1, 1)], finite_sum / 8.0));
        // The NaN's own pixel is reconstructed from its valid neighbors,
        // not propagated.
        assert!(value[(1, 1)].is_finite());
    }

    #[test]
    fn all_invalid_window_yields_nan_zero_weight() {
        let image = Array2::<f64>::from_elem((3, 3), f64::NAN);
        let kernel = Array2::<f64>::from_elem((3, 3), 1.0);
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        for v in value.iter() {
            assert!(v.is_nan());
        }
        for w in weight.iter() {
            assert!(approx_eq_f64(*w, 0.0));
        }
    }

    #[test]
    fn plus_minus_inf_treated_same_as_nan() {
        // Single is_finite predicate: +inf / -inf are invalid exactly
        // like NaN. Same image with NaN vs +inf vs -inf at the center
        // must give bitwise-identical (value, weight).
        let kernel = Array2::<f64>::from_elem((3, 3), 1.0);
        let make = |bad: f64| array![[1.0_f64, 2.0, 3.0], [4.0, bad, 6.0], [7.0, 8.0, 9.0]];
        let (v_nan, w_nan) = conv2d_renorm(make(f64::NAN).view(), kernel.view());
        let (v_pinf, w_pinf) = conv2d_renorm(make(f64::INFINITY).view(), kernel.view());
        let (v_ninf, w_ninf) = conv2d_renorm(make(f64::NEG_INFINITY).view(), kernel.view());
        for idx in 0..9 {
            let (i, j) = (idx / 3, idx % 3);
            assert!(bits_eq(v_nan[(i, j)], v_pinf[(i, j)]));
            assert!(bits_eq(v_nan[(i, j)], v_ninf[(i, j)]));
            assert!(bits_eq(w_nan[(i, j)], w_pinf[(i, j)]));
            assert!(bits_eq(w_nan[(i, j)], w_ninf[(i, j)]));
        }
    }

    #[test]
    fn single_pass_matches_two_pass_reference_bitwise() {
        // The central pin (ROADMAP D5 / §7): the fused single pass is
        // bit-exact with two bare correlations then divide, including
        // NaN, +inf, -inf and image-edge (OOB = missing) behavior.
        let mut image = Array2::<f64>::zeros((9, 11));
        for ((i, j), v) in image.indexed_iter_mut() {
            *v = ((i * 5 + j * 3) % 17) as f64 * 0.37 - 2.0 + (j as f64).cos();
        }
        image[(0, 0)] = f64::NAN; // corner: also exercises OOB
        image[(4, 5)] = f64::INFINITY;
        image[(7, 2)] = f64::NEG_INFINITY;
        image[(8, 10)] = f64::NAN; // opposite corner
        let kernel = array![[0.2_f64, 0.5, 0.1], [0.3, 1.0, 0.25], [0.05, 0.4, 0.15]];
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        let (ref_value, ref_weight) = two_pass_2d_reference(image.view(), kernel.view());
        for i in 0..image.nrows() {
            for j in 0..image.ncols() {
                assert!(
                    bits_eq(value[(i, j)], ref_value[(i, j)]),
                    "value mismatch at ({i},{j})"
                );
                assert!(
                    bits_eq(weight[(i, j)], ref_weight[(i, j)]),
                    "weight mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn parallel_matches_sequential_bitwise() {
        let mut image = Array2::<f64>::zeros((40, 47));
        for ((i, j), v) in image.indexed_iter_mut() {
            *v = ((i * 11 + j * 13) % 23) as f64 / 7.0 + (i as f64).sin();
        }
        image[(10, 10)] = f64::NAN;
        image[(25, 30)] = f64::INFINITY;
        image[(0, 46)] = f64::NEG_INFINITY;
        let kernel = array![
            [0.1_f64, 0.2, 0.3, 0.05, 0.0],
            [0.2, 0.5, -0.1, 0.07, 0.15],
            [0.05, 0.25, 1.0, 0.3, 0.02],
            [0.0, 0.12, 0.08, 0.33, 0.09],
            [0.04, 0.06, 0.11, 0.01, 0.13]
        ];
        let (p_value, p_weight) = conv2d_renorm(image.view(), kernel.view());
        let (s_value, s_weight) = conv2d_renorm_sequential(image.view(), kernel.view());
        for i in 0..image.nrows() {
            for j in 0..image.ncols() {
                assert!(bits_eq(p_value[(i, j)], s_value[(i, j)]));
                assert!(bits_eq(p_weight[(i, j)], s_weight[(i, j)]));
            }
        }
    }

    #[test]
    fn conv_axis_renorm_matches_two_pass_reference_bitwise() {
        let mut image = Array2::<f64>::zeros((7, 8));
        for ((i, j), v) in image.indexed_iter_mut() {
            *v = ((i * 3 + j * 5) % 13) as f64 * 0.5 - 1.5;
        }
        image[(0, 2)] = f64::NAN;
        image[(6, 5)] = f64::INFINITY;
        image[(3, 0)] = f64::NEG_INFINITY;
        let kernel = array![0.25_f64, 0.5, 1.0, 0.5, 0.25];
        for axis in [Axis(0), Axis(1)] {
            let (value, weight) = conv_axis_renorm(image.view(), kernel.view(), axis);
            // Two-pass reference using the bare conv_axis.
            let filled = image.mapv(|v| if v.is_finite() { v } else { 0.0 });
            let valid = image.mapv(|v| if v.is_finite() { 1.0 } else { 0.0 });
            let numerator = conv_axis(filled.view(), kernel.view(), axis, Boundary::Zero);
            let denominator = conv_axis(valid.view(), kernel.view(), axis, Boundary::Zero);
            for i in 0..image.nrows() {
                for j in 0..image.ncols() {
                    let d = denominator[(i, j)];
                    let expected_value = if d > 0.0 {
                        numerator[(i, j)] / d
                    } else {
                        f64::NAN
                    };
                    assert!(bits_eq(value[(i, j)], expected_value));
                    assert!(bits_eq(weight[(i, j)], d));
                }
            }
        }
    }

    #[test]
    fn conv_axis_renorm_constant_recovers_constant_f64() {
        let image = Array2::<f64>::from_elem((5, 6), -3.0);
        let kernel = array![1.0_f64, 1.0, 1.0];
        let (value, weight) = conv_axis_renorm(image.view(), kernel.view(), Axis(0));
        for v in value.iter() {
            assert!(approx_eq_f64(*v, -3.0));
        }
        // Along Axis(0) (columns of length 5): interior rows see 3 taps,
        // the two end rows see 2 (OOB = missing).
        assert!(approx_eq_f64(weight[(0, 0)], 2.0));
        assert!(approx_eq_f64(weight[(2, 0)], 3.0));
        assert!(approx_eq_f64(weight[(4, 0)], 2.0));
    }

    #[test]
    #[should_panic(expected = "kernel must be non-empty")]
    fn conv2d_renorm_empty_kernel_panics() {
        let image = Array2::<f64>::zeros((3, 3));
        let kernel = Array2::<f64>::zeros((3, 0));
        let _ = conv2d_renorm(image.view(), kernel.view());
    }

    #[test]
    fn works_with_f32() {
        let image = Array2::<f32>::from_elem((4, 4), 2.0);
        let kernel = Array2::<f32>::from_elem((3, 3), 1.0);
        let (value, weight) = conv2d_renorm(image.view(), kernel.view());
        for v in value.iter() {
            assert!((*v - 2.0).abs() <= 1e-5);
        }
        assert!((weight[(1, 1)] - 9.0).abs() <= 1e-5);

        let mut with_nan = Array1::<f32>::from_elem(5, 1.0);
        with_nan[2] = f32::NAN;
        let signal = with_nan.insert_axis(Axis(0));
        let kernel_1d = array![1.0_f32, 1.0, 1.0];
        let (v_axis, w_axis) = conv_axis_renorm(signal.view(), kernel_1d.view(), Axis(1));
        // Column index 2 is NaN; output at 2 averages neighbors 1 and 3
        // (both 1.0) -> 1.0, weight 2.
        assert!((v_axis[(0, 2)] - 1.0).abs() <= 1e-5);
        assert!((w_axis[(0, 2)] - 2.0).abs() <= 1e-5);
    }
}

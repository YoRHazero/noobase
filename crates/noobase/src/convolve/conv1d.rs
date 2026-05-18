use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::convolve::Boundary;
use crate::float::Float;

/// 1-D **correlation** of `signal` with `kernel` (the kernel is *not*
/// flipped). With `center = kernel.len() / 2`,
/// ```text
/// out[i] = Σ_k kernel[k] * sample(signal, i + k - center)
/// ```
/// for every `i in 0..signal.len()`, so the output is the same length as
/// the signal ('same'-size). `sample` resolves out-of-bounds indices via
/// `boundary`.
///
/// This is a pure computational primitive: it performs only `Σ w·v` and
/// is **NaN-naive** — a NaN (or ±inf) anywhere inside a tap window
/// propagates straight into that output element. NaN-as-missing
/// renormalization lives in a separate wrapper layer, not here. True
/// convolution flips the kernel before calling this; for a symmetric
/// kernel the two are identical.
///
/// The kernel length is conventionally odd (so `center` is the exact
/// middle), but any non-empty kernel is accepted; `center` is always
/// `kernel.len() / 2`. Panics if `kernel` is empty.
pub fn conv1d<T: Float>(
    signal: ArrayView1<T>,
    kernel: ArrayView1<T>,
    boundary: Boundary,
) -> Array1<T> {
    assert!(!kernel.is_empty(), "conv1d kernel must be non-empty");
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let center = kernel_len / 2;
    let mut output = Array1::<T>::zeros(signal_len);
    if signal_len == 0 {
        return output;
    }
    for output_index in 0..signal_len {
        let mut accumulator = T::zero();
        // Ascending tap order is fixed so the floating-point
        // accumulation is deterministic; the 2-D drivers reuse this
        // per-lane order to stay bitwise-stable under parallelism.
        for kernel_index in 0..kernel_len {
            let offset = kernel_index as isize - center as isize;
            let source = output_index as isize + offset;
            if let Some(sample_index) = resolve_index(source, signal_len, boundary) {
                accumulator = accumulator + kernel[kernel_index] * signal[sample_index];
            }
        }
        output[output_index] = accumulator;
    }
    output
}

/// Apply [`conv1d`] independently along `axis` of a 2-D image: every
/// lane parallel to `axis` is correlated with the 1-D `kernel`.
/// `Axis(0)` correlates down each column, `Axis(1)` along each row. The
/// output has the same shape as `image` and inherits the NaN-naive
/// correlation semantics of [`conv1d`].
///
/// Panics if `kernel` is empty.
pub fn conv_axis<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView1<T>,
    axis: Axis,
    boundary: Boundary,
) -> Array2<T> {
    assert!(!kernel.is_empty(), "conv_axis kernel must be non-empty");
    let mut output = Array2::<T>::zeros(image.raw_dim());
    // `lanes(axis)` iterates the 1-D slices that run along `axis`, in the
    // same logical order for `image` and the freshly allocated `output`,
    // so the zip pairs corresponding lanes regardless of memory layout.
    for (input_lane, mut output_lane) in image.lanes(axis).into_iter().zip(output.lanes_mut(axis)) {
        let convolved = conv1d(input_lane, kernel, boundary);
        output_lane.assign(&convolved);
    }
    output
}

/// Resolve a (possibly out-of-range) source index against a signal of
/// length `length` under `boundary`. Returns `None` only for
/// [`Boundary::Zero`] out-of-range indices (the tap contributes nothing).
/// `length` is assumed `>= 1` (callers early-return on empty signals).
#[inline]
fn resolve_index(source: isize, length: usize, boundary: Boundary) -> Option<usize> {
    let n = length as isize;
    if source >= 0 && source < n {
        return Some(source as usize);
    }
    match boundary {
        Boundary::Zero => None,
        Boundary::Nearest => Some(source.clamp(0, n - 1) as usize),
        Boundary::Reflect => {
            if n == 1 {
                return Some(0);
            }
            // Half-sample symmetric folding with period 2n: indices in
            // [n, 2n) mirror back onto [0, n). Works for arbitrarily
            // out-of-range indices, not just one kernel half-width.
            let period = 2 * n;
            let mut folded = source % period;
            if folded < 0 {
                folded += period;
            }
            if folded >= n {
                folded = period - 1 - folded;
            }
            Some(folded as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    const TOL_F64: f64 = 1e-12;
    const TOL_F32: f32 = 1e-5;

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL_F64 * a.abs().max(b.abs()).max(1.0)
    }

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL_F32 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn single_tap_kernel_is_identity_f64() {
        let signal = array![1.0_f64, -2.0, 3.5, 4.0, 5.0];
        let kernel = array![1.0_f64];
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let output = conv1d(signal.view(), kernel.view(), boundary);
            for i in 0..signal.len() {
                assert!(approx_eq_f64(output[i], signal[i]));
            }
        }
    }

    #[test]
    fn delta_kernel_is_identity_including_edges_f64() {
        // Centered delta [0, 1, 0]: every output element picks exactly
        // the center sample, so identity holds even at the boundaries
        // regardless of the boundary mode.
        let signal = array![7.0_f64, 1.0, -3.0, 9.0];
        let kernel = array![0.0_f64, 1.0, 0.0];
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let output = conv1d(signal.view(), kernel.view(), boundary);
            for i in 0..signal.len() {
                assert!(approx_eq_f64(output[i], signal[i]));
            }
        }
    }

    #[test]
    fn correlation_does_not_flip_kernel_f64() {
        // Asymmetric kernel pins ROADMAP D3: this is correlation, not
        // convolution. Convolution would flip [1,2,3] -> [3,2,1] and
        // give a different interior (out[1] would be 10, not 14).
        let signal = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let kernel = array![1.0_f64, 2.0, 3.0]; // center = 1
        let output = conv1d(signal.view(), kernel.view(), Boundary::Zero);
        // out[i] = 1*s[i-1] + 2*s[i] + 3*s[i+1], zero outside.
        let expected = array![8.0_f64, 14.0, 20.0, 26.0, 14.0];
        for i in 0..signal.len() {
            assert!(
                approx_eq_f64(output[i], expected[i]),
                "i={i} got {} expected {}",
                output[i],
                expected[i]
            );
        }
    }

    #[test]
    fn symmetric_kernel_correlation_equals_convolution_f64() {
        // For a symmetric kernel, correlating with it and with its
        // reverse must give identical results.
        let signal = array![2.0_f64, -1.0, 4.0, 0.5, 3.0, -2.0];
        let kernel = array![1.0_f64, 2.0, 1.0];
        let reversed = array![1.0_f64, 2.0, 1.0]; // already symmetric
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let a = conv1d(signal.view(), kernel.view(), boundary);
            let b = conv1d(signal.view(), reversed.view(), boundary);
            for i in 0..signal.len() {
                assert!(approx_eq_f64(a[i], b[i]));
            }
        }
    }

    #[test]
    fn zero_boundary_drops_edges_f64() {
        let signal = array![2.0_f64, 2.0, 2.0];
        let kernel = array![1.0_f64, 1.0, 1.0];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Zero);
        // edges miss one tap each: [2+2, 2+2+2, 2+2] = [4, 6, 4]
        assert!(approx_eq_f64(output[0], 4.0));
        assert!(approx_eq_f64(output[1], 6.0));
        assert!(approx_eq_f64(output[2], 4.0));
    }

    #[test]
    fn nearest_boundary_preserves_constant_f64() {
        let signal = array![2.0_f64, 2.0, 2.0];
        let kernel = array![1.0_f64, 1.0, 1.0];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Nearest);
        for value in output.iter() {
            assert!(approx_eq_f64(*value, 6.0));
        }
    }

    #[test]
    fn sum_kernel_with_reflect_preserves_constant_f64() {
        // Σ=1 box kernel under Reflect leaves a constant signal
        // unchanged everywhere (integral-preserving at the edges too).
        let signal = Array1::<f64>::from_elem(7, 4.25);
        let third = 1.0 / 3.0;
        let kernel = array![third, third, third];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Reflect);
        for value in output.iter() {
            assert!(approx_eq_f64(*value, 4.25));
        }
    }

    #[test]
    fn reflect_boundary_known_values_f64() {
        let signal = array![1.0_f64, 2.0, 3.0];
        let kernel = array![1.0_f64, 1.0, 1.0];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Reflect);
        // out[0] = s(reflect -1)+s0+s1 = 1+1+2 = 4
        // out[1] = s0+s1+s2 = 6
        // out[2] = s1+s2+s(reflect 3) = 2+3+3 = 8
        assert!(approx_eq_f64(output[0], 4.0));
        assert!(approx_eq_f64(output[1], 6.0));
        assert!(approx_eq_f64(output[2], 8.0));
    }

    #[test]
    fn nan_is_naive_propagates_f64() {
        // Pure Σ w·v: a NaN in a tap window poisons that output element
        // and only that window's reach (no renormalization here).
        let signal = array![1.0_f64, f64::NAN, 3.0, 4.0, 5.0];
        let kernel = array![1.0_f64, 1.0, 1.0];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Zero);
        assert!(output[0].is_nan());
        assert!(output[1].is_nan());
        assert!(output[2].is_nan());
        // Index 3's window is [2,3,4] — clear of the NaN.
        assert!(approx_eq_f64(output[3], 12.0));
        assert!(approx_eq_f64(output[4], 9.0));
    }

    #[test]
    #[should_panic(expected = "kernel must be non-empty")]
    fn empty_kernel_panics() {
        let signal = array![1.0_f64, 2.0];
        let kernel = Array1::<f64>::zeros(0);
        let _ = conv1d(signal.view(), kernel.view(), Boundary::Zero);
    }

    #[test]
    fn single_tap_kernel_is_identity_f32() {
        let signal = array![1.0_f32, -2.0, 3.5, 4.0];
        let kernel = array![1.0_f32];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Nearest);
        for i in 0..signal.len() {
            assert!(approx_eq_f32(output[i], signal[i]));
        }
    }

    #[test]
    fn correlation_does_not_flip_kernel_f32() {
        let signal = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = array![1.0_f32, 2.0, 3.0];
        let output = conv1d(signal.view(), kernel.view(), Boundary::Zero);
        let expected = array![8.0_f32, 14.0, 20.0, 26.0, 14.0];
        for i in 0..signal.len() {
            assert!(approx_eq_f32(output[i], expected[i]));
        }
    }

    #[test]
    fn conv_axis_delta_is_identity_both_axes_f64() {
        let image = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let kernel = array![0.0_f64, 1.0, 0.0];
        for axis in [Axis(0), Axis(1)] {
            let output = conv_axis(image.view(), kernel.view(), axis, Boundary::Zero);
            for i in 0..image.shape()[0] {
                for j in 0..image.shape()[1] {
                    assert!(approx_eq_f64(output[(i, j)], image[(i, j)]));
                }
            }
        }
    }

    #[test]
    fn conv_axis_matches_per_lane_conv1d_f64() {
        let image = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let kernel = array![1.0_f64, 2.0, 3.0];
        let boundary = Boundary::Reflect;

        // Axis(1): each output row equals conv1d of the input row.
        let along_rows = conv_axis(image.view(), kernel.view(), Axis(1), boundary);
        for i in 0..image.shape()[0] {
            let row = image.row(i);
            let expected = conv1d(row, kernel.view(), boundary);
            for j in 0..image.shape()[1] {
                assert!(approx_eq_f64(along_rows[(i, j)], expected[j]));
            }
        }

        // Axis(0): each output column equals conv1d of the input column.
        let along_cols = conv_axis(image.view(), kernel.view(), Axis(0), boundary);
        for j in 0..image.shape()[1] {
            let column = image.column(j);
            let expected = conv1d(column, kernel.view(), boundary);
            for i in 0..image.shape()[0] {
                assert!(approx_eq_f64(along_cols[(i, j)], expected[i]));
            }
        }
    }

    #[test]
    fn conv_axis_orientation_is_explicit_f64() {
        // Hand-computed: Axis(1) correlates along each row. Row 0 of
        // [[1,2,3],[4,5,6]] with kernel [1,2,3] (center 1), Zero:
        //   out = [2*1+3*2, 1*1+2*2+3*3, 1*2+2*3] = [8, 14, 8]
        let image = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let kernel = array![1.0_f64, 2.0, 3.0];
        let output = conv_axis(image.view(), kernel.view(), Axis(1), Boundary::Zero);
        assert!(approx_eq_f64(output[(0, 0)], 8.0));
        assert!(approx_eq_f64(output[(0, 1)], 14.0));
        assert!(approx_eq_f64(output[(0, 2)], 8.0));
    }

    #[test]
    fn conv_axis_delta_is_identity_f32() {
        let image = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let kernel = array![0.0_f32, 1.0, 0.0];
        let output = conv_axis(image.view(), kernel.view(), Axis(0), Boundary::Reflect);
        for i in 0..image.shape()[0] {
            for j in 0..image.shape()[1] {
                assert!(approx_eq_f32(output[(i, j)], image[(i, j)]));
            }
        }
    }
}

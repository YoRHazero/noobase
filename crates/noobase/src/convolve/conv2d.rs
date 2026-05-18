use ndarray::{Array2, ArrayView2, ArrayViewMut1, Axis};
use rayon::prelude::*;

use crate::convolve::{Boundary, resolve_index};
use crate::float::Float;

/// Dense 2-D **correlation** of `image` with `kernel` (the kernel is
/// *not* flipped). With
/// `center = (kernel.nrows() / 2, kernel.ncols() / 2)`,
/// ```text
/// out[i,j] = Σ_{a,b} kernel[a,b]
///                    * sample(image, i + a - center.0, j + b - center.1)
/// ```
/// over every `(i, j)` of `image` ('same'-size output). `sample`
/// resolves out-of-bounds indices per axis via `boundary`.
///
/// Pure computational primitive: only `Σ w·v`, **NaN-naive** — a NaN (or
/// ±inf) inside a kernel window propagates straight into that output
/// element (ROADMAP D3/D5). True convolution flips the kernel before
/// calling this (see `image::convolve_psf`); for a symmetric kernel the
/// two coincide.
///
/// Evaluated **directly**, parallelized across output rows with rayon.
/// Per ROADMAP D6 there is deliberately no FFT and no separable path in
/// v1 — those are benchmark-driven, not added preemptively, and the
/// crossover (set by *kernel* side length, not image size) is
/// hardware-dependent. Output rows are independent and every pixel
/// accumulates its taps in a fixed order (row-major over the kernel), so
/// the result is **bitwise-identical** to a sequential evaluation
/// regardless of how rayon schedules the rows (cf.
/// [`crate::image::reproject_exact`]).
///
/// Panics if `kernel` has a zero dimension.
pub fn conv2d<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
    boundary: Boundary,
) -> Array2<T> {
    assert!(
        kernel.nrows() > 0 && kernel.ncols() > 0,
        "conv2d kernel must be non-empty"
    );
    let mut output = Array2::<T>::zeros(image.raw_dim());
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row_index, output_row)| {
            correlate_row(row_index, output_row, image, kernel, boundary);
        });
    output
}

/// Sequential driver, retained for the parallel-vs-sequential
/// equivalence test. It must reuse the exact same per-row work unit so
/// the rayon driver cannot change the result bitwise.
#[cfg(test)]
fn conv2d_sequential<T: Float>(
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
    boundary: Boundary,
) -> Array2<T> {
    assert!(
        kernel.nrows() > 0 && kernel.ncols() > 0,
        "conv2d kernel must be non-empty"
    );
    let mut output = Array2::<T>::zeros(image.raw_dim());
    for (row_index, output_row) in output.axis_iter_mut(Axis(0)).enumerate() {
        correlate_row(row_index, output_row, image, kernel, boundary);
    }
    output
}

/// Compute one output row in place. Factored out of the drivers so the
/// parallel and sequential paths share the identical per-row work unit
/// (the structural guarantee behind the bitwise equivalence test).
fn correlate_row<T: Float>(
    row_index: usize,
    mut output_row: ArrayViewMut1<T>,
    image: ArrayView2<T>,
    kernel: ArrayView2<T>,
    boundary: Boundary,
) {
    let height = image.nrows();
    let width = image.ncols();
    let kernel_rows = kernel.nrows();
    let kernel_cols = kernel.ncols();
    let center_row = kernel_rows / 2;
    let center_col = kernel_cols / 2;
    for column_index in 0..width {
        let mut accumulator = T::zero();
        // Fixed row-major tap order keeps the floating-point
        // accumulation deterministic across the parallel/sequential
        // drivers.
        for kernel_row in 0..kernel_rows {
            let source_row = row_index as isize + kernel_row as isize - center_row as isize;
            let Some(image_row) = resolve_index(source_row, height, boundary) else {
                continue;
            };
            for kernel_col in 0..kernel_cols {
                let source_col = column_index as isize + kernel_col as isize - center_col as isize;
                let Some(image_col) = resolve_index(source_col, width, boundary) else {
                    continue;
                };
                accumulator =
                    accumulator + kernel[(kernel_row, kernel_col)] * image[(image_row, image_col)];
            }
        }
        output_row[column_index] = accumulator;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

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
        let image = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let kernel = array![[1.0_f64]];
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let output = conv2d(image.view(), kernel.view(), boundary);
            for i in 0..image.nrows() {
                for j in 0..image.ncols() {
                    assert!(approx_eq_f64(output[(i, j)], image[(i, j)]));
                }
            }
        }
    }

    #[test]
    fn delta_kernel_is_identity_including_edges_f64() {
        // 3x3 centered delta: every output pixel picks exactly its own
        // input, so identity holds even at the borders for any boundary.
        let image = array![[7.0_f64, 1.0, -3.0], [9.0, 2.0, 4.0], [0.5, 8.0, 6.0]];
        let kernel = array![[0.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let output = conv2d(image.view(), kernel.view(), boundary);
            for i in 0..image.nrows() {
                for j in 0..image.ncols() {
                    assert!(approx_eq_f64(output[(i, j)], image[(i, j)]));
                }
            }
        }
    }

    #[test]
    fn correlation_does_not_flip_kernel_f64() {
        // Asymmetric kernel pins ROADMAP D3 (correlation, not
        // convolution). Single bright pixel; correlation places the
        // kernel as-is, convolution would place it mirrored.
        let mut image = Array2::<f64>::zeros((5, 5));
        image[(2, 2)] = 1.0;
        // Distinct off-center weights so a flip is detectable.
        let kernel = array![[0.0_f64, 0.0, 0.0], [0.0, 0.0, 7.0], [0.0, 0.0, 0.0]];
        let output = conv2d(image.view(), kernel.view(), Boundary::Zero);
        // kernel center = (1, 1); the weight 7 sits at (1, 2) i.e. one
        // column right of center. Correlation: out[i,j] picks
        // image[i, j+1], so the impulse at (2,2) appears at (2,1).
        assert!(approx_eq_f64(output[(2, 1)], 7.0));
        // Convolution would have put it at (2, 3); assert it did not.
        assert!(approx_eq_f64(output[(2, 3)], 0.0));
    }

    #[test]
    fn symmetric_kernel_correlation_equals_convolution_f64() {
        let image = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        // Symmetric under a 180-degree flip of both axes.
        let kernel = array![[1.0_f64, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];
        let mut flipped = kernel.clone();
        flipped.invert_axis(Axis(0));
        flipped.invert_axis(Axis(1));
        for boundary in [Boundary::Zero, Boundary::Reflect, Boundary::Nearest] {
            let a = conv2d(image.view(), kernel.view(), boundary);
            let b = conv2d(image.view(), flipped.view(), boundary);
            for i in 0..image.nrows() {
                for j in 0..image.ncols() {
                    assert!(approx_eq_f64(a[(i, j)], b[(i, j)]));
                }
            }
        }
    }

    #[test]
    fn sum_kernel_preserves_constant_with_nearest_f64() {
        let image = Array2::<f64>::from_elem((6, 7), 3.5);
        let ninth = 1.0 / 9.0;
        let kernel = Array2::<f64>::from_elem((3, 3), ninth);
        let output = conv2d(image.view(), kernel.view(), Boundary::Nearest);
        for value in output.iter() {
            assert!(approx_eq_f64(*value, 3.5));
        }
    }

    #[test]
    fn nan_is_naive_propagates_f64() {
        let mut image = Array2::<f64>::from_elem((4, 4), 1.0);
        image[(1, 1)] = f64::NAN;
        let kernel = Array2::<f64>::from_elem((3, 3), 1.0);
        let output = conv2d(image.view(), kernel.view(), Boundary::Zero);
        // Every output pixel whose 3x3 window covers (1,1) is NaN.
        for i in 0..3 {
            for j in 0..3 {
                assert!(output[(i, j)].is_nan(), "expected NaN at ({i},{j})");
            }
        }
        // (3,3)'s window is rows 2..4, cols 2..4 — clear of the NaN.
        assert!(approx_eq_f64(output[(3, 3)], 4.0));
    }

    #[test]
    #[should_panic(expected = "kernel must be non-empty")]
    fn empty_kernel_panics() {
        let image = Array2::<f64>::zeros((3, 3));
        let kernel = Array2::<f64>::zeros((0, 3));
        let _ = conv2d(image.view(), kernel.view(), Boundary::Zero);
    }

    #[test]
    fn parallel_matches_sequential_bitwise_on_moderate_input() {
        // Large enough that rayon actually splits rows; an asymmetric
        // kernel and Reflect boundary so every pixel does real work.
        let height = 41usize;
        let width = 48usize;
        let mut image = Array2::<f64>::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                image[(i, j)] = ((i * 13 + j * 7) % 97) as f64 / 11.0 + (i as f64).sin();
            }
        }
        let kernel = array![
            [0.2_f64, -0.1, 0.3, 0.05, 0.0],
            [0.1, 0.4, -0.2, 0.07, 0.15],
            [-0.05, 0.25, 0.5, -0.3, 0.02],
            [0.0, 0.12, -0.08, 0.33, 0.09],
            [0.04, -0.06, 0.11, 0.01, -0.13]
        ];
        let parallel = conv2d(image.view(), kernel.view(), Boundary::Reflect);
        let sequential = conv2d_sequential(image.view(), kernel.view(), Boundary::Reflect);
        assert_eq!(parallel.shape(), sequential.shape());
        for i in 0..height {
            for j in 0..width {
                assert_eq!(
                    parallel[(i, j)].to_bits(),
                    sequential[(i, j)].to_bits(),
                    "mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn single_tap_kernel_is_identity_f32() {
        let image = array![[1.0_f32, -2.0], [3.5, 4.0], [5.0, 6.0]];
        let kernel = array![[1.0_f32]];
        let output = conv2d(image.view(), kernel.view(), Boundary::Nearest);
        for i in 0..image.nrows() {
            for j in 0..image.ncols() {
                assert!(approx_eq_f32(output[(i, j)], image[(i, j)]));
            }
        }
    }

    #[test]
    fn correlation_known_values_f32() {
        // 3x3 box, Zero boundary: interior pixel is the full 9-sum,
        // corner sees only a 2x2 in-bounds window.
        let image = Array2::<f32>::from_elem((3, 3), 2.0);
        let kernel = Array2::<f32>::from_elem((3, 3), 1.0);
        let output = conv2d(image.view(), kernel.view(), Boundary::Zero);
        assert!(approx_eq_f32(output[(1, 1)], 18.0)); // 9 * 2
        assert!(approx_eq_f32(output[(0, 0)], 8.0)); // 4 * 2
        assert!(approx_eq_f32(output[(0, 1)], 12.0)); // 6 * 2
    }
}

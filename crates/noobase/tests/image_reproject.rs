//! Cross-dtype integration tests for `image::reproject_exact`.
//!
//! The kernel always runs in `f64`, with `f32` inputs upcast at the
//! entry point. These tests pin that contract: the same image expressed
//! as `f32` or `f64` produces results that agree to within `f32`
//! precision, and the output is always `f64` regardless of the input
//! dtype.

use ndarray::{Array2, Array3};
use noobase::image::reproject_exact;

fn shifted_identity_corners(height_out: usize, width_out: usize, shift_x: f64, shift_y: f64) -> Array3<f64> {
    let mut corners = Array3::<f64>::zeros((height_out + 1, width_out + 1, 2));
    for i_node in 0..=height_out {
        for j_node in 0..=width_out {
            corners[(i_node, j_node, 0)] = j_node as f64 - 0.5 + shift_x;
            corners[(i_node, j_node, 1)] = i_node as f64 - 0.5 + shift_y;
        }
    }
    corners
}

#[test]
fn f32_and_f64_inputs_agree_within_f32_precision() {
    // Mildly non-trivial test image: a ramp.
    let height = 8usize;
    let width = 10usize;
    let mut image_f64: Array2<f64> = Array2::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            image_f64[(i, j)] = (i as f64) * 0.5 + (j as f64) * 0.25 + 1.0;
        }
    }
    let image_f32: Array2<f32> = image_f64.mapv(|value| value as f32);

    // Sub-pixel shift to force genuine area-weighted averaging across
    // multiple input pixels.
    let corners = shifted_identity_corners(height, width, 0.3, -0.2);

    let output_f64 = reproject_exact(image_f64.view(), corners.view()).unwrap();
    let output_f32 = reproject_exact(image_f32.view(), corners.view()).unwrap();

    assert_eq!(output_f64.image.shape(), &[height, width]);
    assert_eq!(output_f32.image.shape(), &[height, width]);

    let tolerance = 1e-5;
    for i in 0..height {
        for j in 0..width {
            let value_f64 = output_f64.image[(i, j)];
            let value_f32 = output_f32.image[(i, j)];
            let weight_f64 = output_f64.weight[(i, j)];
            let weight_f32 = output_f32.weight[(i, j)];

            // Weight only depends on the corner geometry, so f32 and
            // f64 paths must produce *identical* weights.
            assert_eq!(weight_f64, weight_f32, "weight mismatch at ({i}, {j})");

            if value_f64.is_nan() {
                assert!(value_f32.is_nan());
                continue;
            }
            let diff = (value_f64 - value_f32).abs();
            let scale = value_f64.abs().max(value_f32.abs()).max(1.0);
            assert!(
                diff <= tolerance * scale,
                "f32/f64 mismatch at ({i}, {j}): {value_f64} vs {value_f32}"
            );
        }
    }
}

#[test]
fn output_dtype_is_always_f64_for_f32_input() {
    // Statically guaranteed by the return type; this test exists
    // mostly to document the invariant and pin the API.
    let image: Array2<f32> = Array2::from_elem((2, 2), 1.5_f32);
    let corners = shifted_identity_corners(2, 2, 0.0, 0.0);
    let output = reproject_exact(image.view(), corners.view()).unwrap();
    let _check_dtype: &Array2<f64> = &output.image;
    let _check_weight_dtype: &Array2<f64> = &output.weight;
}

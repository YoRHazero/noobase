//! Surface-brightness-conserving exact image reprojection.
//!
//! The caller supplies an input image on its own pixel grid plus a corner
//! field that gives, for every node of the *output* pixel grid, the
//! corresponding location in the input image's pixel coordinates. The
//! reprojector then computes each output pixel by polygon-clipping the
//! input pixels that fall under the quadrilateral defined by the four
//! output corners and taking the area-weighted mean of the contributing
//! input values.
//!
//! Design notes:
//!
//! - **No WCS.** The corner field is plain pixel coordinates. The caller
//!   is expected to derive it from whichever WCS toolkit they prefer
//!   (astropy.wcs, gwcs, etc.) and hand the result in as an ndarray.
//!   Spherical curvature is therefore *not* corrected — the residual
//!   error is the curvature of the input projection over a single
//!   output pixel, which is negligible for any reasonable astronomical
//!   image.
//!
//! - **Pixel coordinate convention.** Integer `(x, y) = (column, row)`
//!   refers to the *center* of pixel `(row=y, column=x)` (matching
//!   astropy). Pixel `(i, j)` therefore occupies the continuous square
//!   `[j - 0.5, j + 0.5] x [i - 0.5, i + 0.5]`. Internally the
//!   reprojector applies a `+0.5` shift to every corner so input pixels
//!   align with integer-aligned unit cells `[j, j + 1] x [i, i + 1]`;
//!   this shift is invisible to the caller.
//!
//! - **Corner indexing.** Output pixel `(i_out, j_out)` is bounded by
//!   `pixel_corners[i_out, j_out]`, `[i_out, j_out + 1]`,
//!   `[i_out + 1, j_out + 1]`, `[i_out + 1, j_out]`, in that winding
//!   order. Adjacent output pixels share corners by construction, which
//!   makes the polygon partition watertight.
//!
//! - **NaN semantics.** Any NaN in the four corners of an output pixel
//!   produces `(NaN, 0)` for that output. NaN values inside the
//!   contributing input pixels are excluded from both numerator and
//!   denominator (NaN-as-mask).
//!
//! - **Surface brightness conservation.** The default reduction is the
//!   area-weighted mean, which conserves surface brightness. For total
//!   flux, multiply the output image by the returned weight and by the
//!   output pixel area in input-pixel units.

use ndarray::{Array2, ArrayView2, ArrayView3, ArrayViewMut1, Axis};
use rayon::prelude::*;
use thiserror::Error;

use crate::float::Float;
use crate::image::polygon::{Point, clip_quad_against_unit_cell, signed_area};

/// Output of [`reproject_exact`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReprojectExactOutput {
    /// Surface-brightness-conserving reprojection of `image_in`.
    /// Shape: `(H_out, W_out)`. Pixels that received no valid input
    /// contribution are `NaN`.
    pub image: Array2<f64>,
    /// Per-output-pixel overlap fraction in `[0, 1]`: the ratio of the
    /// in-bounds, non-NaN input area to the area of the output pixel
    /// projected into input-pixel space. Shape: `(H_out, W_out)`.
    pub weight: Array2<f64>,
}

/// Errors returned by [`reproject_exact`] for ill-shaped inputs.
///
/// Algorithmic edge cases (out-of-footprint output pixels, NaN corners,
/// NaN input pixels, degenerate quadrilaterals) are *not* errors — they
/// fall out of the NaN / zero-weight conventions documented on
/// [`reproject_exact`].
#[derive(Debug, Error, PartialEq)]
pub enum ReprojectError {
    #[error(
        "pixel_corners must have shape (H_out + 1, W_out + 1, 2) with last dim == 2 and both grid dims >= 2; got ({rows}, {cols}, {last})"
    )]
    PixelCornersShape {
        rows: usize,
        cols: usize,
        last: usize,
    },
}

/// Surface-brightness-conserving exact reprojection of a 2-D image.
///
/// See the module-level documentation for the coordinate convention,
/// corner indexing rule, NaN semantics, and conservation property.
///
/// # Parameters
///
/// - `image_in`: input image of shape `(H_in, W_in)`. NaN values are
///   treated as mask (excluded from both numerator and denominator).
///   May be `f32` or `f64`; internally upcast to `f64` for the kernel.
/// - `pixel_corners`: shape `(H_out + 1, W_out + 1, 2)`. The last
///   dimension stores `[x_in, y_in]`, i.e. the location of the
///   corresponding output-pixel corner in the input image's continuous
///   pixel coordinates (integer = pixel center). NaN corners propagate
///   to `(NaN, 0)` in the output for any output pixel that touches
///   them.
///
/// # Returns
///
/// `(image_out, weight)` both shape `(H_out, W_out)` and dtype `f64`.
///
/// # Errors
///
/// Returns [`ReprojectError::PixelCornersShape`] if `pixel_corners`
/// does not have last dim `2`, or if either of its front dimensions is
/// below `2` (which would produce zero output pixels in that
/// direction).
pub fn reproject_exact<T: Float>(
    image_in: ArrayView2<T>,
    pixel_corners: ArrayView3<f64>,
) -> Result<ReprojectExactOutput, ReprojectError> {
    let corner_shape = pixel_corners.shape();
    let rows = corner_shape[0];
    let cols = corner_shape[1];
    let last = corner_shape[2];
    if last != 2 || rows < 2 || cols < 2 {
        return Err(ReprojectError::PixelCornersShape { rows, cols, last });
    }

    // Upcast input image to f64 once, up front. The kernel is f64-only
    // because the polygon math is f64; carrying T into the kernel would
    // either force a generic kernel (more code, no benefit) or repeat
    // the cast inside the hot loop.
    let image_in_f64: Array2<f64> = image_in.mapv(|value| value.to_f64().unwrap_or(f64::NAN));

    let height_out = rows - 1;
    let width_out = cols - 1;
    let mut image_out = Array2::<f64>::from_elem((height_out, width_out), f64::NAN);
    let mut weight = Array2::<f64>::zeros((height_out, width_out));

    // Rows are independent: each output row only reads `image_in_f64`
    // and the slice of `pixel_corners` between rows `i_out` and
    // `i_out + 1`, and only writes its own row of `image_out` /
    // `weight`. Drive them in parallel via rayon.
    image_out
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(weight.axis_iter_mut(Axis(0)).into_par_iter())
        .enumerate()
        .for_each(|(row_index, (row_image, row_weight))| {
            process_output_row(
                row_index,
                row_image,
                row_weight,
                image_in_f64.view(),
                pixel_corners,
            );
        });

    Ok(ReprojectExactOutput {
        image: image_out,
        weight,
    })
}

/// Sequential driver retained for tests. Used by the parallel-vs-
/// sequential equivalence test to confirm the rayon driver does not
/// change the result bitwise.
#[cfg(test)]
fn reproject_exact_sequential<T: Float>(
    image_in: ArrayView2<T>,
    pixel_corners: ArrayView3<f64>,
) -> Result<ReprojectExactOutput, ReprojectError> {
    let corner_shape = pixel_corners.shape();
    let rows = corner_shape[0];
    let cols = corner_shape[1];
    let last = corner_shape[2];
    if last != 2 || rows < 2 || cols < 2 {
        return Err(ReprojectError::PixelCornersShape { rows, cols, last });
    }
    let image_in_f64: Array2<f64> = image_in.mapv(|value| value.to_f64().unwrap_or(f64::NAN));
    let height_out = rows - 1;
    let width_out = cols - 1;
    let mut image_out = Array2::<f64>::from_elem((height_out, width_out), f64::NAN);
    let mut weight = Array2::<f64>::zeros((height_out, width_out));
    for (row_index, (row_image, row_weight)) in image_out
        .outer_iter_mut()
        .zip(weight.outer_iter_mut())
        .enumerate()
    {
        process_output_row(
            row_index,
            row_image,
            row_weight,
            image_in_f64.view(),
            pixel_corners,
        );
    }
    Ok(ReprojectExactOutput {
        image: image_out,
        weight,
    })
}

/// Compute one output row in place. Factored out of the driver so a
/// parallel driver can reuse the exact same per-row work unit.
pub(crate) fn process_output_row(
    row_index: usize,
    mut row_image: ArrayViewMut1<f64>,
    mut row_weight: ArrayViewMut1<f64>,
    image_in_f64: ArrayView2<f64>,
    pixel_corners: ArrayView3<f64>,
) {
    let height_in = image_in_f64.shape()[0];
    let width_in = image_in_f64.shape()[1];
    let width_out = row_image.len();

    for column_index in 0..width_out {
        // Read the four corners that bound output pixel
        // (row_index, column_index). Winding (CCW or CW) is irrelevant
        // because we take the absolute area; the only requirement is
        // that consecutive indices walk around the quad.
        let corner_tl = corner_at(pixel_corners, row_index, column_index);
        let corner_tr = corner_at(pixel_corners, row_index, column_index + 1);
        let corner_br = corner_at(pixel_corners, row_index + 1, column_index + 1);
        let corner_bl = corner_at(pixel_corners, row_index + 1, column_index);

        let any_nan_corner = corner_tl[0].is_nan()
            || corner_tl[1].is_nan()
            || corner_tr[0].is_nan()
            || corner_tr[1].is_nan()
            || corner_br[0].is_nan()
            || corner_br[1].is_nan()
            || corner_bl[0].is_nan()
            || corner_bl[1].is_nan();
        if any_nan_corner {
            row_image[column_index] = f64::NAN;
            row_weight[column_index] = 0.0;
            continue;
        }

        // Apply the half-pixel shift so input pixel (i, j) occupies the
        // unit cell [j, j+1] x [i, i+1]. Keeping this internal lets
        // callers reason in astropy's convention (integer = center).
        let quad: [Point; 4] = [
            [corner_tl[0] + 0.5, corner_tl[1] + 0.5],
            [corner_tr[0] + 0.5, corner_tr[1] + 0.5],
            [corner_br[0] + 0.5, corner_br[1] + 0.5],
            [corner_bl[0] + 0.5, corner_bl[1] + 0.5],
        ];

        let output_pixel_area = signed_area(&quad).abs();

        // Bounding box of the (shifted) quad in continuous coords.
        let mut x_min_f = quad[0][0];
        let mut x_max_f = quad[0][0];
        let mut y_min_f = quad[0][1];
        let mut y_max_f = quad[0][1];
        for vertex in quad.iter().skip(1) {
            if vertex[0] < x_min_f {
                x_min_f = vertex[0];
            }
            if vertex[0] > x_max_f {
                x_max_f = vertex[0];
            }
            if vertex[1] < y_min_f {
                y_min_f = vertex[1];
            }
            if vertex[1] > y_max_f {
                y_max_f = vertex[1];
            }
        }

        // Clamp to image bounds. Right-open: pixel (W_in - 1) covers
        // [W_in - 1, W_in] after the shift, so the maximum integer cell
        // index we enumerate is W_in - 1 (inclusive).
        if x_max_f <= 0.0
            || y_max_f <= 0.0
            || x_min_f >= width_in as f64
            || y_min_f >= height_in as f64
        {
            row_image[column_index] = f64::NAN;
            row_weight[column_index] = 0.0;
            continue;
        }

        let column_lo = x_min_f.floor().max(0.0) as i64;
        let column_hi = (x_max_f.ceil() as i64).min(width_in as i64);
        let row_lo = y_min_f.floor().max(0.0) as i64;
        let row_hi = (y_max_f.ceil() as i64).min(height_in as i64);

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for cell_row in row_lo..row_hi {
            for cell_column in column_lo..column_hi {
                let pixel_value =
                    image_in_f64[(cell_row as usize, cell_column as usize)];
                if pixel_value.is_nan() {
                    // NaN-as-mask: drop this input pixel from both the
                    // numerator and the denominator. Other pixels that
                    // overlap the output quad still contribute normally.
                    continue;
                }
                let clipped = clip_quad_against_unit_cell(
                    &quad,
                    (cell_column as i32, cell_row as i32),
                );
                if clipped.is_empty() {
                    continue;
                }
                let overlap_area = signed_area(&clipped).abs();
                if overlap_area == 0.0 {
                    continue;
                }
                numerator += overlap_area * pixel_value;
                denominator += overlap_area;
            }
        }

        if denominator > 0.0 {
            row_image[column_index] = numerator / denominator;
        } else {
            row_image[column_index] = f64::NAN;
        }
        let weight_value = if output_pixel_area > 0.0 {
            (denominator / output_pixel_area).clamp(0.0, 1.0)
        } else {
            0.0
        };
        row_weight[column_index] = weight_value;
    }
}

/// Read the `(x, y)` corner at grid node `(row, column)` out of the
/// corner field.
#[inline]
fn corner_at(pixel_corners: ArrayView3<f64>, row: usize, column: usize) -> [f64; 2] {
    [
        pixel_corners[(row, column, 0)],
        pixel_corners[(row, column, 1)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    /// Build an identity corner field: output grid coincides with input
    /// pixel boundaries. Output pixel (i, j) is bounded by input-pixel
    /// corners at integer + 0.5 spacing (centers convention), but the
    /// caller passes them as the boundary nodes, i.e. integer-aligned
    /// half-pixel-offset coordinates.
    fn identity_corners(height_out: usize, width_out: usize) -> Array3<f64> {
        // Output pixel (i, j) should sample input pixel (i, j). In
        // continuous coords, input pixel (i, j) has center (j, i) and
        // edges at j - 0.5 and i - 0.5. So corner (i_node, j_node) is
        // at (j_node - 0.5, i_node - 0.5).
        let mut corners = Array3::<f64>::zeros((height_out + 1, width_out + 1, 2));
        for i_node in 0..=height_out {
            for j_node in 0..=width_out {
                corners[(i_node, j_node, 0)] = j_node as f64 - 0.5;
                corners[(i_node, j_node, 1)] = i_node as f64 - 0.5;
            }
        }
        corners
    }

    #[test]
    fn identity_reprojection_preserves_image_and_weight_f64() {
        let image: Array2<f64> = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let corners = identity_corners(3, 3);
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(output.image[(i, j)], image[(i, j)], TOL));
                assert!(approx_eq(output.weight[(i, j)], 1.0, TOL));
            }
        }
    }

    #[test]
    fn identity_reprojection_preserves_image_and_weight_f32() {
        let image: Array2<f32> = ndarray::array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ];
        let corners = identity_corners(3, 3);
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(output.image[(i, j)], image[(i, j)] as f64, TOL));
                assert!(approx_eq(output.weight[(i, j)], 1.0, TOL));
            }
        }
    }

    #[test]
    fn constant_image_yields_constant_output_where_weight_positive() {
        let constant_value = 4.2_f64;
        let image: Array2<f64> = Array2::from_elem((5, 5), constant_value);
        // Translate output grid by (0.3, -0.2) in input-pixel units;
        // shape stays the same.
        let mut corners = identity_corners(5, 5);
        for i_node in 0..corners.shape()[0] {
            for j_node in 0..corners.shape()[1] {
                corners[(i_node, j_node, 0)] += 0.3;
                corners[(i_node, j_node, 1)] -= 0.2;
            }
        }
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                if output.weight[(i, j)] > 0.0 {
                    assert!(
                        approx_eq(output.image[(i, j)], constant_value, TOL),
                        "i={i} j={j} got {} weight {}",
                        output.image[(i, j)],
                        output.weight[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn half_pixel_shift_linear_average_of_two_columns() {
        // Image: 1x4. Shift output by +0.5 in x. Each output pixel
        // should average two adjacent input pixels (1:1 mix).
        let image: Array2<f64> = ndarray::array![[1.0, 2.0, 3.0, 4.0]];
        // Identity corners for 1x4 output.
        let mut corners = identity_corners(1, 4);
        for i_node in 0..corners.shape()[0] {
            for j_node in 0..corners.shape()[1] {
                corners[(i_node, j_node, 0)] += 0.5;
            }
        }
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        // Output pixel j has x-extent (j - 0.5 + 0.5, j + 0.5 + 0.5)
        //                              = (j, j + 1).
        // After internal +0.5 shift -> (j + 0.5, j + 1.5).
        // Input pixel j occupies cell [j, j + 1] (after shift).
        // So output pixel 0 overlaps input pixels 0 and 1, each by 0.5.
        // => image_out[0, 0] = (image[0]*0.5 + image[1]*0.5) / 1.0
        //   = 1.5; weight = 1.0.
        assert!(approx_eq(output.image[(0, 0)], 1.5, TOL));
        assert!(approx_eq(output.image[(0, 1)], 2.5, TOL));
        assert!(approx_eq(output.image[(0, 2)], 3.5, TOL));
        // Output pixel 3 overlaps input pixel 3 (half) only; pixel 4
        // does not exist, so weight = 0.5 and image_out = 4.0.
        assert!(approx_eq(output.image[(0, 3)], 4.0, TOL));
        assert!(approx_eq(output.weight[(0, 0)], 1.0, TOL));
        assert!(approx_eq(output.weight[(0, 3)], 0.5, TOL));
    }

    #[test]
    fn rotation_by_90_degrees_preserves_surface_brightness() {
        // Rotate output grid by 90 degrees about the (1, 1) input
        // center. Use a constant image -> output should be the same
        // constant everywhere it is fully covered.
        let constant_value = 7.0_f64;
        let image: Array2<f64> = Array2::from_elem((3, 3), constant_value);
        // For output pixel node (i_node, j_node), instead of mapping
        // 1:1 we rotate -90 degrees: (x_in, y_in) =
        //   (y_node_centered + center_x, -x_node_centered + center_y).
        // With center = (1, 1).
        let mut corners = Array3::<f64>::zeros((4, 4, 2));
        for i_node in 0..4 {
            for j_node in 0..4 {
                let x_orig = j_node as f64 - 0.5;
                let y_orig = i_node as f64 - 0.5;
                let x_centered = x_orig - 1.0;
                let y_centered = y_orig - 1.0;
                let x_rotated = y_centered + 1.0;
                let y_rotated = -x_centered + 1.0;
                corners[(i_node, j_node, 0)] = x_rotated;
                corners[(i_node, j_node, 1)] = y_rotated;
            }
        }
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(output.image[(i, j)], constant_value, TOL));
                assert!(approx_eq(output.weight[(i, j)], 1.0, TOL));
            }
        }
    }

    #[test]
    fn output_pixels_outside_footprint_are_nan_zero() {
        // 2x2 input, output corner field places the whole output
        // grid far away from the input image.
        let image: Array2<f64> = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let mut corners = Array3::<f64>::zeros((3, 3, 2));
        for i_node in 0..3 {
            for j_node in 0..3 {
                corners[(i_node, j_node, 0)] = 100.0 + j_node as f64;
                corners[(i_node, j_node, 1)] = 100.0 + i_node as f64;
            }
        }
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(output.image[(i, j)].is_nan());
                assert_eq!(output.weight[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn nan_input_pixel_is_excluded_from_both_numerator_and_denominator() {
        // 1x3 image with the middle pixel NaN. Output pixel that spans
        // the middle two pixels in equal proportion should reduce to
        // the right-hand pixel's value with weight = 0.5 (only half of
        // the area contributed, because the NaN half is removed from
        // BOTH numerator and denominator -> denominator = 0.5 of the
        // output pixel area).
        let image: Array2<f64> = ndarray::array![[1.0, f64::NAN, 3.0]];
        // Identity 1x3 corners, then shift +0.5 in x: output pixel j
        // spans input pixels j and j+1, 50/50.
        let mut corners = identity_corners(1, 3);
        for i_node in 0..corners.shape()[0] {
            for j_node in 0..corners.shape()[1] {
                corners[(i_node, j_node, 0)] += 0.5;
            }
        }
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        // Output pixel 0 sees input 0 (1.0) + input 1 (NaN). NaN
        // drops; remaining = 1.0 over area 0.5 -> value 1.0,
        // weight 0.5.
        assert!(approx_eq(output.image[(0, 0)], 1.0, TOL));
        assert!(approx_eq(output.weight[(0, 0)], 0.5, TOL));
        // Output pixel 1 sees input 1 (NaN) + input 2 (3.0).
        assert!(approx_eq(output.image[(0, 1)], 3.0, TOL));
        assert!(approx_eq(output.weight[(0, 1)], 0.5, TOL));
    }

    #[test]
    fn nan_in_corners_marks_output_nan_zero() {
        let image: Array2<f64> = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let mut corners = identity_corners(2, 2);
        corners[(1, 1, 0)] = f64::NAN;
        let output = reproject_exact(image.view(), corners.view()).unwrap();
        // The NaN corner is shared by output pixels (0, 0), (0, 1),
        // (1, 0), (1, 1) — all 4. Each must be NaN/0.
        for i in 0..2 {
            for j in 0..2 {
                assert!(output.image[(i, j)].is_nan());
                assert_eq!(output.weight[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn shape_error_last_dim_not_two() {
        let image: Array2<f64> = Array2::zeros((2, 2));
        let corners = Array3::<f64>::zeros((3, 3, 3));
        let err = reproject_exact(image.view(), corners.view()).unwrap_err();
        assert_eq!(
            err,
            ReprojectError::PixelCornersShape {
                rows: 3,
                cols: 3,
                last: 3
            }
        );
    }

    #[test]
    fn parallel_matches_sequential_bitwise_on_moderate_input() {
        // Moderate size so rayon actually splits the work, and a
        // sub-pixel-shifted corner field so every output pixel does
        // real area-weighted averaging.
        let height = 32usize;
        let width = 40usize;
        let mut image: Array2<f64> = Array2::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                image[(i, j)] =
                    ((i * 13 + j * 7) % 97) as f64 / 11.0 + (i as f64).sin();
            }
        }
        let mut corners = Array3::<f64>::zeros((height + 1, width + 1, 2));
        for i_node in 0..=height {
            for j_node in 0..=width {
                corners[(i_node, j_node, 0)] = j_node as f64 - 0.5 + 0.37;
                corners[(i_node, j_node, 1)] = i_node as f64 - 0.5 - 0.21;
            }
        }
        let parallel = reproject_exact(image.view(), corners.view()).unwrap();
        let sequential = reproject_exact_sequential(image.view(), corners.view()).unwrap();
        assert_eq!(parallel.image.shape(), sequential.image.shape());
        for i in 0..height {
            for j in 0..width {
                let p_image = parallel.image[(i, j)];
                let s_image = sequential.image[(i, j)];
                if p_image.is_nan() {
                    assert!(s_image.is_nan());
                } else {
                    // Bitwise equality: each output pixel's per-input
                    // accumulation order is fixed (row-major over the
                    // bbox) and lives entirely inside one row task, so
                    // floats accumulate identically regardless of how
                    // rayon schedules rows.
                    assert_eq!(p_image.to_bits(), s_image.to_bits());
                }
                assert_eq!(
                    parallel.weight[(i, j)].to_bits(),
                    sequential.weight[(i, j)].to_bits()
                );
            }
        }
    }

    #[test]
    fn shape_error_front_dims_too_small() {
        let image: Array2<f64> = Array2::zeros((2, 2));
        let corners = Array3::<f64>::zeros((1, 3, 2));
        let err = reproject_exact(image.view(), corners.view()).unwrap_err();
        assert_eq!(
            err,
            ReprojectError::PixelCornersShape {
                rows: 1,
                cols: 3,
                last: 2
            }
        );
    }
}

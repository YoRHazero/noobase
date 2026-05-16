//! Single-image, PSF-independent stamp extraction.
//!
//! Given one caller-pre-sliced rough cutout, this leaf locates an
//! init-quality centroid inside it, picks an integer-aligned odd square
//! window that lies fully inside the cutout, slices the stamp out with
//! *no* interpolation, and records the residual sub-pixel offset.
//!
//! Design notes:
//!
//! - **Pixel coordinate convention.** Integer index = pixel center
//!   (astropy convention, same as `reproject_exact`). All centroid math
//!   is done in cutout-local pixel-index coordinates. The geometric
//!   center of an `(H, W)` cutout is `((H - 1) / 2, (W - 1) / 2)`, which
//!   is the centroid iteration's initial guess.
//!
//! - **Centroid is a rough initializer.** The Gaussian-weighted
//!   first-moment centroid here is deliberately an init-quality rough
//!   estimate, *not* a final astrometric centroid. In particular the
//!   per-pixel weight clips background-subtracted values at zero; this is
//!   correct for a centroid initializer only and is *not* applied to the
//!   output data anywhere.
//!
//! - **Background.** A robust constant background (median of the outer
//!   1-pixel border ring, excluding non-finite and masked pixels) is
//!   subtracted *only* inside the centroid moment computation. The output
//!   `stamp` always carries the original cutout values; no background is
//!   ever baked into the outputs.
//!
//! - **No interpolation.** Once the integer window is chosen the stamp is
//!   a plain slice of the cutout (and of the error array, if provided).
//!   The sub-pixel offset between the float centroid and the integer
//!   window center is reported via `delta` so a downstream consumer can
//!   account for it.
//!
//! - **Algorithmic skips vs errors.** Ill-shaped inputs (even
//!   `stamp_size`, window larger than the cutout, mismatched `error` or
//!   `mask` shapes) are hard preconditions returned as
//!   [`StampError`]. Cases where no usable stamp can be produced (too few
//!   valid pixels, zero total weight, window cannot fit fully inside the
//!   cutout) are *not* errors — they return `Ok(None)`.

use ndarray::{Array2, ArrayView2};
use thiserror::Error;

use crate::float::Float;

/// Default FWHM (pixels) of the Gaussian centroid window.
pub const DEFAULT_WEIGHT_FWHM: f64 = 3.0;
/// Default centroid iteration cap.
pub const DEFAULT_CENTROID_MAX_ITER: usize = 10;
/// Default centroid convergence tolerance (pixels).
pub const DEFAULT_CENTROID_TOL: f64 = 1e-3;

/// Output of [`build_stamp`].
#[derive(Debug, Clone, PartialEq)]
pub struct StampResult {
    /// Extracted stamp of shape `(stamp_size, stamp_size)`. Carries the
    /// *original* cutout values upcast to `f64`; no background is removed.
    pub stamp: Array2<f64>,
    /// Windowed error of shape `(stamp_size, stamp_size)`. `Some` iff the
    /// `error` argument was `Some`; the same window as `stamp` is sliced.
    pub error: Option<Array2<f64>>,
    /// Per-pixel validity of shape `(stamp_size, stamp_size)`. `true`
    /// means the pixel is valid (positive polarity — the opposite of the
    /// input `mask`, whose `true` means invalid).
    pub valid: Array2<bool>,
    /// `(centroid - round(centroid))` per axis `(row, col)`. Each
    /// component lands in `[-0.5, 0.5)`.
    pub delta: (f64, f64),
    /// `(row, col)` of the window's top-left corner, in cutout-local
    /// indices.
    pub origin: (i64, i64),
}

/// Errors returned by [`build_stamp`] for ill-shaped inputs.
///
/// Algorithmic edge cases (too few valid pixels, zero total centroid
/// weight, a window that cannot fit fully inside the cutout) are *not*
/// errors — they yield `Ok(None)` per the conventions documented on
/// [`build_stamp`].
#[derive(Debug, Error, PartialEq)]
pub enum StampError {
    #[error("stamp_size must be odd; got {stamp_size}")]
    StampSizeEven { stamp_size: usize },
    #[error(
        "stamp_size ({stamp_size}) must not exceed cutout dimensions; cutout is ({rows}, {cols})"
    )]
    StampSizeTooLarge {
        stamp_size: usize,
        rows: usize,
        cols: usize,
    },
    #[error(
        "error shape {error_shape:?} must equal cutout shape {cutout_shape:?}"
    )]
    ErrorShapeMismatch {
        error_shape: (usize, usize),
        cutout_shape: (usize, usize),
    },
    #[error(
        "mask shape {mask_shape:?} must equal cutout shape {cutout_shape:?}"
    )]
    MaskShapeMismatch {
        mask_shape: (usize, usize),
        cutout_shape: (usize, usize),
    },
}

/// Locate an init-quality centroid in `cutout`, pick an integer-aligned
/// odd square window fully inside it, and slice the stamp.
///
/// See the module-level documentation for the coordinate convention,
/// background handling, the centroid-initializer caveat, and the
/// algorithmic-skip (`Ok(None)`) vs error conventions.
///
/// # Parameters
///
/// - `cutout`: rough cutout of any shape `(rows, cols)`. May be `f32` or
///   `f64`; internally upcast to `f64`. Non-finite values are excluded
///   from the centroid and marked invalid in `valid`.
/// - `stamp_size`: edge length of the (square) output stamp. Must be odd.
/// - `error`: optional 1-sigma error array. If `Some`, its shape must
///   equal the cutout shape.
/// - `mask`: optional mask where `true` marks an *invalid* (excluded)
///   pixel. If `Some`, its shape must equal the cutout shape.
/// - `weight_fwhm`: FWHM (pixels) of the Gaussian centroid window.
/// - `max_iter`: centroid iteration cap.
/// - `tol`: centroid convergence tolerance (pixels).
///
/// # Returns
///
/// `Ok(Some(StampResult))` on success; `Ok(None)` when no usable stamp
/// can be produced (see the module docs); `Err(StampError)` for the hard
/// shape preconditions.
///
/// # Errors
///
/// Returns a [`StampError`] if `stamp_size` is even, if `stamp_size`
/// exceeds either cutout dimension, or if a provided `error` / `mask`
/// shape does not equal the cutout shape.
pub fn build_stamp<T: Float>(
    cutout: ArrayView2<T>,
    stamp_size: usize,
    error: Option<ArrayView2<T>>,
    mask: Option<ArrayView2<bool>>,
    weight_fwhm: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Option<StampResult>, StampError> {
    let cutout_rows = cutout.shape()[0];
    let cutout_cols = cutout.shape()[1];

    // --- Hard preconditions (returned as Err). ---
    if stamp_size % 2 == 0 {
        return Err(StampError::StampSizeEven { stamp_size });
    }
    if stamp_size > cutout_rows || stamp_size > cutout_cols {
        return Err(StampError::StampSizeTooLarge {
            stamp_size,
            rows: cutout_rows,
            cols: cutout_cols,
        });
    }
    if let Some(error_view) = error.as_ref() {
        let error_shape = (error_view.shape()[0], error_view.shape()[1]);
        if error_shape != (cutout_rows, cutout_cols) {
            return Err(StampError::ErrorShapeMismatch {
                error_shape,
                cutout_shape: (cutout_rows, cutout_cols),
            });
        }
    }
    if let Some(mask_view) = mask.as_ref() {
        let mask_shape = (mask_view.shape()[0], mask_view.shape()[1]);
        if mask_shape != (cutout_rows, cutout_cols) {
            return Err(StampError::MaskShapeMismatch {
                mask_shape,
                cutout_shape: (cutout_rows, cutout_cols),
            });
        }
    }

    // Upcast the cutout to f64 once, up front. The centroid math is
    // f64-only; carrying T through the iteration would just repeat the
    // cast in the hot loop for no benefit.
    let cutout_f64: Array2<f64> = cutout.mapv(|value| value.to_f64().unwrap_or(f64::NAN));

    // Helper: a pixel participates in the centroid iff it is finite and
    // not masked. (Input mask is true = invalid.)
    let is_masked = |row: usize, column: usize| -> bool {
        mask.as_ref()
            .map(|mask_view| mask_view[(row, column)])
            .unwrap_or(false)
    };
    let participates = |row: usize, column: usize| -> bool {
        cutout_f64[(row, column)].is_finite() && !is_masked(row, column)
    };

    // --- Robust background for centroiding ONLY. ---
    // Median of the outer 1-pixel border ring, excluding non-finite and
    // masked pixels. This constant is subtracted only inside the moment
    // computation; it never touches the output stamp.
    let mut border_values: Vec<f64> = Vec::new();
    for column in 0..cutout_cols {
        // Top and bottom rows.
        for &row in &[0usize, cutout_rows - 1] {
            if participates(row, column) {
                border_values.push(cutout_f64[(row, column)]);
            }
        }
    }
    // Left and right columns, excluding the corners already covered by
    // the top/bottom rows.
    if cutout_rows > 2 {
        for row in 1..(cutout_rows - 1) {
            for &column in &[0usize, cutout_cols - 1] {
                if participates(row, column) {
                    border_values.push(cutout_f64[(row, column)]);
                }
            }
        }
    }
    let background = median(&mut border_values);

    // --- Centroid iteration (init-quality rough estimate). ---
    let sigma = weight_fwhm / (2.0 * (2.0 * 2_f64.ln()).sqrt());

    // Geometric-center initial guess in cutout-local pixel coords.
    let mut centroid_row = (cutout_rows as f64 - 1.0) / 2.0;
    let mut centroid_column = (cutout_cols as f64 - 1.0) / 2.0;

    let mut converged_weight_sum = 0.0_f64;
    let mut participating_count: usize = 0;

    for _iteration in 0..max_iter.max(1) {
        let mut weight_sum = 0.0_f64;
        let mut weighted_row_sum = 0.0_f64;
        let mut weighted_column_sum = 0.0_f64;
        let mut count_this_pass: usize = 0;

        for row in 0..cutout_rows {
            for column in 0..cutout_cols {
                if !participates(row, column) {
                    continue;
                }
                count_this_pass += 1;
                let background_subtracted = cutout_f64[(row, column)] - background;
                // Clipping negatives to zero is intentional and correct
                // for a centroid *initializer* only — it is NOT the data
                // model and is applied nowhere else.
                let positive_part = background_subtracted.max(0.0);
                let row_distance = row as f64 - centroid_row;
                let column_distance = column as f64 - centroid_column;
                let squared_distance =
                    row_distance * row_distance + column_distance * column_distance;
                let gaussian = (-0.5 * squared_distance / (sigma * sigma)).exp();
                let pixel_weight = positive_part * gaussian;
                weight_sum += pixel_weight;
                weighted_row_sum += pixel_weight * row as f64;
                weighted_column_sum += pixel_weight * column as f64;
            }
        }

        participating_count = count_this_pass;
        converged_weight_sum = weight_sum;

        if weight_sum <= 0.0 {
            // Zero total weight: no usable centroid. Decided here so the
            // subsequent convergence check never divides by zero.
            break;
        }

        let new_centroid_row = weighted_row_sum / weight_sum;
        let new_centroid_column = weighted_column_sum / weight_sum;
        let shift = ((new_centroid_row - centroid_row).powi(2)
            + (new_centroid_column - centroid_column).powi(2))
        .sqrt();
        centroid_row = new_centroid_row;
        centroid_column = new_centroid_column;
        if shift < tol {
            break;
        }
    }

    // Algorithmic skips (NOT errors): too few participating pixels, or
    // zero total centroid weight.
    if participating_count < 4 || converged_weight_sum <= 0.0 {
        return Ok(None);
    }

    // --- Integer-aligned window. ---
    let center_row = centroid_row.round();
    let center_column = centroid_column.round();
    let half = (stamp_size / 2) as i64;
    let origin_row = center_row as i64 - half;
    let origin_column = center_column as i64 - half;

    // The window must lie fully inside [0, H) x [0, W). If it does not,
    // that is an algorithmic skip, not an error.
    if origin_row < 0
        || origin_column < 0
        || origin_row + stamp_size as i64 > cutout_rows as i64
        || origin_column + stamp_size as i64 > cutout_cols as i64
    {
        return Ok(None);
    }

    let delta_row = centroid_row - center_row;
    let delta_column = centroid_column - center_column;

    // --- Slice the stamp (NO interpolation). ---
    let mut stamp = Array2::<f64>::zeros((stamp_size, stamp_size));
    let mut valid = Array2::<bool>::from_elem((stamp_size, stamp_size), false);
    let mut windowed_error: Option<Array2<f64>> = error
        .as_ref()
        .map(|_| Array2::<f64>::zeros((stamp_size, stamp_size)));

    for a in 0..stamp_size {
        for b in 0..stamp_size {
            let source_row = (origin_row + a as i64) as usize;
            let source_column = (origin_column + b as i64) as usize;
            let value = cutout_f64[(source_row, source_column)];
            stamp[(a, b)] = value;

            let error_ok = match error.as_ref() {
                Some(error_view) => {
                    let error_value =
                        error_view[(source_row, source_column)].to_f64().unwrap_or(f64::NAN);
                    windowed_error.as_mut().unwrap()[(a, b)] = error_value;
                    error_value.is_finite() && error_value > 0.0
                }
                None => true,
            };

            valid[(a, b)] =
                value.is_finite() && !is_masked(source_row, source_column) && error_ok;
        }
    }

    Ok(Some(StampResult {
        stamp,
        error: windowed_error,
        valid,
        delta: (delta_row, delta_column),
        origin: (origin_row, origin_column),
    }))
}

/// Median of the given values, ignoring nothing (the caller is expected
/// to have filtered non-finite / masked entries already). Returns `0.0`
/// for an empty slice so an all-masked border contributes no background
/// offset. Mutates the slice (in-place partial sort).
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let length = values.len();
    if length % 2 == 1 {
        values[length / 2]
    } else {
        0.5 * (values[length / 2 - 1] + values[length / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const TOL: f64 = 1e-9;

    /// Build a cutout with a single Gaussian point source centered at
    /// `(source_row, source_column)` plus a flat background.
    fn gaussian_cutout(
        rows: usize,
        cols: usize,
        source_row: f64,
        source_column: f64,
        amplitude: f64,
        sigma: f64,
        background: f64,
    ) -> Array2<f64> {
        let mut image = Array2::<f64>::from_elem((rows, cols), background);
        for row in 0..rows {
            for column in 0..cols {
                let row_distance = row as f64 - source_row;
                let column_distance = column as f64 - source_column;
                let squared_distance =
                    row_distance * row_distance + column_distance * column_distance;
                image[(row, column)] +=
                    amplitude * (-0.5 * squared_distance / (sigma * sigma)).exp();
            }
        }
        image
    }

    #[test]
    fn centered_point_source_zero_delta_f64() {
        // 11x11 cutout, source exactly at the geometric center (5, 5).
        let cutout = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let result = build_stamp(
            cutout.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        assert!(result.delta.0.abs() < 1e-6, "delta_row = {}", result.delta.0);
        assert!(
            result.delta.1.abs() < 1e-6,
            "delta_col = {}",
            result.delta.1
        );
        // half = 2, center = (5, 5) -> origin = (3, 3).
        assert_eq!(result.origin, (3, 3));
        // Stamp must carry ORIGINAL values (background not removed):
        // center pixel = 1.0 + 100.0 = 101.0.
        assert!((result.stamp[(2, 2)] - 101.0).abs() < TOL);
        // Corner of the 5x5 window corresponds to cutout (3, 3).
        assert!((result.stamp[(0, 0)] - cutout[(3, 3)]).abs() < TOL);
        // No error provided -> error is None.
        assert!(result.error.is_none());
        // All finite, unmasked, no error term -> all valid.
        assert!(result.valid.iter().all(|&v| v));
    }

    #[test]
    fn centered_point_source_zero_delta_f32() {
        let cutout_f64 = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let result = build_stamp(
            cutout_f32.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        assert!(result.delta.0.abs() < 1e-4, "delta_row = {}", result.delta.0);
        assert!(
            result.delta.1.abs() < 1e-4,
            "delta_col = {}",
            result.delta.1
        );
        assert_eq!(result.origin, (3, 3));
        // f32 cast then upcast to f64.
        assert!((result.stamp[(2, 2)] - 101.0).abs() < 1e-3);
        assert!(result.error.is_none());
        assert!(result.valid.iter().all(|&v| v));
    }

    #[test]
    fn off_center_source_integer_window_and_delta_range_f64() {
        // Source at (6.3, 4.7) in a 13x13 cutout.
        let cutout = gaussian_cutout(13, 13, 6.3, 4.7, 80.0, 1.5, 2.0);
        let result = build_stamp(
            cutout.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        // delta components must land in [-0.5, 0.5).
        assert!(
            result.delta.0 >= -0.5 && result.delta.0 < 0.5,
            "delta_row = {}",
            result.delta.0
        );
        assert!(
            result.delta.1 >= -0.5 && result.delta.1 < 0.5,
            "delta_col = {}",
            result.delta.1
        );
        // half = 2, so the integer window center is origin + half.
        let center_row = result.origin.0 + 2;
        let center_column = result.origin.1 + 2;
        // Reconstruct the float centroid: centroid = center + delta.
        // It must sit near the true source location (6.3, 4.7) for this
        // well-isolated point source.
        let centroid_row = center_row as f64 + result.delta.0;
        let centroid_column = center_column as f64 + result.delta.1;
        assert!(
            (centroid_row - 6.3).abs() < 0.3,
            "centroid_row = {centroid_row}"
        );
        assert!(
            (centroid_column - 4.7).abs() < 0.3,
            "centroid_col = {centroid_column}"
        );
        // delta is exactly centroid - round(centroid) per axis.
        assert!((result.delta.0 - (centroid_row - centroid_row.round())).abs() < TOL);
        assert!(
            (result.delta.1 - (centroid_column - centroid_column.round())).abs() < TOL
        );
    }

    #[test]
    fn off_center_source_integer_window_and_delta_range_f32() {
        let cutout_f64 = gaussian_cutout(13, 13, 6.3, 4.7, 80.0, 1.5, 2.0);
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let result = build_stamp(
            cutout_f32.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        assert!(result.delta.0 >= -0.5 && result.delta.0 < 0.5);
        assert!(result.delta.1 >= -0.5 && result.delta.1 < 0.5);
        let center_row = result.origin.0 + 2;
        let center_column = result.origin.1 + 2;
        let centroid_row = center_row as f64 + result.delta.0;
        let centroid_column = center_column as f64 + result.delta.1;
        assert!((centroid_row - 6.3).abs() < 0.3);
        assert!((centroid_column - 4.7).abs() < 0.3);
    }

    #[test]
    fn window_cannot_fit_returns_none_f64() {
        // Source pinned to the top-left corner of a 9x9 cutout with a
        // large stamp_size; the window cannot be fully inside.
        let cutout = gaussian_cutout(9, 9, 0.0, 0.0, 100.0, 1.0, 0.0);
        let result = build_stamp(
            cutout.view(),
            7,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn window_cannot_fit_returns_none_f32() {
        let cutout_f64 = gaussian_cutout(9, 9, 0.0, 0.0, 100.0, 1.0, 0.0);
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let result = build_stamp(
            cutout_f32.view(),
            7,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn too_few_valid_pixels_returns_none_f64() {
        // 5x5 cutout, almost all masked: only 3 unmasked finite pixels.
        let cutout = Array2::<f64>::from_elem((5, 5), 10.0);
        let mut mask = Array2::<bool>::from_elem((5, 5), true);
        mask[(2, 2)] = false;
        mask[(2, 3)] = false;
        mask[(3, 2)] = false;
        let result = build_stamp(
            cutout.view(),
            3,
            None,
            Some(mask.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn too_few_valid_pixels_returns_none_f32() {
        let cutout = Array2::<f32>::from_elem((5, 5), 10.0);
        let mut mask = Array2::<bool>::from_elem((5, 5), true);
        mask[(2, 2)] = false;
        mask[(2, 3)] = false;
        mask[(3, 2)] = false;
        let result = build_stamp(
            cutout.view(),
            3,
            None,
            Some(mask.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn mask_excludes_pixels_from_centroid_and_marks_invalid_f64() {
        // Bright contaminant at (1, 1) plus a real source at the center.
        // Without masking, the contaminant would pull the centroid.
        let mut cutout = gaussian_cutout(11, 11, 5.0, 5.0, 50.0, 1.5, 1.0);
        cutout[(1, 1)] = 10_000.0;
        let mut mask = Array2::<bool>::from_elem((11, 11), false);
        mask[(1, 1)] = true; // true = invalid

        let result = build_stamp(
            cutout.view(),
            5,
            None,
            Some(mask.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        // Centroid should stay near the true center (contaminant
        // excluded), so the window is centered at (5, 5) -> origin
        // (3, 3), and the masked pixel (1, 1) is outside this window.
        assert_eq!(result.origin, (3, 3));
        assert!(result.delta.0.abs() < 0.5 && result.delta.1.abs() < 0.5);

        // Now mask a pixel that lands INSIDE the window to check the
        // valid map polarity.
        let mut mask2 = Array2::<bool>::from_elem((11, 11), false);
        mask2[(4, 4)] = true; // inside the 5x5 window at origin (3, 3)
        let result2 = build_stamp(
            cutout.view(),
            5,
            None,
            Some(mask2.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        // Window origin is (3, 3); cutout (4, 4) -> stamp (1, 1).
        assert!(
            !result2.valid[(1, 1)],
            "masked pixel must be valid=false"
        );
        // A neighbouring unmasked finite pixel stays valid.
        assert!(result2.valid[(0, 0)]);
    }

    #[test]
    fn mask_excludes_pixels_from_centroid_and_marks_invalid_f32() {
        let mut cutout_f64 = gaussian_cutout(11, 11, 5.0, 5.0, 50.0, 1.5, 1.0);
        cutout_f64[(1, 1)] = 10_000.0;
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let mut mask = Array2::<bool>::from_elem((11, 11), false);
        mask[(1, 1)] = true;

        let result = build_stamp(
            cutout_f32.view(),
            5,
            None,
            Some(mask.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        assert_eq!(result.origin, (3, 3));
    }

    #[test]
    fn error_none_means_result_error_none_and_valid_ignores_error_f64() {
        let cutout = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let result = build_stamp(
            cutout.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        assert!(result.error.is_none());
        // Without an error array, validity only depends on finiteness
        // and mask -> all valid here.
        assert!(result.valid.iter().all(|&v| v));
    }

    #[test]
    fn error_none_means_result_error_none_and_valid_ignores_error_f32() {
        let cutout_f64 = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let result = build_stamp(
            cutout_f32.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        assert!(result.error.is_none());
        assert!(result.valid.iter().all(|&v| v));
    }

    #[test]
    fn error_some_returns_windowed_error_and_valid_requires_positive_f64() {
        let cutout = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let mut error = Array2::<f64>::from_elem((11, 11), 2.0);
        // A non-positive error inside the window must mark valid=false.
        error[(5, 5)] = 0.0;
        // A non-finite error inside the window must mark valid=false.
        error[(4, 4)] = f64::NAN;

        let result = build_stamp(
            cutout.view(),
            5,
            Some(error.view()),
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        let windowed_error = result.error.as_ref().expect("error must be Some");
        // Window origin (3, 3); cutout (3, 3) -> stamp (0, 0); error=2.0.
        assert!((windowed_error[(0, 0)] - 2.0).abs() < TOL);
        // cutout (5, 5) -> stamp (2, 2): error 0.0 -> valid=false.
        assert!((windowed_error[(2, 2)] - 0.0).abs() < TOL);
        assert!(!result.valid[(2, 2)]);
        // cutout (4, 4) -> stamp (1, 1): error NaN -> valid=false.
        assert!(windowed_error[(1, 1)].is_nan());
        assert!(!result.valid[(1, 1)]);
        // A pixel with positive finite error stays valid.
        assert!(result.valid[(0, 0)]);
    }

    #[test]
    fn error_some_returns_windowed_error_and_valid_requires_positive_f32() {
        let cutout_f64 = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let mut error = Array2::<f32>::from_elem((11, 11), 2.0);
        error[(5, 5)] = 0.0;
        error[(4, 4)] = f32::NAN;

        let result = build_stamp(
            cutout_f32.view(),
            5,
            Some(error.view()),
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");

        let windowed_error = result.error.as_ref().expect("error must be Some");
        assert!((windowed_error[(0, 0)] - 2.0).abs() < 1e-5);
        assert!(!result.valid[(2, 2)]);
        assert!(!result.valid[(1, 1)]);
        assert!(result.valid[(0, 0)]);
    }

    #[test]
    fn nan_in_cutout_excluded_from_centroid_and_invalid_f64() {
        let mut cutout = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        // NaN inside what will be the window.
        cutout[(5, 6)] = f64::NAN;
        let result = build_stamp(
            cutout.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        // Origin (3, 3); cutout (5, 6) -> stamp (2, 3).
        assert!(result.stamp[(2, 3)].is_nan());
        assert!(!result.valid[(2, 3)], "NaN pixel must be valid=false");
        // Surrounding finite pixels remain valid.
        assert!(result.valid[(0, 0)]);
    }

    #[test]
    fn nan_in_cutout_excluded_from_centroid_and_invalid_f32() {
        let mut cutout_f64 = gaussian_cutout(11, 11, 5.0, 5.0, 100.0, 1.5, 1.0);
        cutout_f64[(5, 6)] = f64::NAN;
        let cutout_f32: Array2<f32> = cutout_f64.mapv(|v| v as f32);
        let result = build_stamp(
            cutout_f32.view(),
            5,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap()
        .expect("expected a stamp");
        assert!(result.stamp[(2, 3)].is_nan());
        assert!(!result.valid[(2, 3)]);
        assert!(result.valid[(0, 0)]);
    }

    #[test]
    fn error_precondition_stamp_size_even() {
        let cutout = Array2::<f64>::zeros((9, 9));
        let err = build_stamp(
            cutout.view(),
            4,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap_err();
        assert_eq!(err, StampError::StampSizeEven { stamp_size: 4 });
    }

    #[test]
    fn error_precondition_stamp_size_too_large() {
        let cutout = Array2::<f64>::zeros((5, 9));
        let err = build_stamp(
            cutout.view(),
            7,
            None,
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StampError::StampSizeTooLarge {
                stamp_size: 7,
                rows: 5,
                cols: 9,
            }
        );
    }

    #[test]
    fn error_precondition_error_shape_mismatch() {
        let cutout = Array2::<f64>::zeros((9, 9));
        let error = Array2::<f64>::zeros((9, 8));
        let err = build_stamp(
            cutout.view(),
            5,
            Some(error.view()),
            None,
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StampError::ErrorShapeMismatch {
                error_shape: (9, 8),
                cutout_shape: (9, 9),
            }
        );
    }

    #[test]
    fn error_precondition_mask_shape_mismatch() {
        let cutout = Array2::<f64>::zeros((9, 9));
        let mask = Array2::<bool>::from_elem((8, 9), false);
        let err = build_stamp(
            cutout.view(),
            5,
            None,
            Some(mask.view()),
            DEFAULT_WEIGHT_FWHM,
            DEFAULT_CENTROID_MAX_ITER,
            DEFAULT_CENTROID_TOL,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StampError::MaskShapeMismatch {
                mask_shape: (8, 9),
                cutout_shape: (9, 9),
            }
        );
    }
}

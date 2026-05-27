//! Region-growing driver entry point.
//!
//! The greedy heap loop, [`Connectivity`]-driven neighbour expansion,
//! cutout-edge handling, segmentation-label gating, and both stop
//! criteria (SNR and radial-gradient, each with independent
//! hysteresis) are in place. Growth runs until any enabled stop fires,
//! the mask touches the cutout edge, or the heap empties (the latter
//! two are reported as [`StopReason::Filled`]).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array2, ArrayView2};

use super::annulus::extract_annuli;
use super::config::{Connectivity, GrowthConfig, LabelInput};
use super::result::{GrowError, GrowthResult, StopReason};
use super::stop::StopState;

/// Internal heap entry. The newtype exists solely to let
/// [`BinaryHeap`] accept `f64`: the standard library requires
/// `T: Ord` at the type level, and `f64` only implements `PartialOrd`.
/// `total_cmp` gives a deterministic total order; non-finite fluxes are
/// filtered at push time so we never see NaN here.
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    flux: f64,
    row: usize,
    col: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.flux.total_cmp(&other.flux)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.flux == other.flux
    }
}

impl Eq for HeapItem {}

/// Push every in-bounds, unmasked, finite-flux, label-allowed neighbour
/// of `(row, col)` onto `heap`. Duplicates (same coordinate pushed
/// multiple times via different parents) are not de-duplicated here —
/// they are skipped at pop time by the `mask[(row, col)]` check.
fn push_unvisited_neighbors(
    row: usize,
    col: usize,
    data: ArrayView2<f64>,
    mask: &Array2<bool>,
    label: Option<&LabelInput>,
    connectivity: Connectivity,
    heap: &mut BinaryHeap<HeapItem>,
) {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    for &(d_row, d_col) in connectivity.offsets() {
        let next_row_signed = row as isize + d_row;
        let next_col_signed = col as isize + d_col;
        if next_row_signed < 0 || next_col_signed < 0 {
            continue;
        }
        let next_row = next_row_signed as usize;
        let next_col = next_col_signed as usize;
        if next_row >= rows || next_col >= cols {
            continue;
        }
        if mask[(next_row, next_col)] {
            continue;
        }
        if let Some(label) = label {
            if !label.allowed.contains(&label.map[(next_row, next_col)]) {
                continue;
            }
        }
        let flux = data[(next_row, next_col)];
        if !flux.is_finite() {
            continue;
        }
        heap.push(HeapItem {
            flux,
            row: next_row,
            col: next_col,
        });
    }
}

/// Grow a boolean source mask outward from one or more seed pixels.
///
/// See the [`region_growing` module documentation](super) for the
/// algorithm overview, the meaning of the stop criteria, and the
/// `StopReason::Filled` failure semantics.
///
/// # Parameters
///
/// - `data`: science image (one band). Pixel values are the heap key.
/// - `err`: optional 1-sigma error image; required if and only if the
///   SNR stop criterion is enabled. Shape must equal `data`.
/// - `label`: optional segmentation constraint (see [`LabelInput`]).
///   Shape must equal `data`.
/// - `seed_pixels`: one or more `(row, col)` starting pixels. All must
///   lie inside `data`; with a `label`, all must sit on an allowed
///   label.
/// - `config`: algorithm configuration. See [`GrowthConfig`].
///
/// # Errors
///
/// Returns [`GrowError`] for hard input-validation failures. Algorithmic
/// outcomes (mask reached the edge, heap exhausted before any stop
/// fired) are reported via [`StopReason::Filled`] in `Ok`, not as
/// errors.
pub fn grow_mask(
    data: ArrayView2<f64>,
    err: Option<ArrayView2<f64>>,
    label: Option<LabelInput>,
    seed_pixels: &[(usize, usize)],
    config: &GrowthConfig,
) -> Result<GrowthResult, GrowError> {
    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let shape = (rows, cols);

    // --- Config invariants (independent of data shapes). ---
    if config.check_interval == 0 {
        return Err(GrowError::CheckIntervalZero);
    }
    if config.stop.snr.is_none() && config.stop.gradient.is_none() {
        return Err(GrowError::NoStopCriterion);
    }
    // The err / SnrStop binding is bidirectional: enabling one without
    // the other is a caller bug, not a degraded mode.
    match (err.as_ref(), config.stop.snr) {
        (Some(_), None) => return Err(GrowError::ErrWithoutSnrStop),
        (None, Some(_)) => return Err(GrowError::SnrStopWithoutErr),
        _ => {}
    }

    // --- Shape invariants. ---
    if let Some(err_view) = err.as_ref() {
        let err_shape = (err_view.shape()[0], err_view.shape()[1]);
        if err_shape != shape {
            return Err(GrowError::ErrShapeMismatch {
                err_shape,
                data_shape: shape,
            });
        }
    }

    // Validate label shape and allowed-nonempty before any seed check,
    // because the seed-on-allowed check below indexes into `label.map`.
    if let Some(label) = label.as_ref() {
        let label_shape = (label.map.shape()[0], label.map.shape()[1]);
        if label_shape != shape {
            return Err(GrowError::LabelShapeMismatch {
                label_shape,
                data_shape: shape,
            });
        }
        if label.allowed.is_empty() {
            return Err(GrowError::LabelAllowedEmpty);
        }
    }

    for &seed in seed_pixels {
        if seed.0 >= rows || seed.1 >= cols {
            return Err(GrowError::SeedOutOfBounds { seed, shape });
        }
        if let Some(label) = label.as_ref() {
            let label_at_seed = label.map[(seed.0, seed.1)];
            if !label.allowed.contains(&label_at_seed) {
                return Err(GrowError::SeedOnDisallowedLabel {
                    seed,
                    label: label_at_seed,
                });
            }
        }
    }

    let mut mask = Array2::<bool>::from_elem(shape, false);
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut touches_edge = false;

    let on_edge = |row: usize, col: usize| -> bool {
        row == 0 || row + 1 == rows || col == 0 || col + 1 == cols
    };

    for &(row, col) in seed_pixels {
        if mask[(row, col)] {
            // Duplicate seed coordinate: silently collapse.
            continue;
        }
        mask[(row, col)] = true;
        if on_edge(row, col) {
            touches_edge = true;
        }
        push_unvisited_neighbors(
            row,
            col,
            data,
            &mask,
            label.as_ref(),
            config.connectivity,
            &mut heap,
        );
    }

    let mut n_iterations: usize = 0;
    let mut stop_state = StopState::new();

    loop {
        if touches_edge {
            return Ok(GrowthResult {
                mask,
                stop_reason: StopReason::Filled,
                n_iterations,
            });
        }
        let Some(item) = heap.pop() else {
            return Ok(GrowthResult {
                mask,
                stop_reason: StopReason::Filled,
                n_iterations,
            });
        };
        let (row, col) = (item.row, item.col);
        if mask[(row, col)] {
            // Lazy dedup: the same coordinate may have been pushed
            // multiple times via different parents.
            continue;
        }
        mask[(row, col)] = true;
        n_iterations += 1;
        if on_edge(row, col) {
            touches_edge = true;
        }
        push_unvisited_neighbors(
            row,
            col,
            data,
            &mask,
            label.as_ref(),
            config.connectivity,
            &mut heap,
        );

        if n_iterations >= config.min_pixels_before_stop_check
            && n_iterations.is_multiple_of(config.check_interval)
        {
            let annuli = extract_annuli(
                mask.view(),
                label.as_ref(),
                config.connectivity,
                config.annulus_thickness,
            );
            if let Some(reason) = stop_state.evaluate(&annuli, data, err, &config.stop) {
                return Ok(GrowthResult {
                    mask,
                    stop_reason: reason,
                    n_iterations,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aperture::region_growing::config::{
        Connectivity, GradientStop, SnrStop, StopCriterion,
    };
    use ndarray::Array2;

    /// Default test config with a never-firing SNR stop: the threshold
    /// is so low that SNR > threshold always, and the hysteresis is so
    /// large that no realistic test could accumulate enough violations.
    /// Used by tests that want to exercise non-stop behaviour (edge
    /// fill, seed errors, label gating) without forcing each one to
    /// hand-build a config that satisfies [`GrowError::NoStopCriterion`].
    fn trivial_config() -> GrowthConfig {
        GrowthConfig {
            connectivity: Connectivity::Eight,
            stop: StopCriterion {
                snr: Some(SnrStop {
                    threshold: 0.5,
                    hysteresis: usize::MAX,
                }),
                gradient: None,
            },
            min_pixels_before_stop_check: 0,
            check_interval: 1,
            annulus_thickness: 1,
        }
    }

    /// Constant-1 err matching `shape`, satisfying the SnrStop-requires-err
    /// invariant for tests that use [`trivial_config`].
    fn ones_err(shape: (usize, usize)) -> Array2<f64> {
        Array2::<f64>::from_elem(shape, 1.0)
    }

    #[test]
    fn flat_field_grows_until_edge_touch() {
        let data = Array2::<f64>::from_elem((5, 5), 1.0);
        let err = ones_err((5, 5));
        let seeds = [(2, 2)];
        let result = grow_mask(
            data.view(),
            Some(err.view()),
            None,
            &seeds,
            &trivial_config(),
        )
        .expect("flat-field growth must succeed");

        assert_eq!(result.stop_reason, StopReason::Filled);
        assert!(result.n_iterations >= 1);
        assert!(result.mask[(2, 2)], "seed must be preserved");

        // Structural invariant: every `true` in the mask is either a
        // seed (1 here) or was admitted by the heap loop.
        let true_count = result.mask.iter().filter(|&&v| v).count();
        assert_eq!(true_count, 1 + result.n_iterations);

        // Filled must mean the mask reached the cutout edge.
        let touched_edge = (0..5).any(|i| {
            result.mask[(0, i)] || result.mask[(4, i)] || result.mask[(i, 0)] || result.mask[(i, 4)]
        });
        assert!(touched_edge, "Filled requires the mask to have hit an edge");
    }

    #[test]
    fn seed_out_of_bounds_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let err_array = ones_err((3, 3));
        let seeds = [(3, 0)];
        let err = grow_mask(
            data.view(),
            Some(err_array.view()),
            None,
            &seeds,
            &trivial_config(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            GrowError::SeedOutOfBounds {
                seed: (3, 0),
                shape: (3, 3),
            }
        );
    }

    /// Two well-separated bright blobs, each carrying its own
    /// segmentation label, with background label 0 everywhere else. The
    /// caller whitelists only `{0, 1}` (background + blob A). The mask
    /// must never enter any pixel carrying label 2.
    #[test]
    fn label_gate_prevents_growth_into_disallowed_region() {
        let rows = 7;
        let cols = 7;

        // Image: bright at the two blob centres, weak elsewhere.
        let mut data = Array2::<f64>::from_elem((rows, cols), 0.1);
        let blob_a = (1, 2);
        let blob_b = (5, 5);
        for &(blob_row, blob_col) in &[blob_a, blob_b] {
            for d_row in -1..=1_isize {
                for d_col in -1..=1_isize {
                    let row = (blob_row as isize + d_row) as usize;
                    let col = (blob_col as isize + d_col) as usize;
                    data[(row, col)] = 10.0;
                }
            }
        }

        // Label map: blob A pixels = 1, blob B pixels = 2, rest = 0.
        let mut label_map = Array2::<i32>::zeros((rows, cols));
        for d_row in -1..=1_isize {
            for d_col in -1..=1_isize {
                label_map[(
                    (blob_a.0 as isize + d_row) as usize,
                    (blob_a.1 as isize + d_col) as usize,
                )] = 1;
                label_map[(
                    (blob_b.0 as isize + d_row) as usize,
                    (blob_b.1 as isize + d_col) as usize,
                )] = 2;
            }
        }

        let label = LabelInput {
            map: label_map.view(),
            allowed: vec![0, 1],
        };
        let err = ones_err((rows, cols));
        let seeds = [blob_a];
        let result = grow_mask(
            data.view(),
            Some(err.view()),
            Some(label),
            &seeds,
            &trivial_config(),
        )
        .expect("label-gated growth must succeed");

        // Core invariant: no pixel with label 2 may be admitted.
        for row in 0..rows {
            for col in 0..cols {
                if result.mask[(row, col)] {
                    assert_ne!(
                        label_map[(row, col)],
                        2,
                        "mask leaked into disallowed label at ({row}, {col})",
                    );
                }
            }
        }
        // Sanity: the seed is in the mask and the loop actually ran.
        assert!(result.mask[blob_a]);
        assert!(result.n_iterations >= 1);
    }

    #[test]
    fn seed_on_disallowed_label_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let err_array = ones_err((3, 3));
        let label_map = Array2::<i32>::zeros((3, 3));
        let label = LabelInput {
            map: label_map.view(),
            allowed: vec![1],
        };
        let seeds = [(1, 1)];
        let err = grow_mask(
            data.view(),
            Some(err_array.view()),
            Some(label),
            &seeds,
            &trivial_config(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            GrowError::SeedOnDisallowedLabel {
                seed: (1, 1),
                label: 0,
            }
        );
    }

    /// Centered Gaussian source with constant per-pixel err; the SNR
    /// stop must fire before the mask reaches the cutout edge.
    #[test]
    fn snr_stop_fires_on_gaussian_with_per_pixel_err() {
        let n = 21;
        let center = 10;
        let sigma = 2.0_f64;
        let amplitude = 100.0_f64;

        let mut data = Array2::<f64>::zeros((n, n));
        for row in 0..n {
            for col in 0..n {
                let d_row = row as f64 - center as f64;
                let d_col = col as f64 - center as f64;
                data[(row, col)] =
                    amplitude * (-(d_row * d_row + d_col * d_col) / (2.0 * sigma * sigma)).exp();
            }
        }
        let err = ones_err((n, n));

        let config = GrowthConfig {
            connectivity: Connectivity::Eight,
            stop: StopCriterion {
                snr: Some(SnrStop {
                    threshold: 2.0,
                    hysteresis: 3,
                }),
                gradient: None,
            },
            min_pixels_before_stop_check: 5,
            check_interval: 1,
            annulus_thickness: 1,
        };

        let result = grow_mask(
            data.view(),
            Some(err.view()),
            None,
            &[(center, center)],
            &config,
        )
        .expect("growth must succeed");

        assert_eq!(result.stop_reason, StopReason::SnrBelow);
        assert!(result.mask[(center, center)], "seed must be in mask");

        // If SnrBelow really fired, the mask must not have leaked all
        // the way to the cutout edge (otherwise Filled would have won).
        let touched_edge = (0..n).any(|i| {
            result.mask[(0, i)]
                || result.mask[(n - 1, i)]
                || result.mask[(i, 0)]
                || result.mask[(i, n - 1)]
        });
        assert!(
            !touched_edge,
            "SnrBelow must fire before the mask reaches the edge"
        );
    }

    /// Two equal-amplitude Gaussian blobs on the same row, separated by
    /// a low-flux basin. With only the gradient stop enabled, the mask
    /// seeded at blob A must terminate before its inner annulus climbs
    /// the slope of blob B.
    #[test]
    fn gradient_stop_prevents_crossing_into_neighbour_blob() {
        // Generous cutout with the blob pair centred along its
        // mid-row: the mask has plenty of margin in every direction
        // other than the one toward blob B, so the gradient flip is
        // the only realistic exit besides reaching blob B itself.
        let rows = 31;
        let cols = 31;
        let sigma = 2.0_f64;
        let amplitude = 100.0_f64;
        let blob_a = (15, 11);
        let blob_b = (15, 21);

        let mut data = Array2::<f64>::zeros((rows, cols));
        for &(blob_row, blob_col) in &[blob_a, blob_b] {
            for row in 0..rows {
                for col in 0..cols {
                    let d_row = row as f64 - blob_row as f64;
                    let d_col = col as f64 - blob_col as f64;
                    data[(row, col)] += amplitude
                        * (-(d_row * d_row + d_col * d_col) / (2.0 * sigma * sigma)).exp();
                }
            }
        }

        let config = GrowthConfig {
            connectivity: Connectivity::Eight,
            stop: StopCriterion {
                snr: None,
                gradient: Some(GradientStop {
                    ratio_threshold: 1.0,
                    hysteresis: 2,
                }),
            },
            min_pixels_before_stop_check: 5,
            check_interval: 1,
            annulus_thickness: 2,
        };

        let result =
            grow_mask(data.view(), None, None, &[blob_a], &config).expect("growth must succeed");

        assert_eq!(result.stop_reason, StopReason::GradientFlip);
        assert!(result.mask[blob_a], "seed (blob A centre) must be in mask");
        assert!(
            !result.mask[blob_b],
            "blob B centre must NOT be reached — gradient must stop the crossing",
        );

        let touched_edge = (0..rows)
            .any(|row| result.mask[(row, 0)] || result.mask[(row, cols - 1)])
            || (0..cols).any(|col| result.mask[(0, col)] || result.mask[(rows - 1, col)]);
        assert!(
            !touched_edge,
            "GradientFlip must fire before the mask reaches the edge",
        );
    }
}

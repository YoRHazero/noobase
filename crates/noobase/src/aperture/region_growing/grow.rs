//! Region-growing driver entry point.
//!
//! The greedy heap loop, [`Connectivity`]-driven neighbour expansion,
//! cutout-edge handling, and segmentation-label gating are in place;
//! annulus extraction and stop-criterion evaluation land in subsequent
//! commits. Until those arrive, growth continues until the mask touches
//! the cutout edge or the heap empties — both are reported as
//! [`StopReason::Filled`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array2, ArrayView2};

use super::config::{Connectivity, GrowthConfig, LabelInput};
use super::result::{GrowError, GrowthResult, StopReason};

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
    // Reserved for later commits (err <-> SnrStop binding).
    let _ = err;

    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let shape = (rows, cols);

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aperture::region_growing::config::{Connectivity, StopCriterion};
    use ndarray::Array2;

    fn trivial_config() -> GrowthConfig {
        GrowthConfig {
            connectivity: Connectivity::Eight,
            stop: StopCriterion::default(),
            min_pixels_before_stop_check: 0,
            check_interval: 1,
            annulus_thickness: 1,
        }
    }

    #[test]
    fn flat_field_grows_until_edge_touch() {
        let data = Array2::<f64>::from_elem((5, 5), 1.0);
        let seeds = [(2, 2)];
        let result = grow_mask(data.view(), None, None, &seeds, &trivial_config())
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
        let seeds = [(3, 0)];
        let err = grow_mask(data.view(), None, None, &seeds, &trivial_config()).unwrap_err();
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
        let seeds = [blob_a];
        let result = grow_mask(data.view(), None, Some(label), &seeds, &trivial_config())
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
        let label_map = Array2::<i32>::zeros((3, 3));
        let label = LabelInput {
            map: label_map.view(),
            allowed: vec![1],
        };
        let seeds = [(1, 1)];
        let err = grow_mask(data.view(), None, Some(label), &seeds, &trivial_config()).unwrap_err();
        assert_eq!(
            err,
            GrowError::SeedOnDisallowedLabel {
                seed: (1, 1),
                label: 0,
            }
        );
    }
}

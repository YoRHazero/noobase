//! Region-growing driver entry point.
//!
//! This file is currently a **skeleton stub**: it only validates that
//! every seed pixel is in bounds and returns a mask with exactly the
//! seed pixels set, [`StopReason::Filled`], and `n_iterations = 0`. The
//! heap loop, connectivity-driven expansion, label gating, annulus
//! extraction, and stop-criterion evaluation are added in subsequent
//! commits.

use ndarray::{Array2, ArrayView2};

use super::config::{GrowthConfig, LabelInput};
use super::result::{GrowError, GrowthResult, StopReason};

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
    // Inputs reserved for later commits (heap loop, label gating, stop
    // criteria). Silence unused-variable warnings without bending the
    // public signature.
    let _ = (err, label, config);

    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let shape = (rows, cols);

    for &seed in seed_pixels {
        if seed.0 >= rows || seed.1 >= cols {
            return Err(GrowError::SeedOutOfBounds { seed, shape });
        }
    }

    let mut mask = Array2::<bool>::from_elem(shape, false);
    for &(row, col) in seed_pixels {
        mask[(row, col)] = true;
    }

    Ok(GrowthResult {
        mask,
        stop_reason: StopReason::Filled,
        n_iterations: 0,
    })
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
    fn stub_returns_seed_only_mask_and_filled() {
        let data = Array2::<f64>::zeros((3, 3));
        let seeds = [(1, 1)];
        let result = grow_mask(data.view(), None, None, &seeds, &trivial_config())
            .expect("stub must succeed");

        assert_eq!(result.stop_reason, StopReason::Filled);
        assert_eq!(result.n_iterations, 0);
        for row in 0..3 {
            for col in 0..3 {
                let expected = (row, col) == (1, 1);
                assert_eq!(
                    result.mask[(row, col)],
                    expected,
                    "mask[{row},{col}] should be {expected}",
                );
            }
        }
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
}

//! Region-growing driver entry point.
//!
//! The greedy heap loop, [`Connectivity`]-driven neighbour expansion,
//! cutout-edge handling, segmentation-label gating, and both stop
//! criteria (SNR and radial-gradient, each with independent
//! hysteresis) are in place. Growth runs until any enabled stop fires,
//! the mask touches the cutout edge, or the heap empties (the latter
//! two are reported as [`StopReason::Filled`]).
//!
//! # Detection vs measurement
//!
//! Growth is driven by a separate `detection` image (the heap key),
//! while the stop criteria measure the `data` image (plus `err`). The
//! caller is expected to pass a smoothed / matched-filter detection
//! image so the heap sees a denoised flux landscape and does not chase
//! single-pixel noise ridges; the raw `data` keeps the stop statistics
//! and downstream photometry honest. Passing the same array for both
//! recovers brightest-pixel growth on the raw image.
//!
//! # Shape regularisation
//!
//! The heap priority is `detection_value + shape_weight * fraction`,
//! where `fraction` is the candidate's in-mask neighbour count over
//! [`Connectivity::max_neighbors`]. This biases the growth *order*
//! toward filling concavities before extending arms. On top of it a
//! hard `min_neighbor_support` floor (activated after a warm-up) refuses
//! pixels with too few in-mask neighbours, forbidding one-pixel-wide
//! tendrils outright. A gated pixel is simply dropped from the heap; it
//! re-enters naturally the next time one of its neighbours is admitted,
//! by which point its support has risen — so deferral needs no extra
//! bookkeeping and the loop still terminates in `O(pixels * degree)`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array2, ArrayView2};

use super::annulus::extract_annuli;
use super::config::{Connectivity, GrowthConfig, LabelInput};
use super::result::{GrowError, GrowthResult, StopReason};
use super::stop::StopState;

/// Internal heap entry. The newtype exists partly to let
/// [`BinaryHeap`] accept an `f64` key (the standard library requires
/// `T: Ord`, and `f64` is only `PartialOrd`), and partly to attach the
/// pixel coordinates.
///
/// `priority` is `detection_value + shape_weight * support_fraction`,
/// computed at push time; non-finite detection values are filtered
/// before construction so we never see NaN here. The `(row, col)`
/// tiebreak makes the order *total* (no two distinct pixels compare
/// equal), so heap pop order is fully deterministic even when two
/// candidates share a priority. `total_cmp` is used so the `Eq` impl
/// stays consistent with `Ord`.
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    priority: f64,
    row: usize,
    col: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then(self.row.cmp(&other.row))
            .then(self.col.cmp(&other.col))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapItem {}

/// Count the in-mask neighbours of `(row, col)` under `connectivity`.
///
/// This is the support used both for the soft shape term (via its
/// `[0, 1]` fraction) at push time and for the hard `min_neighbor_support`
/// floor at pop time. It walks the same offset table the heap and the
/// annuli use, so "support" is measured in the topology the mask grows
/// in.
fn count_mask_neighbors(
    row: usize,
    col: usize,
    mask: &Array2<bool>,
    connectivity: Connectivity,
) -> usize {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    let mut support = 0;
    for &(d_row, d_col) in connectivity.offsets() {
        let neighbor_row_signed = row as isize + d_row;
        let neighbor_col_signed = col as isize + d_col;
        if neighbor_row_signed < 0 || neighbor_col_signed < 0 {
            continue;
        }
        let neighbor_row = neighbor_row_signed as usize;
        let neighbor_col = neighbor_col_signed as usize;
        if neighbor_row >= rows || neighbor_col >= cols {
            continue;
        }
        if mask[(neighbor_row, neighbor_col)] {
            support += 1;
        }
    }
    support
}

/// Count the in-mask *cardinal* (edge-sharing) neighbours of `(row, col)`.
///
/// This is the support the concavity fill keys on: a pixel boxed in on
/// three or more of its four cardinal sides sits in a notch (or hole)
/// whose closure shrinks the 4-connected perimeter. Diagonal neighbours
/// touch only at a corner and do not change edge exposure, so the count
/// — and hence the fill — is identical under either [`Connectivity`].
fn count_cardinal_mask_neighbors(row: usize, col: usize, mask: &Array2<bool>) -> usize {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    let mut support = 0;
    for &(d_row, d_col) in Connectivity::Four.offsets() {
        let neighbor_row = row as isize + d_row;
        let neighbor_col = col as isize + d_col;
        if neighbor_row < 0 || neighbor_col < 0 {
            continue;
        }
        let neighbor_row = neighbor_row as usize;
        let neighbor_col = neighbor_col as usize;
        if neighbor_row >= rows || neighbor_col >= cols {
            continue;
        }
        if mask[(neighbor_row, neighbor_col)] {
            support += 1;
        }
    }
    support
}

/// Push the in-bounds, not-yet-masked cardinal neighbours of `(row, col)`
/// onto `stack` as fill candidates.
fn push_cardinal_unvisited(
    row: usize,
    col: usize,
    mask: &Array2<bool>,
    stack: &mut Vec<(usize, usize)>,
) {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    for &(d_row, d_col) in Connectivity::Four.offsets() {
        let next_row = row as isize + d_row;
        let next_col = col as isize + d_col;
        if next_row < 0 || next_col < 0 {
            continue;
        }
        let next_row = next_row as usize;
        let next_col = next_col as usize;
        if next_row >= rows || next_col >= cols {
            continue;
        }
        if !mask[(next_row, next_col)] {
            stack.push((next_row, next_col));
        }
    }
}

/// Push every in-bounds, unmasked, finite-detection, label-allowed
/// neighbour of `(row, col)` onto `heap`, keyed by the shape-regularised
/// priority `detection_value + shape_weight * support_fraction`.
///
/// The support fraction is recomputed fresh on every push, so each time
/// a neighbour of some pixel is admitted that pixel is re-pushed with an
/// up-to-date (higher) priority. The stale lower-priority copies left in
/// the heap are harmless: the highest-priority copy is popped first and
/// admitted, and the rest are skipped at pop time by the
/// `mask[(row, col)]` check. This is also how a pixel deferred by the
/// `min_neighbor_support` floor re-enters once its support has risen.
fn push_unvisited_neighbors(
    row: usize,
    col: usize,
    detection: ArrayView2<f64>,
    mask: &Array2<bool>,
    label: Option<&LabelInput>,
    config: &GrowthConfig,
    heap: &mut BinaryHeap<HeapItem>,
) {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    let connectivity = config.connectivity;
    let max_neighbors = connectivity.max_neighbors() as f64;
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
        if let Some(label) = label
            && !label.allowed.contains(&label.map[(next_row, next_col)])
        {
            continue;
        }
        let detection_value = detection[(next_row, next_col)];
        if !detection_value.is_finite() {
            continue;
        }
        let support = count_mask_neighbors(next_row, next_col, mask, connectivity);
        let support_fraction = support as f64 / max_neighbors;
        let priority = detection_value + config.shape_weight * support_fraction;
        heap.push(HeapItem {
            priority,
            row: next_row,
            col: next_col,
        });
    }
}

/// Unconditionally close concavities reachable from the just-admitted
/// pixels: any candidate with at least `fill_min_cardinal_support` in-mask
/// cardinal neighbours is admitted immediately, ignoring its flux (and
/// finiteness), then its own cardinal neighbours are re-examined so a
/// notch or one-pixel slot zips shut in a single cascade.
///
/// Disabled (`fill_min_cardinal_support == None`) it is a no-op. Filled
/// pixels increment `n_iterations` (preserving the
/// `mask.count() == seeds + n_iterations` invariant), feed the heap their
/// neighbours for continued flux growth, and flip `touches_edge` if they
/// land on the cutout edge. The label whitelist is honoured — source
/// separation stays with segmentation, never this fill.
#[allow(clippy::too_many_arguments)]
fn fill_enclosed_cascade(
    started_from: &[(usize, usize)],
    detection: ArrayView2<f64>,
    mask: &mut Array2<bool>,
    label: Option<&LabelInput>,
    config: &GrowthConfig,
    heap: &mut BinaryHeap<HeapItem>,
    n_iterations: &mut usize,
    touches_edge: &mut bool,
) {
    let Some(threshold) = config.fill_min_cardinal_support else {
        return;
    };
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];

    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &(row, col) in started_from {
        push_cardinal_unvisited(row, col, mask, &mut stack);
    }

    while let Some((row, col)) = stack.pop() {
        if mask[(row, col)] {
            continue;
        }
        if let Some(label) = label
            && !label.allowed.contains(&label.map[(row, col)])
        {
            continue;
        }
        if count_cardinal_mask_neighbors(row, col, mask) < threshold {
            continue;
        }
        mask[(row, col)] = true;
        *n_iterations += 1;
        if row == 0 || row + 1 == rows || col == 0 || col + 1 == cols {
            *touches_edge = true;
        }
        push_unvisited_neighbors(row, col, detection, mask, label, config, heap);
        push_cardinal_unvisited(row, col, mask, &mut stack);
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
/// - `detection`: image driving the heap priority (one band). The
///   caller is expected to pass a smoothed / matched-filter image here;
///   pass the same array as `data` for plain brightest-pixel growth.
///   Shape must equal `data`. Non-finite pixels are never admitted.
/// - `data`: science image used for the stop statistics (SNR and
///   radial gradient) and returned to the caller for photometry.
/// - `err`: optional 1-sigma error image; required if and only if the
///   SNR stop criterion is enabled. Shape must equal `data`.
/// - `label`: optional segmentation constraint (see [`LabelInput`]).
///   Shape must equal `data`.
/// - `seed_pixels`: one or more `(row, col)` starting pixels. All must
///   lie inside `data`; with a `label`, all must sit on an allowed
///   label. Seeds are admitted unconditionally — the shape floor never
///   applies to them.
/// - `config`: algorithm configuration. See [`GrowthConfig`], including
///   the shape-regularisation fields.
///
/// # Errors
///
/// Returns [`GrowError`] for hard input-validation failures. Algorithmic
/// outcomes (mask reached the edge, heap exhausted before any stop
/// fired) are reported via [`StopReason::Filled`] in `Ok`, not as
/// errors.
pub fn grow_mask(
    detection: ArrayView2<f64>,
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
    let max_neighbors = config.connectivity.max_neighbors();
    if config.min_neighbor_support > max_neighbors {
        return Err(GrowError::MinNeighborSupportTooLarge {
            min_neighbor_support: config.min_neighbor_support,
            max_neighbors,
        });
    }
    if let Some(gradient) = config.stop.gradient {
        let (lo, hi) = (gradient.lo_percentile, gradient.hi_percentile);
        if !(lo >= 0.0 && lo < hi && hi <= 100.0) {
            return Err(GrowError::GradientPercentileInvalid { lo, hi });
        }
    }
    if let Some(threshold) = config.fill_min_cardinal_support
        && !(3..=4).contains(&threshold)
    {
        return Err(GrowError::FillSupportInvalid { value: threshold });
    }
    // The err / SnrStop binding is bidirectional: enabling one without
    // the other is a caller bug, not a degraded mode.
    match (err.as_ref(), config.stop.snr) {
        (Some(_), None) => return Err(GrowError::ErrWithoutSnrStop),
        (None, Some(_)) => return Err(GrowError::SnrStopWithoutErr),
        _ => {}
    }

    // --- Shape invariants. ---
    let detection_shape = (detection.shape()[0], detection.shape()[1]);
    if detection_shape != shape {
        return Err(GrowError::DetectionShapeMismatch {
            detection_shape,
            data_shape: shape,
        });
    }
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
            detection,
            &mask,
            label.as_ref(),
            config,
            &mut heap,
        );
    }

    let mut n_iterations: usize = 0;
    let mut stop_state = StopState::new();

    // Close any concavity the seed set itself already encloses before the
    // flux-driven loop starts (a no-op for a single seed, or when the
    // fill is disabled).
    fill_enclosed_cascade(
        seed_pixels,
        detection,
        &mut mask,
        label.as_ref(),
        config,
        &mut heap,
        &mut n_iterations,
        &mut touches_edge,
    );

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
            // multiple times via different parents (or re-pushed as its
            // support rose). All but the first pop are skipped here.
            continue;
        }
        // Hard shape floor: once the seed core is established, refuse a
        // pixel that has too few in-mask neighbours. Dropping it (rather
        // than re-pushing) is correct because it will be re-pushed by
        // `push_unvisited_neighbors` the next time one of its neighbours
        // is admitted, by which point its support has grown. Seeds and
        // the warm-up window bypass this; `min_neighbor_support <= 1`
        // disables it (every frontier pixel has >= 1 in-mask neighbour).
        if config.min_neighbor_support > 1
            && n_iterations >= config.min_pixels_before_shape_gate
            && count_mask_neighbors(row, col, &mask, config.connectivity)
                < config.min_neighbor_support
        {
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
            detection,
            &mask,
            label.as_ref(),
            config,
            &mut heap,
        );
        // Real-time concavity closing: the pixel just admitted may have
        // boxed in a neighbouring notch, which this zips shut at once.
        fill_enclosed_cascade(
            &[(row, col)],
            detection,
            &mut mask,
            label.as_ref(),
            config,
            &mut heap,
            &mut n_iterations,
            &mut touches_edge,
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
            // Shape regularisation disabled: `shape_weight == 0` and
            // `min_neighbor_support <= 1` reproduce the pure
            // brightest-pixel algorithm, so these tests pin behaviour
            // independent of the shape machinery (which has its own
            // dedicated tests below).
            shape_weight: 0.0,
            min_neighbor_support: 1,
            min_pixels_before_shape_gate: 0,
            fill_min_cardinal_support: None,
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
            shape_weight: 0.0,
            min_neighbor_support: 1,
            min_pixels_before_shape_gate: 0,
            fill_min_cardinal_support: None,
            min_pixels_before_stop_check: 5,
            check_interval: 1,
            annulus_thickness: 1,
        };

        let result = grow_mask(
            data.view(),
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
                    lo_percentile: 75.0,
                    hi_percentile: 99.0,
                }),
            },
            shape_weight: 0.0,
            min_neighbor_support: 1,
            min_pixels_before_shape_gate: 0,
            fill_min_cardinal_support: None,
            min_pixels_before_stop_check: 5,
            check_interval: 1,
            annulus_thickness: 2,
        };

        let result = grow_mask(data.view(), data.view(), None, None, &[blob_a], &config)
            .expect("growth must succeed");

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

    /// The heap must follow the `detection` image, not the `data` image.
    /// Detection carries a bright ridge running right from the seed;
    /// data carries an equally bright ridge running down. With pure
    /// brightest-first settings the mask must chase detection's ridge to
    /// the right edge and ignore data's downward ridge entirely.
    #[test]
    fn detection_drives_growth_not_data() {
        let n = 11;
        let seed = (5, 5);

        let mut detection = Array2::<f64>::from_elem((n, n), 1.0);
        for col in seed.1..n {
            detection[(seed.0, col)] = 100.0; // bright ridge to the right
        }
        let mut data = Array2::<f64>::from_elem((n, n), 1.0);
        for row in seed.0..n {
            data[(row, seed.1)] = 100.0; // bright ridge going down
        }
        let err = ones_err((n, n));

        let result = grow_mask(
            detection.view(),
            data.view(),
            Some(err.view()),
            None,
            &[seed],
            &trivial_config(),
        )
        .expect("growth must succeed");

        // Reached the end of detection's rightward ridge ...
        assert!(
            result.mask[(5, 10)],
            "growth must follow the detection ridge to the right edge"
        );
        // ... and never followed data's downward ridge.
        assert!(
            !result.mask[(10, 5)],
            "growth must not follow the data ridge — data drives stops, not the heap"
        );
    }

    /// The soft `shape_weight` term must reorder growth toward compact
    /// shapes. A bright one-pixel ridge runs from the central seed to
    /// the right edge over a dim background. With `shape_weight == 0` the
    /// heap races straight down the ridge and terminates the instant it
    /// touches the edge, yielding a thin mask. With a large
    /// `shape_weight` the high-support background pixels flanking the
    /// ridge outrank the ridge tip, so growth fattens out instead.
    #[test]
    fn shape_weight_prefers_compact_growth_over_bright_tendril() {
        let n = 11;
        let seed = (5, 5);
        // First flank pixel compact growth reaches: it sits off the
        // ridge (row 5) and, sharing the top priority among the support-2
        // neighbours, is admitted before any edge touch can freeze the
        // mask — so the assertion does not hinge on the pop tiebreak.
        let off_ridge = (6, 6);

        let mut data = Array2::<f64>::from_elem((n, n), 1.0);
        for col in seed.1..n {
            data[(seed.0, col)] = 100.0;
        }
        let err = ones_err((n, n));

        let mut config = trivial_config();
        config.shape_weight = 0.0;
        let thin = grow_mask(
            data.view(),
            data.view(),
            Some(err.view()),
            None,
            &[seed],
            &config,
        )
        .expect("growth must succeed");

        config.shape_weight = 1000.0;
        let fat = grow_mask(
            data.view(),
            data.view(),
            Some(err.view()),
            None,
            &[seed],
            &config,
        )
        .expect("growth must succeed");

        // Brightest-first: a thin ridge straight to the right edge.
        assert!(thin.mask[(5, 10)], "ridge growth must reach the right edge");
        assert!(
            !thin.mask[off_ridge],
            "without shape_weight, growth must not leave the ridge"
        );

        // Shape-regularised: growth leaves the ridge to fill compactly,
        // so it admits the flanking pixel and runs longer before any
        // edge touch ends it.
        assert!(
            fat.mask[off_ridge],
            "shape_weight must pull growth off the ridge into the flank"
        );
        assert!(
            fat.n_iterations > thin.n_iterations,
            "compact growth must admit more pixels before terminating \
             ({} vs {})",
            fat.n_iterations,
            thin.n_iterations,
        );
    }

    /// The hard `min_neighbor_support` floor must forbid extending a
    /// one-pixel-wide tendril. A bright 3x3 core has a two-pixel arm
    /// sticking out; everything else is NaN so the only growth direction
    /// is along the arm. The arm's first pixel touches the core on three
    /// sides (support 3) and is always admitted; its second pixel
    /// touches only the first (support 1). The floor must reject that
    /// second pixel while a disabled floor admits it.
    #[test]
    fn min_neighbor_support_floor_blocks_one_pixel_tendril() {
        let n = 7;
        let arm_root = (3, 5);
        let arm_tip = (3, 6);

        // NaN everywhere except a bright 3x3 core and the two-pixel arm.
        let mut image = Array2::<f64>::from_elem((n, n), f64::NAN);
        for row in 2..=4 {
            for col in 2..=4 {
                image[(row, col)] = 10.0;
            }
        }
        image[arm_root] = 9.0;
        image[arm_tip] = 9.0;
        let err = ones_err((n, n));

        // Floor disabled: the tendril tip is admitted.
        let mut config = trivial_config();
        config.min_neighbor_support = 1;
        let unguarded = grow_mask(
            image.view(),
            image.view(),
            Some(err.view()),
            None,
            &[(3, 3)],
            &config,
        )
        .expect("growth must succeed");
        assert!(
            unguarded.mask[arm_tip],
            "with the floor disabled the tendril tip must be admitted"
        );

        // Floor enabled (after the core fills): the tip is rejected, the
        // arm root is still admitted because it has support 3.
        config.min_neighbor_support = 2;
        config.min_pixels_before_shape_gate = 8;
        let guarded = grow_mask(
            image.view(),
            image.view(),
            Some(err.view()),
            None,
            &[(3, 3)],
            &config,
        )
        .expect("growth must succeed");
        assert!(
            guarded.mask[arm_root],
            "the arm root has support 3 and must still be admitted"
        );
        assert!(
            !guarded.mask[arm_tip],
            "the min_neighbor_support floor must reject the one-pixel tendril tip"
        );
    }

    #[test]
    fn detection_shape_mismatch_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let detection = Array2::<f64>::zeros((2, 2));
        let err_array = ones_err((3, 3));
        let err = grow_mask(
            detection.view(),
            data.view(),
            Some(err_array.view()),
            None,
            &[(1, 1)],
            &trivial_config(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            GrowError::DetectionShapeMismatch {
                detection_shape: (2, 2),
                data_shape: (3, 3),
            }
        );
    }

    /// A 3x3 bright block with a dead (NaN) centre, walled off by NaN so
    /// the flux-driven heap can never admit the centre. The cardinal fill
    /// must close it once its four sides are in the mask, regardless of
    /// its (non-finite) value; with the fill disabled it stays a hole.
    #[test]
    fn fill_closes_enclosed_pixel_regardless_of_flux() {
        let mut data = Array2::<f64>::from_elem((5, 5), f64::NAN);
        for row in 1..=3 {
            for col in 1..=3 {
                data[(row, col)] = 10.0;
            }
        }
        data[(2, 2)] = f64::NAN; // dead centre the heap can never reach
        let err = ones_err((5, 5));
        let seeds = [(1, 1)];

        let mut config = trivial_config();
        config.fill_min_cardinal_support = None;
        let unfilled = grow_mask(
            data.view(),
            data.view(),
            Some(err.view()),
            None,
            &seeds,
            &config,
        )
        .expect("growth must succeed");
        assert!(
            !unfilled.mask[(2, 2)],
            "without the fill the dead centre must stay a hole"
        );

        config.fill_min_cardinal_support = Some(3);
        let filled = grow_mask(
            data.view(),
            data.view(),
            Some(err.view()),
            None,
            &seeds,
            &config,
        )
        .expect("growth must succeed");
        assert!(
            filled.mask[(2, 2)],
            "the cardinal fill must close the enclosed dead centre"
        );

        // The eight bright sides are admitted either way.
        for &(row, col) in &[
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        ] {
            assert!(filled.mask[(row, col)] && unfilled.mask[(row, col)]);
        }
    }

    #[test]
    fn fill_support_out_of_range_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let err_array = ones_err((3, 3));
        let mut config = trivial_config();
        config.fill_min_cardinal_support = Some(2); // < 3: would run away
        let err = grow_mask(
            data.view(),
            data.view(),
            Some(err_array.view()),
            None,
            &[(1, 1)],
            &config,
        )
        .unwrap_err();
        assert_eq!(err, GrowError::FillSupportInvalid { value: 2 });
    }

    #[test]
    fn gradient_percentile_invalid_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let config = GrowthConfig {
            connectivity: Connectivity::Eight,
            stop: StopCriterion {
                snr: None,
                gradient: Some(GradientStop {
                    ratio_threshold: 1.0,
                    hysteresis: 2,
                    lo_percentile: 99.0,
                    hi_percentile: 75.0, // lo >= hi: invalid
                }),
            },
            shape_weight: 0.0,
            min_neighbor_support: 1,
            min_pixels_before_shape_gate: 0,
            fill_min_cardinal_support: None,
            min_pixels_before_stop_check: 5,
            check_interval: 1,
            annulus_thickness: 2,
        };
        let err = grow_mask(data.view(), data.view(), None, None, &[(1, 1)], &config).unwrap_err();
        assert_eq!(
            err,
            GrowError::GradientPercentileInvalid { lo: 99.0, hi: 75.0 }
        );
    }

    #[test]
    fn min_neighbor_support_too_large_errors() {
        let data = Array2::<f64>::zeros((3, 3));
        let err_array = ones_err((3, 3));
        let mut config = trivial_config();
        config.connectivity = Connectivity::Four; // max 4 neighbours
        config.min_neighbor_support = 5;
        let err = grow_mask(
            data.view(),
            data.view(),
            Some(err_array.view()),
            None,
            &[(1, 1)],
            &config,
        )
        .unwrap_err();
        assert_eq!(
            err,
            GrowError::MinNeighborSupportTooLarge {
                min_neighbor_support: 5,
                max_neighbors: 4,
            }
        );
    }
}

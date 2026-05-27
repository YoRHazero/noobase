//! Morphological dilation and inner/outer annulus extraction.
//!
//! These are the geometric primitives consumed by the stop-criterion
//! code. Iterative dilation expands a boolean mask along a chosen
//! [`Connectivity`] for a configurable number of steps; the two-ring
//! annulus extraction yields the pair of rings used by both the SNR
//! and the radial-gradient stops:
//!
//! ```text
//!     inner = (dilate(mask, t)      \ mask)             ∩ label_allowed
//!     outer = (dilate(mask, 2t)     \ dilate(mask, t))  ∩ label_allowed
//! ```
//!
//! where `t` is `annulus_thickness` and the label filter is dropped
//! when no [`LabelInput`] is supplied. The implementation reuses
//! `dilate(mask, t)` to compute `dilate(mask, 2t)` (morphological
//! dilation by the same structuring element composes additively, so
//! `dilate(dilate(mask, t), t) == dilate(mask, 2t)`), halving the work.

use ndarray::{Array2, ArrayView2};

use super::config::{Connectivity, LabelInput};

/// Inner / outer rings returned by [`extract_annuli`]. Both arrays have
/// the same shape as the input mask.
pub(super) struct Annuli {
    pub inner: Array2<bool>,
    pub outer: Array2<bool>,
}

/// Iteratively dilate `mask` by `iterations` steps along `connectivity`.
///
/// The implementation double-buffers (read `current`, write `next`,
/// swap) so that pixels set true in one iteration do not propagate
/// further within the same iteration — without this, `iterations`
/// passes would produce a geometry different from `iterations` true
/// dilations.
pub(super) fn dilate(
    mask: ArrayView2<bool>,
    connectivity: Connectivity,
    iterations: usize,
) -> Array2<bool> {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];
    let mut current = mask.to_owned();
    if iterations == 0 {
        return current;
    }
    let offsets = connectivity.offsets();
    let mut next = current.clone();
    for _ in 0..iterations {
        next.assign(&current);
        for row in 0..rows {
            for col in 0..cols {
                if !current[(row, col)] {
                    continue;
                }
                for &(d_row, d_col) in offsets {
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
                    next[(next_row, next_col)] = true;
                }
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    current
}

/// Extract the inner and outer annulus rings around `mask`.
///
/// `thickness` is the per-ring dilation count; `connectivity` matches
/// the topology the mask grows in. When `label` is supplied, both
/// rings are intersected with the allowed-label set on a per-pixel
/// basis (a pixel whose label is not in `label.allowed` is excluded
/// from both rings).
///
/// `thickness == 0` yields two all-false rings (the dilations collapse
/// to `mask`, so the set differences are empty). Callers that treat an
/// empty ring as "skip this stop check" therefore degrade gracefully.
pub(super) fn extract_annuli(
    mask: ArrayView2<bool>,
    label: Option<&LabelInput>,
    connectivity: Connectivity,
    thickness: usize,
) -> Annuli {
    let rows = mask.shape()[0];
    let cols = mask.shape()[1];

    let dilated_1 = dilate(mask, connectivity, thickness);
    // dilate(_, t) ∘ dilate(_, t) = dilate(_, 2t), so reuse dilated_1.
    let dilated_2 = dilate(dilated_1.view(), connectivity, thickness);

    let mut inner = Array2::<bool>::from_elem((rows, cols), false);
    let mut outer = Array2::<bool>::from_elem((rows, cols), false);

    for row in 0..rows {
        for col in 0..cols {
            let label_ok = match label {
                Some(label) => label.allowed.contains(&label.map[(row, col)]),
                None => true,
            };
            if !label_ok {
                continue;
            }
            let in_mask = mask[(row, col)];
            let in_d1 = dilated_1[(row, col)];
            let in_d2 = dilated_2[(row, col)];
            if in_d1 && !in_mask {
                inner[(row, col)] = true;
            }
            if in_d2 && !in_d1 {
                outer[(row, col)] = true;
            }
        }
    }

    Annuli { inner, outer }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn count_true(mask: &Array2<bool>) -> usize {
        mask.iter().filter(|&&v| v).count()
    }

    #[test]
    fn dilate_expands_correctly_per_connectivity() {
        let mut mask = Array2::<bool>::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        // Four: single step gives the 5-cell plus sign.
        let four = dilate(mask.view(), Connectivity::Four, 1);
        assert_eq!(count_true(&four), 5);
        for &(row, col) in &[(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)] {
            assert!(four[(row, col)], "Four-dilated must include ({row}, {col})");
        }

        // Eight: single step gives the 3x3 square.
        let eight = dilate(mask.view(), Connectivity::Eight, 1);
        assert_eq!(count_true(&eight), 9);
        for row in 1..=3 {
            for col in 1..=3 {
                assert!(
                    eight[(row, col)],
                    "Eight-dilated must include ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn extract_annuli_yields_inner_outer_with_label_gating() {
        // Single-pixel mask at (2, 2) in a 5x5 image.
        let mut mask = Array2::<bool>::from_elem((5, 5), false);
        mask[(2, 2)] = true;

        // Label map: everything is label 1 except (2, 3) which is 2.
        // With allowed = [1], the cell (2, 3) should be excluded from
        // the inner ring even though geometrically it is a neighbour.
        let mut label_map = Array2::<i32>::from_elem((5, 5), 1);
        label_map[(2, 3)] = 2;
        let label = LabelInput {
            map: label_map.view(),
            allowed: vec![1],
        };

        let annuli = extract_annuli(mask.view(), Some(&label), Connectivity::Four, 1);

        // Inner = Four-neighbours of (2, 2) minus the label-gated (2, 3).
        let expected_inner = [(1, 2), (2, 1), (3, 2)];
        assert_eq!(count_true(&annuli.inner), expected_inner.len());
        for &(row, col) in &expected_inner {
            assert!(
                annuli.inner[(row, col)],
                "inner must include ({row}, {col})"
            );
        }

        // Outer = distance-2 cells under Four-connectivity (the 8 cells
        // reachable in exactly two Four-steps that lie outside the
        // distance-1 ring). None of them is (2, 3), so the label gate
        // does not subtract from outer here.
        assert_eq!(count_true(&annuli.outer), 8);
    }
}

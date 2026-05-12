use ndarray::{Array1, ArrayView1};

use crate::bins::Grid;
use crate::float::Float;

/// Two-pointer sweep over source and target bin edges. The visitor is invoked
/// once for each non-empty linear-space intersection `(target_index,
/// source_index, overlap_width)`. Overlap is measured as a plain difference of
/// edge values, regardless of whether the underlying Grid was constructed with
/// `Linear` or `Log` spacing — log-uniform grids are not log-uniform once their
/// edges are taken, so a single convention (linear width on the edge axis)
/// keeps the semantics consistent across spacings.
///
/// The traversal is O(M + N) where M and N are the number of source and target
/// bins respectively. If the two grids do not overlap at all, the visitor is
/// simply never called — this is not considered an error.
pub fn for_each<T, F>(source: &Grid<T>, target: &Grid<T>, mut visitor: F)
where
    T: Float,
    F: FnMut(usize, usize, T),
{
    let source_edges_grid = source.to_edges();
    let target_edges_grid = target.to_edges();
    let source_edges = source_edges_grid.values();
    let target_edges = target_edges_grid.values();
    let source_bin_count = source_edges.len() - 1;
    let target_bin_count = target_edges.len() - 1;
    if source_bin_count == 0 || target_bin_count == 0 {
        return;
    }
    let mut source_index = 0usize;
    let mut target_index = 0usize;
    while source_index < source_bin_count && target_index < target_bin_count {
        let source_lo = source_edges[source_index];
        let source_hi = source_edges[source_index + 1];
        let target_lo = target_edges[target_index];
        let target_hi = target_edges[target_index + 1];
        let lo = if source_lo > target_lo {
            source_lo
        } else {
            target_lo
        };
        let hi = if source_hi < target_hi {
            source_hi
        } else {
            target_hi
        };
        if hi > lo {
            visitor(target_index, source_index, hi - lo);
        }
        // Advance whichever bin ends first; if both end at the same edge,
        // advance both to avoid emitting a zero-width pair on the next step.
        if source_hi < target_hi {
            source_index += 1;
        } else if target_hi < source_hi {
            target_index += 1;
        } else {
            source_index += 1;
            target_index += 1;
        }
    }
}

/// Flux-density-conserving rebin from `source` onto `target`.
///
/// For each target bin `i`,
/// ```text
/// out[i] = (Σ_j overlap[i, j] * source_values[j]) / target_width[i]
/// ```
/// where `overlap[i, j]` is the linear-space intersection width between target
/// bin `i` and source bin `j`, and `target_width[i]` is the full linear-space
/// width of target bin `i`. This preserves the integral
/// `Σ_i out[i] * target_width[i]` over the region where source and target
/// overlap, so the values behave like a density (flux per unit x).
///
/// Target bins that lie partially or fully outside the source range are NOT
/// treated as errors: the sum is simply taken over whatever overlap exists.
/// Callers that need to mask such bins should use [`coverage`].
///
/// Panics if `source_values.len()` does not match the number of source bins
/// (i.e. `source.to_edges().len() - 1`).
pub fn rebin<T: Float>(
    source: &Grid<T>,
    source_values: ArrayView1<T>,
    target: &Grid<T>,
) -> Array1<T> {
    let source_edges_grid = source.to_edges();
    let target_edges_grid = target.to_edges();
    let source_bin_count = source_edges_grid.len() - 1;
    let target_bin_count = target_edges_grid.len() - 1;
    assert_eq!(
        source_values.len(),
        source_bin_count,
        "source_values length {} does not match source bin count {}",
        source_values.len(),
        source_bin_count
    );
    let target_edges = target_edges_grid.values();
    let mut weighted_sum = Array1::<T>::zeros(target_bin_count);
    for_each(
        &source_edges_grid,
        &target_edges_grid,
        |target_index, source_index, overlap_width| {
            weighted_sum[target_index] =
                weighted_sum[target_index] + overlap_width * source_values[source_index];
        },
    );
    let mut output = Array1::<T>::zeros(target_bin_count);
    for target_index in 0..target_bin_count {
        let target_width = target_edges[target_index + 1] - target_edges[target_index];
        if target_width > T::zero() {
            output[target_index] = weighted_sum[target_index] / target_width;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{GridKind, Spacing};
    use ndarray::array;

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn linear_edges(values: &[f64]) -> Grid<f64> {
        Grid::new(
            values.iter().copied().collect(),
            Spacing::Linear,
            GridKind::Edges,
        )
        .unwrap()
    }

    #[test]
    fn for_each_identical_grids_one_to_one() {
        let grid = linear_edges(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut calls: Vec<(usize, usize, f64)> = Vec::new();
        for_each(&grid, &grid, |target_index, source_index, overlap_width| {
            calls.push((target_index, source_index, overlap_width));
        });
        assert_eq!(calls.len(), 4);
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call.0, i);
            assert_eq!(call.1, i);
            assert!(approx_eq(call.2, 1.0, TOL));
        }
    }

    #[test]
    fn for_each_disjoint_below_emits_nothing() {
        let source = linear_edges(&[0.0, 1.0, 2.0]);
        let target = linear_edges(&[10.0, 11.0, 12.0]);
        let mut count = 0usize;
        for_each(&source, &target, |_, _, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_disjoint_above_emits_nothing() {
        let source = linear_edges(&[10.0, 11.0, 12.0]);
        let target = linear_edges(&[0.0, 1.0, 2.0]);
        let mut count = 0usize;
        for_each(&source, &target, |_, _, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_target_spans_two_source_bins() {
        // source bins: [0,1), [1,2), [2,3)
        // target bin:  [0.5, 2.5) -> overlaps source bin 0 by 0.5, bin 1 by 1.0, bin 2 by 0.5
        let source = linear_edges(&[0.0, 1.0, 2.0, 3.0]);
        let target = linear_edges(&[0.5, 2.5]);
        let mut calls: Vec<(usize, usize, f64)> = Vec::new();
        for_each(
            &source,
            &target,
            |target_index, source_index, overlap_width| {
                calls.push((target_index, source_index, overlap_width));
            },
        );
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1, 0);
        assert!(approx_eq(calls[0].2, 0.5, TOL));
        assert_eq!(calls[1].1, 1);
        assert!(approx_eq(calls[1].2, 1.0, TOL));
        assert_eq!(calls[2].1, 2);
        assert!(approx_eq(calls[2].2, 0.5, TOL));
        let total: f64 = calls.iter().map(|c| c.2).sum();
        assert!(approx_eq(total, 2.0, TOL));
    }

    #[test]
    fn for_each_log_source_linear_target_linear_widths() {
        // Log-spaced source edges at 1, 10, 100. Linear target edges at 5, 50.
        // Expected linear-space overlap widths:
        //   target bin [5, 50) vs source bin [1, 10):   overlap = 10 - 5  = 5
        //   target bin [5, 50) vs source bin [10, 100): overlap = 50 - 10 = 40
        let source =
            Grid::new(array![1.0_f64, 10.0, 100.0], Spacing::Log, GridKind::Edges).unwrap();
        let target = linear_edges(&[5.0, 50.0]);
        let mut calls: Vec<(usize, usize, f64)> = Vec::new();
        for_each(
            &source,
            &target,
            |target_index, source_index, overlap_width| {
                calls.push((target_index, source_index, overlap_width));
            },
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1, 0);
        assert!(approx_eq(calls[0].2, 5.0, TOL));
        assert_eq!(calls[1].1, 1);
        assert!(approx_eq(calls[1].2, 40.0, TOL));
    }

    #[test]
    fn rebin_identity_linear() {
        let grid = linear_edges(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let values = array![1.0_f64, 2.0, 3.0, 4.0];
        let output = rebin(&grid, values.view(), &grid);
        assert_eq!(output.len(), values.len());
        for i in 0..values.len() {
            assert!(approx_eq(output[i], values[i], TOL));
        }
    }

    #[test]
    fn rebin_identity_log() {
        let grid = Grid::<f64>::logspace(1.0, 10000.0, 5, GridKind::Edges);
        let values = array![1.0_f64, 2.5, 7.0, 3.0];
        let output = rebin(&grid, values.view(), &grid);
        for i in 0..values.len() {
            assert!(approx_eq(output[i], values[i], TOL));
        }
    }

    #[test]
    fn rebin_constant_input_preserved_on_fully_covered_bins() {
        // Source bins covering [0, 10) with 10 unit-width bins, all-equal value c.
        // Target bins covering [2, 8) with 3 width-2 bins are all fully covered, so
        // the rebinned density should equal c there.
        // The first/last target bins at the original-grid edges would only partially
        // overlap, but we choose target entirely inside source to keep the property
        // exact for every output entry. Partial-coverage bins at the edges of the
        // source range are NOT required to equal c.
        let constant_value = 4.25_f64;
        let source = linear_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let source_values = Array1::<f64>::from_elem(10, constant_value);
        let target = linear_edges(&[2.0, 4.0, 6.0, 8.0]);
        let output = rebin(&source, source_values.view(), &target);
        assert_eq!(output.len(), 3);
        for value in output.iter() {
            assert!(approx_eq(*value, constant_value, TOL));
        }
    }

    #[test]
    fn rebin_downsample_two_to_one() {
        // Source: 4 width-1 bins with values [a, b, c, d]
        // Target: 2 width-2 bins -> expected [(a+b)/2, (c+d)/2]
        let source = linear_edges(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target = linear_edges(&[0.0, 2.0, 4.0]);
        let values = array![1.0_f64, 3.0, 5.0, 9.0];
        let output = rebin(&source, values.view(), &target);
        assert_eq!(output.len(), 2);
        assert!(approx_eq(output[0], (1.0 + 3.0) / 2.0, TOL));
        assert!(approx_eq(output[1], (5.0 + 9.0) / 2.0, TOL));
    }

    #[test]
    #[should_panic(expected = "source_values length")]
    fn rebin_panics_on_length_mismatch() {
        let source = linear_edges(&[0.0, 1.0, 2.0, 3.0]);
        let target = linear_edges(&[0.0, 1.5, 3.0]);
        let bad_values = array![1.0_f64, 2.0]; // 2 != 3 source bins
        let _ = rebin(&source, bad_values.view(), &target);
    }

    #[test]
    fn for_each_one_target_covers_many_source_bins() {
        // Source bins: width-1 from 0 to 5 (i.e. 5 source bins).
        // Target bin: [0, 5) covers all of them with their full widths.
        let source = linear_edges(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let target = linear_edges(&[0.0, 5.0]);
        let mut calls: Vec<(usize, usize, f64)> = Vec::new();
        for_each(
            &source,
            &target,
            |target_index, source_index, overlap_width| {
                calls.push((target_index, source_index, overlap_width));
            },
        );
        assert_eq!(calls.len(), 5);
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call.0, 0);
            assert_eq!(call.1, i);
            assert!(approx_eq(call.2, 1.0, TOL));
        }
    }
}

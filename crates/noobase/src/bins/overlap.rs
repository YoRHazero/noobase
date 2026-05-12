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

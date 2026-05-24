//! Statistical helpers shared across image-pipeline modules.
//!
//! Centralizes the small set of robust statistics (currently: median) so
//! their algorithm and "no-data" policy stay in one place. The empty-slice
//! policy is intentionally externalized: callers pick the default that
//! fits their context (e.g. `0.0` for a background offset, `f64::NAN` for
//! a missing measurement).

/// Median of `values` in O(n) average time via `select_nth_unstable_by`.
/// Returns `None` for an empty slice so the caller can choose the no-data
/// default.
///
/// The slice is reordered in place (partial sort, not a full sort).
/// Callers must filter non-finite samples upstream: the comparator uses
/// `partial_cmp(...).expect(...)` and panics on `NaN`.
pub(crate) fn median_in_place(values: &mut [f64]) -> Option<f64> {
    let length = values.len();
    if length == 0 {
        return None;
    }
    let mid = length / 2;
    let compare = |a: &f64, b: &f64| {
        a.partial_cmp(b)
            .expect("non-finite value in median input")
    };
    let (left, pivot, _) = values.select_nth_unstable_by(mid, compare);
    if length % 2 == 1 {
        Some(*pivot)
    } else {
        // `left.len() == mid >= 1` here (even branch implies length >= 2),
        // so `reduce` always returns `Some`.
        let lower = left.iter().copied().reduce(f64::max).unwrap();
        Some(0.5 * (lower + *pivot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-sort reference: identical algorithm to the per-module helpers
    /// removed in favor of `median_in_place`. Kept here so we can prove
    /// the `select_nth_unstable_by` version agrees value-for-value.
    fn naive_median(values: &mut [f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let length = values.len();
        Some(if length % 2 == 1 {
            values[length / 2]
        } else {
            0.5 * (values[length / 2 - 1] + values[length / 2])
        })
    }

    #[test]
    fn empty_slice_returns_none() {
        let mut values: Vec<f64> = Vec::new();
        assert_eq!(median_in_place(&mut values), None);
    }

    #[test]
    fn single_element_returns_itself() {
        let mut values = vec![3.5];
        assert_eq!(median_in_place(&mut values), Some(3.5));
    }

    #[test]
    fn two_elements_returns_average() {
        let mut values = vec![1.0, 4.0];
        assert_eq!(median_in_place(&mut values), Some(2.5));
    }

    #[test]
    fn odd_length_returns_middle() {
        let mut values = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        assert_eq!(median_in_place(&mut values), Some(3.0));
    }

    #[test]
    fn even_length_returns_average_of_two_middles() {
        let mut values = vec![3.0, 1.0, 4.0, 2.0];
        assert_eq!(median_in_place(&mut values), Some(2.5));
    }

    #[test]
    fn duplicates_around_pivot() {
        // Stress the "max of left partition" path when the (mid-1)-th
        // sorted value equals the pivot itself. With select_nth_unstable
        // the left partition may not be globally sorted, but every entry
        // is guaranteed `<= pivot`, so `max(left)` is still the correct
        // lower middle.
        let mut values = vec![1.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(median_in_place(&mut values), Some(5.0));
    }

    #[test]
    fn matches_naive_reference_across_random_inputs() {
        // SplitMix64 — same dependency-free PRNG used elsewhere in the
        // crate so the comparison is fully deterministic.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for trial in 1..=200usize {
            let length = (trial % 23) + 1; // exercise 1..=23
            let values: Vec<f64> = (0..length)
                .map(|_| {
                    // Map the raw u64 to a roughly [-100, 100) f64.
                    let signed = next() as i64 as f64;
                    signed / (i64::MAX as f64 / 100.0)
                })
                .collect();
            let mut input = values.clone();
            let mut reference_input = values.clone();
            let got = median_in_place(&mut input);
            let want = naive_median(&mut reference_input);
            assert_eq!(got, want, "trial {trial} values {values:?}");
        }
    }
}

//! Statistical helpers shared across image-pipeline modules.
//!
//! Centralizes the small set of robust statistics (median, MAD-based
//! sigma) so their algorithm and "no-data" policy stay in one place. The
//! empty-slice policy for primitives like [`median_in_place`] is
//! intentionally externalized: callers pick the default that fits their
//! context (e.g. `0.0` for a background offset, `f64::NAN` for a missing
//! measurement). The higher-level [`mad_sigma`] picks `NaN` itself since
//! a robust scale is undefined on no data.

/// `1 / 0.6745`: scales a median-absolute-deviation to the equivalent
/// Gaussian standard deviation (the asymptotic conversion at the normal
/// distribution).
pub(crate) const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

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
    let compare = |a: &f64, b: &f64| a.partial_cmp(b).expect("non-finite value in median input");
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

/// Robust Gaussian-equivalent scale of a sample:
/// `MAD_TO_SIGMA * median(|x - median(x)|)`.
///
/// Returns `f64::NAN` for an empty input (a robust scale is undefined
/// with no data). Callers must filter non-finite samples upstream —
/// [`median_in_place`] panics on `NaN`.
///
/// Allocates one `Vec<f64>` of length `samples.len()` and reuses it as
/// scratch for both the center and the deviations.
pub(crate) fn mad_sigma(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut scratch = samples.to_vec();
    // `unwrap` is safe: `scratch.len() == samples.len() >= 1`.
    let center = median_in_place(&mut scratch).unwrap();
    // `scratch` now holds the same multiset of values in a partitioned
    // (not sorted) order. For `|x - center|` the order is irrelevant, so
    // we reuse the buffer instead of reallocating.
    for value in scratch.iter_mut() {
        *value = (*value - center).abs();
    }
    let mad = median_in_place(&mut scratch).unwrap();
    MAD_TO_SIGMA * mad
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
    fn mad_sigma_empty_returns_nan() {
        assert!(mad_sigma(&[]).is_nan());
    }

    #[test]
    fn mad_sigma_odd_length_matches_hand_calculation() {
        // samples = [1, 2, 3, 4, 5]; median = 3;
        // deviations = [2, 1, 0, 1, 2]; MAD = 1; sigma = MAD_TO_SIGMA.
        let got = mad_sigma(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((got - MAD_TO_SIGMA).abs() < 1e-15);
    }

    #[test]
    fn mad_sigma_even_length_matches_hand_calculation() {
        // samples = [1, 2, 3, 4]; median = 2.5;
        // deviations = [1.5, 0.5, 0.5, 1.5]; MAD = (0.5+1.5)/2 = 1;
        // sigma = MAD_TO_SIGMA.
        let got = mad_sigma(&[1.0, 2.0, 3.0, 4.0]);
        assert!((got - MAD_TO_SIGMA).abs() < 1e-15);
    }

    #[test]
    fn mad_sigma_invariant_under_permutation() {
        // Robust scale is a property of the multiset; permuting the input
        // must not change the result (bit-equal in floating point because
        // the deviations are reduced through the same median routine).
        let baseline = mad_sigma(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let permuted = mad_sigma(&[7.0, 1.0, 5.0, 3.0, 6.0, 2.0, 4.0]);
        assert_eq!(baseline, permuted);
    }

    #[test]
    fn mad_sigma_scales_linearly() {
        // sigma(k * x) = |k| * sigma(x) — basic equivariance check.
        let baseline = mad_sigma(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaled = mad_sigma(&[3.0, 6.0, 9.0, 12.0, 15.0]);
        assert!((scaled - 3.0 * baseline).abs() < 1e-14);
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

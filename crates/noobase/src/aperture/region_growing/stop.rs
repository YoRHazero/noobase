//! Stop-criterion evaluation with hysteresis.
//!
//! [`StopState`] owns the per-criterion violation counters that persist
//! across stop checks, and [`StopState::evaluate`] evaluates whichever
//! criteria are enabled in [`StopCriterion`] against a pre-computed
//! [`Annuli`] pair. The annulus extraction itself belongs to
//! [`super::annulus`] — this file is the statistics + hysteresis layer
//! on top of the geometric primitives.
//!
//! Hysteresis semantics: a counter is only incremented when the
//! criterion is in violation, and the "have we hit the threshold?"
//! check sits inside the violation branch. Consequently `hysteresis = 0`
//! degrades to `hysteresis = 1` rather than firing on a passing check —
//! there is no zero-violation-count fires-anyway pathology, so no
//! `HysteresisZero` invariant is needed.

use ndarray::ArrayView2;

use super::annulus::Annuli;
use super::config::{GradientStop, SnrStop, StopCriterion};
use super::result::StopReason;

/// Per-criterion hysteresis state carried across stop checks within a
/// single `grow_mask` invocation.
pub(super) struct StopState {
    snr_violation_count: usize,
    gradient_violation_count: usize,
}

impl StopState {
    pub(super) fn new() -> Self {
        Self {
            snr_violation_count: 0,
            gradient_violation_count: 0,
        }
    }

    /// Evaluate every enabled stop criterion against `annuli` in a
    /// fixed order (SNR first, gradient second). The order matters when
    /// both fire on the same check: [`StopReason::SnrBelow`] wins by
    /// virtue of being evaluated first.
    pub(super) fn evaluate(
        &mut self,
        annuli: &Annuli,
        data: ArrayView2<f64>,
        err: Option<ArrayView2<f64>>,
        criterion: &StopCriterion,
    ) -> Option<StopReason> {
        if let Some(snr) = criterion.snr
            && let Some(reason) = self.evaluate_snr(annuli, data, err, snr)
        {
            return Some(reason);
        }
        if let Some(gradient) = criterion.gradient
            && let Some(reason) = self.evaluate_gradient(annuli, data, gradient)
        {
            return Some(reason);
        }
        None
    }

    fn evaluate_snr(
        &mut self,
        annuli: &Annuli,
        data: ArrayView2<f64>,
        err: Option<ArrayView2<f64>>,
        config: SnrStop,
    ) -> Option<StopReason> {
        // The err/SnrStop bidirectional binding is enforced by
        // `grow_mask`; reaching this branch with `err == None` is a
        // bug in `grow_mask`, not a caller-facing precondition.
        let err = err.expect("invariant: SnrStop enabled implies err.is_some()");

        let rows = annuli.inner.shape()[0];
        let cols = annuli.inner.shape()[1];

        let mut flux_sum: f64 = 0.0;
        let mut err_squared_sum: f64 = 0.0;
        let mut inner_count: usize = 0;
        for row in 0..rows {
            for col in 0..cols {
                if annuli.inner[(row, col)] {
                    inner_count += 1;
                    flux_sum += data[(row, col)];
                    let pixel_err = err[(row, col)];
                    err_squared_sum += pixel_err * pixel_err;
                }
            }
        }

        if inner_count == 0 {
            // Per spec: empty annulus resets the counter (don't divide
            // by zero, don't accumulate violations against geometry we
            // can't measure).
            self.snr_violation_count = 0;
            return None;
        }

        let snr = flux_sum / err_squared_sum.sqrt();
        if snr < config.threshold {
            self.snr_violation_count += 1;
            if self.snr_violation_count >= config.hysteresis {
                return Some(StopReason::SnrBelow);
            }
        } else {
            self.snr_violation_count = 0;
        }
        None
    }

    fn evaluate_gradient(
        &mut self,
        annuli: &Annuli,
        data: ArrayView2<f64>,
        config: GradientStop,
    ) -> Option<StopReason> {
        let rows = annuli.inner.shape()[0];
        let cols = annuli.inner.shape()[1];

        let mut inner_values: Vec<f64> = Vec::new();
        let mut outer_values: Vec<f64> = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                let pixel = data[(row, col)];
                // Non-finite pixels (NaN-masked / dead) cannot take part
                // in a sorted percentile band; drop them from both rings.
                if !pixel.is_finite() {
                    continue;
                }
                if annuli.inner[(row, col)] {
                    inner_values.push(pixel);
                }
                if annuli.outer[(row, col)] {
                    outer_values.push(pixel);
                }
            }
        }

        if inner_values.is_empty() || outer_values.is_empty() {
            // Either ring empty (no geometry, or all non-finite) resets
            // the counter: we cannot measure a gradient we cannot see.
            self.gradient_violation_count = 0;
            return None;
        }

        let inner_band = percentile_band_mean(
            &mut inner_values,
            config.lo_percentile,
            config.hi_percentile,
        );
        let outer_band = percentile_band_mean(
            &mut outer_values,
            config.lo_percentile,
            config.hi_percentile,
        );
        let ratio = outer_band / inner_band;

        // `inner_band == 0` yields inf (positive outer) or NaN (zero
        // outer): `inf > threshold` is true (a genuine flip out of a
        // zero-level basin) and `NaN > threshold` is false (counter
        // resets), so no special-casing is needed.
        if ratio > config.ratio_threshold {
            self.gradient_violation_count += 1;
            if self.gradient_violation_count >= config.hysteresis {
                return Some(StopReason::GradientFlip);
            }
        } else {
            self.gradient_violation_count = 0;
        }
        None
    }
}

/// Linearly-interpolated percentile of a sorted-ascending slice, matching
/// numpy's default ("linear") method. `sorted` must be non-empty and
/// `percentile` in `[0, 100]`.
fn percentile_sorted(sorted: &[f64], percentile: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (percentile / 100.0) * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = rank - lower as f64;
    sorted[lower] + frac * (sorted[upper] - sorted[lower])
}

/// Mean of the values whose magnitude falls in the
/// `[lo_percentile, hi_percentile]` percentile band of `values`.
///
/// Sorts `values` in place. `values` must be non-empty and finite, and
/// the bounds satisfy `0 <= lo < hi <= 100` (validated by `grow_mask`).
/// The degenerate case where the band brackets no actual value (only
/// reachable for a handful of pixels) falls back to the plain mean so
/// the caller never divides by zero. `[0, 100]` reproduces that plain
/// mean exactly.
fn percentile_band_mean(values: &mut [f64], lo_percentile: f64, hi_percentile: f64) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let low_value = percentile_sorted(values, lo_percentile);
    let high_value = percentile_sorted(values, hi_percentile);
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values.iter() {
        if value >= low_value && value <= high_value {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return values.iter().sum::<f64>() / values.len() as f64;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_band_mean_isolates_the_bright_mode() {
        // 70% background (0) + 30% source (10). The plain mean is 3.0,
        // but the [75, 99] band drops the diluting background and keeps
        // the source pixels, so the band mean recovers the source level.
        let mut values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let band = percentile_band_mean(&mut values, 75.0, 99.0);
        assert!(
            (band - 10.0).abs() < 1e-9,
            "band mean should isolate the bright mode, got {band}"
        );
    }

    #[test]
    fn percentile_band_full_range_is_plain_mean() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let band = percentile_band_mean(&mut values, 0.0, 100.0);
        assert!(
            (band - 2.5).abs() < 1e-9,
            "[0, 100] must equal the plain mean, got {band}"
        );
    }

    #[test]
    fn percentile_band_trims_a_single_hot_pixel() {
        // A lone spike at the top is excluded by hi_percentile, while a
        // multi-pixel bright group (a real neighbour) would survive.
        let mut values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0];
        let band = percentile_band_mean(&mut values, 75.0, 95.0);
        assert!(
            band < 2.0,
            "a single hot pixel must be trimmed by the upper bound, got {band}"
        );
    }
}

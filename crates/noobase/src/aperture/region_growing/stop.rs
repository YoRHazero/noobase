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
        if let Some(snr) = criterion.snr {
            if let Some(reason) = self.evaluate_snr(annuli, data, err, snr) {
                return Some(reason);
            }
        }
        if let Some(gradient) = criterion.gradient {
            if let Some(reason) = self.evaluate_gradient(annuli, data, gradient) {
                return Some(reason);
            }
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

        let mut inner_sum: f64 = 0.0;
        let mut inner_count: usize = 0;
        let mut outer_sum: f64 = 0.0;
        let mut outer_count: usize = 0;
        for row in 0..rows {
            for col in 0..cols {
                let pixel = data[(row, col)];
                if annuli.inner[(row, col)] {
                    inner_count += 1;
                    inner_sum += pixel;
                }
                if annuli.outer[(row, col)] {
                    outer_count += 1;
                    outer_sum += pixel;
                }
            }
        }

        if inner_count == 0 || outer_count == 0 {
            // Per spec: either ring empty resets the counter (we cannot
            // measure a gradient we have no geometry for).
            self.gradient_violation_count = 0;
            return None;
        }

        let inner_mean = inner_sum / inner_count as f64;
        let outer_mean = outer_sum / outer_count as f64;
        let ratio = outer_mean / inner_mean;

        // NaN / inf flow through here without special-casing: `NaN >
        // threshold` is false (counter resets) and `inf > threshold` is
        // true (counter advances — correct when inner_mean = 0 with a
        // positive outer_mean, which really is a gradient flip).
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

//! Output and error types for [`grow_mask`](super::grow::grow_mask).

use ndarray::Array2;
use thiserror::Error;

/// Why the growth loop terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The SNR stop criterion fired (mask reached its hysteresis count
    /// for SNR below threshold). The normal success outcome on a
    /// well-isolated source.
    SnrBelow,
    /// The radial-gradient stop fired (mask was about to climb a
    /// neighbouring source). The normal success outcome in crowded
    /// fields.
    GradientFlip,
    /// The growth reached the cutout edge or exhausted the heap before
    /// any stop criterion fired. **This is a failure mode**: the
    /// returned mask is not trustworthy because the geometric annuli
    /// could not be evaluated past the edge. The caller should react by
    /// enlarging the cutout or tightening the seed selection.
    Filled,
}

/// Successful output of [`grow_mask`](super::grow::grow_mask).
#[derive(Debug, Clone)]
pub struct GrowthResult {
    /// Boolean source mask, same shape as the input data. `true` marks
    /// pixels belonging to the source.
    pub mask: Array2<bool>,
    /// Why the growth terminated. See [`StopReason`].
    pub stop_reason: StopReason,
    /// Number of pixels admitted into the mask *after* the seeds.
    /// I.e. `mask.iter().filter(|&&v| v).count() == seed_count + n_iterations`.
    pub n_iterations: usize,
}

/// Hard input-validation errors. Algorithmic outcomes (mask hitting the
/// edge, heap exhausted) are reported as [`StopReason::Filled`], not as
/// errors.
#[derive(Debug, Error, PartialEq)]
pub enum GrowError {
    /// A seed pixel coordinate lies outside the data array.
    #[error("seed pixel {seed:?} is out of bounds for data shape {shape:?}")]
    SeedOutOfBounds {
        seed: (usize, usize),
        shape: (usize, usize),
    },
}

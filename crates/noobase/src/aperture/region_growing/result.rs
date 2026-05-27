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
    /// The label map shape does not match the data shape.
    #[error("label map shape {label_shape:?} must equal data shape {data_shape:?}")]
    LabelShapeMismatch {
        label_shape: (usize, usize),
        data_shape: (usize, usize),
    },
    /// `LabelInput::allowed` was supplied but empty. An empty allowed
    /// list would forbid every pixel — almost certainly a caller bug, so
    /// we reject it rather than silently growing nothing.
    #[error("label.allowed must be non-empty")]
    LabelAllowedEmpty,
    /// A seed pixel sits on a label that is not in the allowed list. The
    /// reported `label` is the actual value at the seed coordinate.
    #[error("seed pixel {seed:?} sits on label {label}, which is not in allowed")]
    SeedOnDisallowedLabel { seed: (usize, usize), label: i32 },
}

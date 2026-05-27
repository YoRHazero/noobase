//! Input configuration for [`grow_mask`](super::grow::grow_mask).

use ndarray::ArrayView2;

/// Pixel adjacency used both for heap-neighbour expansion and for
/// annulus dilation. The two uses must share a single setting so that
/// the annuli are defined in the same topology the mask grows in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connected (axis-aligned neighbours only).
    Four,
    /// 8-connected (axis-aligned + diagonal neighbours).
    Eight,
}

/// Signal-to-noise stop criterion evaluated on the inner annulus.
///
/// Fires when the cumulative SNR inside the inner annulus
/// (`sum(flux) / sqrt(sum(err^2))`) stays strictly below `threshold` for
/// `hysteresis` consecutive checks. Requires `err` to be supplied to
/// [`grow_mask`](super::grow::grow_mask).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnrStop {
    pub threshold: f64,
    pub hysteresis: usize,
}

/// Radial-gradient flip stop criterion evaluated across the two annuli.
///
/// Fires when `mean(outer_annulus.flux) / mean(inner_annulus.flux)`
/// stays strictly above `ratio_threshold` for `hysteresis` consecutive
/// checks. This catches the case where the mask has reached the basin
/// between two sources and is about to climb the neighbour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStop {
    pub ratio_threshold: f64,
    pub hysteresis: usize,
}

/// The set of enabled stop criteria. At least one must be enabled.
///
/// The two criteria are combined with **OR** semantics: whichever
/// reaches its hysteresis count first terminates the growth. They
/// capture different failure modes (SNR = decaying into noise; gradient
/// = encountering another source) and in practice almost never trigger
/// simultaneously; if they do on the same check, [`SnrStop`] wins by
/// evaluation order.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StopCriterion {
    pub snr: Option<SnrStop>,
    pub gradient: Option<GradientStop>,
}

/// Algorithm configuration for [`grow_mask`](super::grow::grow_mask).
///
/// All fields are required. No defaults are exposed at the Rust layer:
/// reasonable defaults are scenario-specific (e.g. JWST/NIRCam high-z
/// galaxies) and belong to the calling boundary (the Python binding).
#[derive(Debug, Clone)]
pub struct GrowthConfig {
    /// Pixel adjacency for heap expansion and annulus dilation.
    pub connectivity: Connectivity,
    /// Enabled stop criteria (at least one must be set).
    pub stop: StopCriterion,
    /// Lower bound on the mask size before stop checks may fire.
    /// Prevents premature termination on the first handful of pixels.
    pub min_pixels_before_stop_check: usize,
    /// Evaluate stop criteria every `check_interval` admitted pixels.
    /// Must be `>= 1`.
    pub check_interval: usize,
    /// Number of morphological dilation iterations per annulus. The
    /// inner annulus is `dilate(mask, thickness) \ mask`; the outer
    /// annulus is `dilate(mask, 2 * thickness) \ dilate(mask, thickness)`.
    pub annulus_thickness: usize,
}

/// Optional segmentation constraint passed to
/// [`grow_mask`](super::grow::grow_mask).
///
/// `map` is a per-pixel integer label image (shape must equal the input
/// data). `allowed` is the *complete* whitelist of labels the mask may
/// occupy — the caller must include the background label explicitly if
/// growth into background is desired.
pub struct LabelInput<'a> {
    pub map: ArrayView2<'a, i32>,
    pub allowed: Vec<i32>,
}

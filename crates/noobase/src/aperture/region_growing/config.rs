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

impl Connectivity {
    /// Per-connectivity neighbour offsets as `(d_row, d_col)`. The
    /// returned slice is the canonical adjacency set shared between
    /// heap-driven mask expansion and morphological annulus dilation.
    pub fn offsets(self) -> &'static [(isize, isize)] {
        match self {
            Connectivity::Four => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            Connectivity::Eight => &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
        }
    }

    /// Maximum number of neighbours a pixel can have under this
    /// connectivity (4 or 8). Used to normalise the neighbour-support
    /// count into the `[0, 1]` fraction that drives the shape term in
    /// the heap priority, so `shape_weight` carries the same meaning
    /// regardless of connectivity.
    pub fn max_neighbors(self) -> usize {
        self.offsets().len()
    }
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
/// Fires when `band_mean(outer_annulus) / band_mean(inner_annulus)`
/// stays strictly above `ratio_threshold` for `hysteresis` consecutive
/// checks. This catches the case where the mask has reached the basin
/// between two sources and is about to climb the neighbour.
///
/// `band_mean` is the mean of each ring's pixels whose value falls in
/// the `[lo_percentile, hi_percentile]` percentile band of that ring.
/// The lower bound drops the sky pixels that otherwise dilute the ring
/// average (so a rising neighbour stays visible even when most of the
/// ring is background); the upper bound trims the brightest pixels
/// (cosmic rays / hot pixels) by count, which a bright multi-pixel
/// neighbour survives. `[0, 100]` recovers the plain ring mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStop {
    pub ratio_threshold: f64,
    pub hysteresis: usize,
    /// Lower percentile bound of the per-ring band mean (e.g. `75.0`).
    pub lo_percentile: f64,
    /// Upper percentile bound of the per-ring band mean (e.g. `99.0`).
    /// Must satisfy `0 <= lo_percentile < hi_percentile <= 100`.
    pub hi_percentile: f64,
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
///
/// # Shape regularisation
///
/// Three fields control how compact the mask stays, suppressing the
/// thin tendrils a pure brightest-pixel heap would chase along noise
/// ridges or bridges to neighbouring sources:
///
/// - `shape_weight` adds a soft reward for admitting pixels that already
///   have many in-mask neighbours, biasing the *order* of growth toward
///   filling concavities before extending arms.
/// - `min_neighbor_support` is a hard floor: once past the warm-up, a
///   pixel is refused admission until enough of its neighbours are in
///   the mask. This sets the minimum aperture "neck width" and directly
///   forbids one-pixel-wide filaments.
/// - `min_pixels_before_shape_gate` delays that floor so the seed core
///   can establish itself (a single seed's neighbours start with only
///   one in-mask neighbour, which the floor would otherwise deadlock).
#[derive(Debug, Clone)]
pub struct GrowthConfig {
    /// Pixel adjacency for heap expansion and annulus dilation.
    pub connectivity: Connectivity,
    /// Enabled stop criteria (at least one must be set).
    pub stop: StopCriterion,
    /// Soft shape-regularisation weight. The heap priority of a
    /// candidate pixel is `detection_value + shape_weight * fraction`,
    /// where `fraction` is its in-mask neighbour count divided by
    /// [`Connectivity::max_neighbors`]. `0.0` disables the soft term
    /// (pure brightest-first growth). Its natural scale is the detection
    /// noise level: at that scale the term only breaks ties near the
    /// noise floor (where tendrils form) and leaves real structure,
    /// whose detection value dominates, untouched.
    pub shape_weight: f64,
    /// Hard lower bound on a candidate pixel's in-mask neighbour count
    /// for admission, enforced once `min_pixels_before_shape_gate` is
    /// reached. `0` or `1` disables the floor (every frontier pixel has
    /// at least one in-mask neighbour by construction). Must not exceed
    /// [`Connectivity::max_neighbors`], otherwise no pixel could ever be
    /// admitted past the warm-up.
    pub min_neighbor_support: usize,
    /// Number of admitted pixels before the `min_neighbor_support` floor
    /// activates. Lets the seed core grow past the size where every
    /// frontier pixel has only a single in-mask neighbour.
    pub min_pixels_before_shape_gate: usize,
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

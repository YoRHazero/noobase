//! Adaptive aperture via greedy region growing.
//!
//! [`grow_mask`] takes a science image plus one or more seed pixels and
//! grows a boolean source mask outward by repeatedly admitting the
//! brightest in-bounds neighbour from a max-heap keyed on the input data.
//! Growth is gated by an optional segmentation `label` map (the mask may
//! only enter labels the caller has whitelisted), and is terminated by a
//! pair of geometric-annulus stop criteria evaluated on a periodic
//! schedule:
//!
//! - **SNR stop**: the cumulative signal-to-noise inside a 1-annulus ring
//!   just outside the current mask falls below a threshold.
//! - **Gradient stop**: the ratio of mean flux in the *outer* annulus to
//!   the *inner* annulus rises above a threshold (the radial profile has
//!   flipped — typically the mask has reached the basin between two
//!   sources).
//!
//! The two criteria are combined with **OR** semantics: whichever fires
//! first terminates the growth. Each carries an independent hysteresis
//! counter to suppress single-check noise. Touching the cutout edge is
//! reported as [`StopReason::Filled`] (a *failure* mode: the caller's
//! cutout was too small for the source).
//!
//! Algorithm parameters (heap-check cadence, annulus thickness,
//! hysteresis counts, thresholds) are deliberately required inputs on the
//! Rust side; default values are a science-scenario policy and live at
//! the Python binding boundary.

mod annulus;
pub mod config;
pub mod grow;
pub mod result;
mod stop;

pub use config::{Connectivity, GradientStop, GrowthConfig, LabelInput, SnrStop, StopCriterion};
pub use grow::grow_mask;
pub use result::{GrowError, GrowthResult, StopReason};

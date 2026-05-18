//! Convolution: pure correlation kernels, NaN-aware wrappers, and thin
//! domain entry points.
//!
//! Three layers, mirroring `bins::overlap` (pure `for_each` kernel ->
//! `rebin`/`coverage` semantic wrappers -> `Spectrum::rebin` domain
//! entry):
//!
//! - **Pure kernels** ([`conv1d`], [`conv_axis`], [`conv2d`]):
//!   NaN-naive `Σ w·v`, no normalization, no boundary renormalization.
//! - **Normalized-convolution wrappers** ([`conv2d_renorm`],
//!   [`conv_axis_renorm`]): NaN-as-missing, backend-uniform.
//! - **Domain entries** (later, `Spectrum::convolve_lsf`,
//!   `image::convolve_*`): the only surface that crosses FFI.
//!
//! The bare kernels compute **correlation** — the kernel is *not*
//! flipped. True convolution is a wrapper that flips the kernel first
//! (see `image::convolve_psf`). For a symmetric kernel correlation and
//! convolution coincide; they differ only for an asymmetric kernel,
//! where convolution mirrors it.

mod conv1d;
mod conv2d;
mod gaussian;
mod nan;

pub use conv1d::{conv_axis, conv1d};
pub use conv2d::conv2d;
pub use gaussian::{GaussianSampling, gaussian1d};
pub use nan::{conv_axis_renorm, conv2d_renorm};

// Shared crate-internal: the variable-width LSF path reuses this single
// erf source so it stays consistent with the fixed-kernel constructor.
pub(crate) use gaussian::normal_cdf;

/// Out-of-bounds handling for the bare correlation kernels.
///
/// Applies only to the pure [`conv1d`] / [`conv_axis`] / [`conv2d`].
/// The normalized-convolution wrappers deliberately do *not*
/// take a `Boundary`: they treat every out-of-bounds tap as missing
/// (zero-extension on both the filled and the validity pass), which is a
/// distinct and mutually exclusive boundary model. Reflect/Nearest would
/// keep the validity pass saturated at the edge, defeating the
/// missing-data renormalization, so the two are kept orthogonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Out-of-bounds taps contribute nothing (the signal is treated as
    /// `0` outside its support).
    Zero,
    /// Half-sample symmetric reflection: index `-1` maps to `0`, `n`
    /// maps to `n - 1` (the edge sample is repeated), matching
    /// `scipy.ndimage`'s default `reflect` mode.
    Reflect,
    /// Out-of-bounds taps clamp to the nearest in-bounds sample (edge
    /// replication).
    Nearest,
}

/// Kernel normalization, applied at kernel-construction time.
///
/// This is orthogonal to the boundary renormalization performed by the
/// NaN wrappers (ROADMAP D4): one is decided when the kernel is built,
/// the other when the convolution runs. Folding both into a single
/// "mode" enum would combinatorially explode, so they stay separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Σ kernel = 1 — flux-conserving (LSF, PSF).
    Sum,
    /// Σ kernel² = 1 — matched-filter S/N optimal (grism line search).
    L2,
    /// Raw samples, left unscaled.
    None,
}

/// Resolve a (possibly out-of-range) source index against an axis of
/// length `length` under `boundary`. Returns `None` only for a
/// [`Boundary::Zero`] out-of-range index (the tap contributes nothing).
///
/// This is the single shared boundary resolver for *every* bare kernel
/// ([`conv1d`], [`conv_axis`], [`conv2d`]) — 2-D correlation applies it
/// per axis. Keeping one implementation is a structural guarantee that
/// 1-D and 2-D agree at the edges; were each kernel to fold indices on
/// its own, any drift would silently desynchronize them. `length` is
/// assumed `>= 1` (callers early-return on empty axes).
#[inline]
pub(crate) fn resolve_index(source: isize, length: usize, boundary: Boundary) -> Option<usize> {
    let n = length as isize;
    if source >= 0 && source < n {
        return Some(source as usize);
    }
    match boundary {
        Boundary::Zero => None,
        Boundary::Nearest => Some(source.clamp(0, n - 1) as usize),
        Boundary::Reflect => {
            if n == 1 {
                return Some(0);
            }
            // Half-sample symmetric folding with period 2n: indices in
            // [n, 2n) mirror back onto [0, n). Works for arbitrarily
            // out-of-range indices, not just one kernel half-width.
            let period = 2 * n;
            let mut folded = source % period;
            if folded < 0 {
                folded += period;
            }
            if folded >= n {
                folded = period - 1 - folded;
            }
            Some(folded as usize)
        }
    }
}

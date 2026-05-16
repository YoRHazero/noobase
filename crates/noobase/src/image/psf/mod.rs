//! Point-spread-function model.
//!
//! Phase 2 ships the forward operator [`render`] (oversampled effective
//! PSF -> predicted detector stamps); Phase 3 adds its exact adjoint
//! [`accumulate`] (detector-grid residuals back-projected onto the model
//! grid). Later phases add the surrounding super-resolution solver under
//! this module.
//!
//! The separable bicubic Catmull-Rom interpolation weights live in the
//! psf-internal [`kernel`] module so that the forward operator and its
//! adjoint share one weight function by construction -- the structural
//! guarantee that `accumulate` is the exact transpose of `render`.

mod kernel;
pub mod accumulate;
pub mod render;

pub use accumulate::{AccumulateError, accumulate};
pub use render::{RenderError, render};

//! Point-spread-function model.
//!
//! Phase 2 ships the forward operator [`render`] (oversampled effective
//! PSF -> predicted detector stamps). Later phases add its exact adjoint
//! and the surrounding super-resolution solver under this module.
//!
//! The separable bicubic Catmull-Rom interpolation weights live in the
//! psf-internal [`kernel`] module so that the forward operator and its
//! adjoint share one weight function by construction.

mod kernel;
pub mod render;

pub use render::{RenderError, render};

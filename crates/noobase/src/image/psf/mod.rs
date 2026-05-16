//! Point-spread-function model.
//!
//! Phase 2 ships the forward operator [`render`] (oversampled effective
//! PSF -> predicted detector stamps). Later phases add its exact adjoint
//! and the surrounding super-resolution solver under this module.

pub mod render;

pub use render::{RenderError, render};

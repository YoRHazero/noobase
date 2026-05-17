//! Point-spread-function model.
//!
//! Phase 2 ships the forward operator [`render`] (oversampled effective
//! PSF -> predicted detector stamps); Phase 3 adds its exact adjoint
//! [`accumulate`] (detector-grid residuals back-projected onto the model
//! grid); Phase 4 adds [`robust_combine`] (cross-stamp robust combination
//! of an aligned native-resolution stack, for the extended-PSF wings);
//! Phase 5 adds [`solve_flux_background`] / [`refine_nuisance`] (per-star
//! flux/background/centroid refinement against the current model);
//! Phase 6 adds [`build_epsf`] (the core super-resolution iteration
//! driver that assembles the operator stack into a projected-Landweber
//! solver). Later phases add the extended-PSF stitching on top.
//!
//! The separable bicubic Catmull-Rom interpolation weights live in the
//! psf-internal [`kernel`] module so that the forward operator and its
//! adjoint share one weight function by construction -- the structural
//! guarantee that `accumulate` is the exact transpose of `render`.

mod kernel;
pub mod accumulate;
pub mod build_epsf;
pub mod nuisance;
pub mod render;
pub mod robust;

pub use accumulate::{AccumulateError, accumulate};
pub use build_epsf::{
    BuildEpsf, BuildEpsfError, BuildEpsfParams, ResidualReweight, build_epsf,
};
pub use nuisance::{
    FluxBackground, NuisanceError, NuisanceRefined, refine_nuisance,
    solve_flux_background,
};
pub use render::{RenderError, render};
pub use robust::{CombineMethod, RobustCombined, RobustError, robust_combine};

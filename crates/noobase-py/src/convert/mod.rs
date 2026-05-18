//! Boundary translation layer.
//!
//! Everything that turns the Python / numpy boundary into the typed
//! `noobase` core call (and back) lives here, so the per-surface
//! binding files (`grid`, `spectrum`, `overlap`, `photometry`, `image`,
//! `psf`) stay free of the repetitive dtype-dispatch and
//! array-extraction boilerplate. No business logic -- the core crate
//! owns every algorithm; this module only translates.

mod array;
mod dtype;

pub(crate) use array::typed_array1;
pub(crate) use dtype::{
    Scalar, dispatch_array, dtype_mismatch_error, grid_dtype_name, is_float32_dtype,
    with_grid_pair,
};

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
mod enums;
mod grid_input;
mod tuple;

pub(crate) use array::{
    optional_bool_array1, optional_typed_array1, required_typed_array1, typed_array1,
};
pub(crate) use dtype::{
    Scalar, dispatch_array, dtype_mismatch_error, grid_dtype_name, is_float32_dtype, map_inner,
    with_grid_pair, with_inner,
};
pub(crate) use enums::{kind_to_str, parse_convention, parse_kind, parse_spacing, spacing_to_str};
pub(crate) use grid_input::{GridChannel, build_grid_from_any, coerce_to_grid, grid_channel};
pub(crate) use tuple::band_triple;

//! Transitional shim. Every boundary helper now lives in
//! `crate::convert`; this only re-exports the few still imported by the
//! not-yet-migrated `photometry` binding and is removed once that file
//! moves over.

pub(crate) use crate::convert::{coerce_to_grid, parse_convention};

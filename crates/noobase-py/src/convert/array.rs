//! numpy array extraction at the boundary.
//!
//! Owns the small family of "extract a numpy array of an exact channel"
//! helpers so the per-binding code never re-implements the
//! readonly-extract / `to_owned` / dtype-error idiom.

use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::convert::{Scalar, dtype_mismatch_error};

/// Extract a 1-D numpy array already known to be channel `T` (e.g. a
/// value array paired with a typed [`Grid`](crate::grid::PyGrid)). A
/// dtype other than `T` is the canonical cross-input mismatch
/// `ValueError`, contextualised by `mismatch_context` and naming the
/// offending `role` (matching the historical `overlap` messages).
pub(crate) fn typed_array1<T: Scalar>(
    value: &Bound<'_, PyAny>,
    role: &str,
    mismatch_context: &str,
) -> PyResult<Array1<T>> {
    match value.extract::<PyReadonlyArray1<'_, T>>() {
        Ok(array) => Ok(array.as_array().to_owned()),
        Err(_) => Err(dtype_mismatch_error(
            T::DTYPE_NAME,
            &format!("{role} not {}", T::DTYPE_NAME),
            mismatch_context,
        )),
    }
}

//! Band-result Python tuple builders.
//!
//! Synthetic photometry returns a small fixed-shape tuple where the
//! error slot is `None` when absent; this is its single home so the
//! optional-None packing is not re-spelled per dtype arm.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn optional<'py>(py: Python<'py>, value: Option<f64>) -> PyResult<Py<PyAny>> {
    match value {
        Some(scalar) => scalar.into_py_any(py),
        None => Ok(py.None()),
    }
}

/// `(band_flux, band_error_or_None, coverage)`.
pub(crate) fn band_triple<'py>(
    py: Python<'py>,
    band_flux: f64,
    band_error: Option<f64>,
    coverage: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let tuple = PyTuple::new(
        py,
        [
            band_flux.into_py_any(py)?,
            optional(py, band_error)?,
            coverage.into_py_any(py)?,
        ],
    )?;
    Ok(tuple.into_any())
}

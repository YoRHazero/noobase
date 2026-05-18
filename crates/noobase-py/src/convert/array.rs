//! numpy array extraction at the boundary.
//!
//! Owns the small family of "extract a numpy array of an exact channel"
//! helpers so the per-binding code never re-implements the
//! readonly-extract / `to_owned` / dtype-error idiom.

use ndarray::{Array1, Array2, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
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

/// Extract a required 1-D channel-`T` array (e.g. a spectrum's paired
/// `transmission_values`). A non-`T` dtype is the canonical cross-input
/// mismatch `ValueError` contextualised by `mismatch_context`.
pub(crate) fn required_typed_array1<T: Scalar>(
    value: &Bound<'_, PyAny>,
    mismatch_context: &str,
) -> PyResult<Array1<T>> {
    match value.extract::<PyReadonlyArray1<'_, T>>() {
        Ok(array) => Ok(array.as_array().to_owned()),
        Err(_) => Err(dtype_mismatch_error(T::DTYPE_NAME, "other", mismatch_context)),
    }
}

/// Extract an optional 1-D channel-`T` companion array (e.g. a
/// spectrum's `error` paired with `flux`). Absent / Python `None` ->
/// `None`; a non-`T` dtype is the canonical cross-input mismatch
/// `ValueError` contextualised by `mismatch_context` (the historical
/// "flux vs <name>" form).
pub(crate) fn optional_typed_array1<T: Scalar>(
    value: Option<&Bound<'_, PyAny>>,
    mismatch_context: &str,
) -> PyResult<Option<Array1<T>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray1<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(dtype_mismatch_error(T::DTYPE_NAME, "other", mismatch_context)),
    }
}

/// Extract an optional 1-D `bool` array (e.g. a spectrum `mask`).
/// Absent / Python `None` -> `None`; a non-`bool` dtype yields
/// "`<name>` must be a 1-D numpy array of dtype bool".
pub(crate) fn optional_bool_array1(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array1<bool>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray1<'_, bool>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 1-D numpy array of dtype bool"
        ))),
    }
}

/// Extract an optional 2-D channel-`T` companion array (e.g. a stamp
/// `error` paired with `cutout`). Absent / Python `None` -> `None`; a
/// non-`T` dtype yields "`<name>` must be a 2-D numpy array of dtype
/// `<T>`".
pub(crate) fn optional_companion_array2<T: Scalar>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array2<T>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype {}",
            T::DTYPE_NAME
        ))),
    }
}

/// Extract an optional 3-D channel-`T` companion array (e.g. a per-pixel
/// `weight` paired with a `(N, h, w)` stack). Same contract as
/// [`optional_companion_array2`].
pub(crate) fn optional_companion_array3<T: Scalar>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array3<T>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray3<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 3-D numpy array of dtype {}",
            T::DTYPE_NAME
        ))),
    }
}

/// Extract a required 2-D `float64` array (the model / delta arrays that
/// are always f64 regardless of the data channel).
pub(crate) fn required_f64_array2(
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Array2<f64>> {
    match value.extract::<PyReadonlyArray2<'_, f64>>() {
        Ok(array) => Ok(array.as_array().to_owned()),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype float64"
        ))),
    }
}

/// Extract an optional 2-D `float64` array (always-f64 optionals such as
/// `psi_init` / `wing_confidence`).
pub(crate) fn optional_f64_array2(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array2<f64>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, f64>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype float64"
        ))),
    }
}

/// Extract an optional 2-D `bool` array (a per-pixel mask, `true =
/// invalid`).
pub(crate) fn optional_bool_array2(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array2<bool>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, bool>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype bool"
        ))),
    }
}

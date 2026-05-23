//! Coerce a Python value into a typed [`PyGrid`].
//!
//! Two entry points share the "build a `Grid` from numpy + flags"
//! boundary: [`build_grid_from_any`] (explicit spacing/kind) and
//! [`coerce_to_grid`] (accepts an already-built `Grid` or infers
//! centers vs edges from a paired array's length). Both keep the dtype
//! channel decided by the array's dtype.

use ::noobase::axis::{Grid as CoreGrid, GridKind, Spacing};
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{Scalar, dtype_mismatch_error};
use crate::axis::{GridInner, PyGrid};

/// Project a [`GridInner`] onto a concrete channel. The `Err` carries
/// the dtype name actually found, so callers can phrase the canonical
/// cross-input mismatch message.
pub(crate) trait GridChannel: ::noobase::Float + Sized {
    fn from_grid_inner(inner: GridInner) -> Result<::noobase::axis::Grid<Self>, &'static str>;
}

impl GridChannel for f32 {
    fn from_grid_inner(inner: GridInner) -> Result<::noobase::axis::Grid<f32>, &'static str> {
        match inner {
            GridInner::F32(grid) => Ok(grid),
            GridInner::F64(_) => Err("float64"),
        }
    }
}

impl GridChannel for f64 {
    fn from_grid_inner(inner: GridInner) -> Result<::noobase::axis::Grid<f64>, &'static str> {
        match inner {
            GridInner::F64(grid) => Ok(grid),
            GridInner::F32(_) => Err("float32"),
        }
    }
}

/// Take the channel-`T` core grid out of a [`PyGrid`], or the canonical
/// cross-input dtype-mismatch `ValueError` (the found dtype on the
/// left, `T` on the right) contextualised by `mismatch_context`.
pub(crate) fn grid_channel<T: Scalar + GridChannel>(
    grid: PyGrid,
    mismatch_context: &str,
) -> PyResult<CoreGrid<T>> {
    T::from_grid_inner(grid.inner)
        .map_err(|found| dtype_mismatch_error(found, T::DTYPE_NAME, mismatch_context))
}

pub(crate) fn build_grid_from_any(
    values: &Bound<'_, PyAny>,
    spacing: Spacing,
    kind: GridKind,
) -> PyResult<GridInner> {
    if let Ok(array_f64) = values.extract::<PyReadonlyArray1<'_, f64>>() {
        let owned: Array1<f64> = array_f64.as_array().to_owned();
        let grid = CoreGrid::<f64>::new(owned, spacing, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(GridInner::F64(grid))
    } else if let Ok(array_f32) = values.extract::<PyReadonlyArray1<'_, f32>>() {
        let owned: Array1<f32> = array_f32.as_array().to_owned();
        let grid = CoreGrid::<f32>::new(owned, spacing, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(GridInner::F32(grid))
    } else {
        Err(PyValueError::new_err(
            "values must be a 1-D numpy array of dtype float32 or float64",
        ))
    }
}

/// Coerce a Python value into a `PyGrid`, matching the dtype of a paired
/// array (e.g. transmission_values for transmission_grid). If the value is
/// already a Grid, only the dtype is checked. If the value is an ndarray, it
/// is interpreted as bin centers when its length equals `paired_array_length`
/// and as bin edges when its length equals `paired_array_length + 1`.
pub(crate) fn coerce_to_grid(
    py_any: &Bound<'_, PyAny>,
    paired_array_length: usize,
    paired_array_dtype_is_f32: bool,
    role_name: &str,
) -> PyResult<PyGrid> {
    if let Ok(grid) = py_any.extract::<PyGrid>() {
        let grid_is_f32 = matches!(grid.inner, GridInner::F32(_));
        if grid_is_f32 != paired_array_dtype_is_f32 {
            let grid_dtype = if grid_is_f32 { "float32" } else { "float64" };
            let paired_dtype = if paired_array_dtype_is_f32 {
                "float32"
            } else {
                "float64"
            };
            return Err(dtype_mismatch_error(
                grid_dtype,
                paired_dtype,
                &format!("{role_name} grid vs paired array"),
            ));
        }
        return Ok(grid);
    }

    let centers_kind = GridKind::Centers;
    let edges_kind = GridKind::Edges;
    if paired_array_dtype_is_f32 {
        let array = py_any.extract::<PyReadonlyArray1<'_, f32>>().map_err(|_| {
            dtype_mismatch_error(
                "float32",
                "other",
                &format!("{role_name} grid array vs paired array"),
            )
        })?;
        let view = array.as_array();
        let kind = if view.len() == paired_array_length {
            centers_kind
        } else if view.len() == paired_array_length + 1 {
            edges_kind
        } else {
            return Err(PyValueError::new_err(format!(
                "{role_name} grid array length {} does not match paired array length {} (centers) or {} (edges)",
                view.len(),
                paired_array_length,
                paired_array_length + 1
            )));
        };
        let owned: Array1<f32> = view.to_owned();
        let grid = CoreGrid::<f32>::new(owned, Spacing::Linear, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyGrid::from_inner(GridInner::F32(grid)))
    } else {
        let array = py_any.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
            dtype_mismatch_error(
                "float64",
                "other",
                &format!("{role_name} grid array vs paired array"),
            )
        })?;
        let view = array.as_array();
        let kind = if view.len() == paired_array_length {
            centers_kind
        } else if view.len() == paired_array_length + 1 {
            edges_kind
        } else {
            return Err(PyValueError::new_err(format!(
                "{role_name} grid array length {} does not match paired array length {} (centers) or {} (edges)",
                view.len(),
                paired_array_length,
                paired_array_length + 1
            )));
        };
        let owned: Array1<f64> = view.to_owned();
        let grid = CoreGrid::<f64>::new(owned, Spacing::Linear, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyGrid::from_inner(GridInner::F64(grid)))
    }
}

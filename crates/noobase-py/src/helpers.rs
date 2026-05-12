use ::noobase::{Grid as CoreGrid, GridKind, Spacing};
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid::GridInner;

pub(crate) fn parse_spacing(spacing: &str) -> PyResult<Spacing> {
    match spacing {
        "linear" => Ok(Spacing::Linear),
        "log" => Ok(Spacing::Log),
        other => Err(PyValueError::new_err(format!(
            "invalid spacing {other:?}; expected one of \"linear\", \"log\""
        ))),
    }
}

pub(crate) fn parse_kind(kind: &str) -> PyResult<GridKind> {
    match kind {
        "centers" => Ok(GridKind::Centers),
        "edges" => Ok(GridKind::Edges),
        other => Err(PyValueError::new_err(format!(
            "invalid kind {other:?}; expected one of \"centers\", \"edges\""
        ))),
    }
}

pub(crate) fn spacing_to_str(spacing: Spacing) -> &'static str {
    match spacing {
        Spacing::Linear => "linear",
        Spacing::Log => "log",
    }
}

pub(crate) fn kind_to_str(kind: GridKind) -> &'static str {
    match kind {
        GridKind::Centers => "centers",
        GridKind::Edges => "edges",
    }
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

pub(crate) fn dtype_mismatch_error(
    left: &'static str,
    right: &'static str,
    context: &str,
) -> PyErr {
    PyValueError::new_err(format!(
        "dtype mismatch: {context} ({left} vs {right}); align dtypes explicitly"
    ))
}

pub(crate) fn grid_dtype_name(inner: &GridInner) -> &'static str {
    match inner {
        GridInner::F32(_) => "float32",
        GridInner::F64(_) => "float64",
    }
}

pub(crate) fn is_float32_dtype(
    py: Python<'_>,
    dtype_arg: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let Some(value) = dtype_arg else {
        return Ok(false);
    };
    if value.is_none() {
        return Ok(false);
    }
    let numpy_module = py.import("numpy")?;
    let numpy_dtype_factory = numpy_module.getattr("dtype")?;
    let resolved = numpy_dtype_factory.call1((value,))?;
    let resolved_name: String = resolved.getattr("name")?.extract()?;
    match resolved_name.as_str() {
        "float32" => Ok(true),
        "float64" => Ok(false),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype {other:?}; expected float32 or float64"
        ))),
    }
}

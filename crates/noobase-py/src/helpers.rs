use ::noobase::photometry::PhotometryConvention;
use ::noobase::{Grid as CoreGrid, GridKind, Spacing};
use ndarray::Array1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid::{GridInner, PyGrid};

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

pub(crate) fn parse_convention(value: &str) -> PyResult<PhotometryConvention> {
    match value {
        "photon_counting" => Ok(PhotometryConvention::PhotonCounting),
        "energy_weighted" => Ok(PhotometryConvention::EnergyWeighted),
        other => Err(PyValueError::new_err(format!(
            "invalid convention {other:?}; expected one of \"photon_counting\", \"energy_weighted\""
        ))),
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

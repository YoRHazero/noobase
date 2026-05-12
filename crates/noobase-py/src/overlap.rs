use ::noobase::bins::overlap as core_overlap;
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid::{GridInner, PyGrid};

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

#[pyfunction]
#[pyo3(name = "rebin")]
fn overlap_rebin<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    source_values: &Bound<'py, PyAny>,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let array = source_values
                .extract::<PyReadonlyArray1<'_, f64>>()
                .map_err(|_| {
                    dtype_mismatch_error("float64", "values not float64", "source grid vs values")
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_values length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f64> = core_overlap::rebin(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let array = source_values
                .extract::<PyReadonlyArray1<'_, f32>>()
                .map_err(|_| {
                    dtype_mismatch_error("float32", "values not float32", "source grid vs values")
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_values length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f32> = core_overlap::rebin(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "rebin_variance")]
fn overlap_rebin_variance<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    source_variance: &Bound<'py, PyAny>,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let array = source_variance
                .extract::<PyReadonlyArray1<'_, f64>>()
                .map_err(|_| {
                    dtype_mismatch_error(
                        "float64",
                        "variance not float64",
                        "source grid vs variance",
                    )
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_variance length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f64> = core_overlap::rebin_variance(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let array = source_variance
                .extract::<PyReadonlyArray1<'_, f32>>()
                .map_err(|_| {
                    dtype_mismatch_error(
                        "float32",
                        "variance not float32",
                        "source grid vs variance",
                    )
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_variance length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f32> = core_overlap::rebin_variance(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "coverage")]
fn overlap_coverage<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let output: Array1<f64> = core_overlap::coverage(source, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let output: Array1<f32> = core_overlap::coverage(source, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let overlap = PyModule::new(py, "overlap")?;
    overlap.add_function(wrap_pyfunction!(overlap_rebin, &overlap)?)?;
    overlap.add_function(wrap_pyfunction!(overlap_rebin_variance, &overlap)?)?;
    overlap.add_function(wrap_pyfunction!(overlap_coverage, &overlap)?)?;
    parent.add_submodule(&overlap)?;

    // Register the submodule under its dotted name so `from noobase._core.overlap
    // import ...` (used by the `noobase.overlap` python wrapper) resolves.
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.overlap", &overlap)?;

    Ok(())
}

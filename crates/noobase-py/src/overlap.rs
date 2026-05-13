use ::noobase::bins::overlap as core_overlap;
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid::{GridInner, PyGrid};
use crate::helpers::{dtype_mismatch_error, grid_dtype_name};

/// Flux-density-conserving rebin from a source grid onto a target grid.
///
/// For each target bin ``i``, the output is
/// ``out[i] = (sum_j overlap[i, j] * source_values[j]) / target_width[i]``,
/// where ``overlap[i, j]`` is the linear-space intersection width between
/// target bin ``i`` and source bin ``j`` and ``target_width[i]`` is the full
/// linear-space width of target bin ``i``. This preserves the integral
/// ``sum_i out[i] * target_width[i]`` over the region of overlap, so the
/// values behave like a density.
///
/// Target bins partially or fully outside the source range are NOT treated
/// as errors: the sum is taken over whatever overlap exists. Use
/// ``coverage`` to identify and mask such bins.
///
/// Parameters
/// ----------
/// source_grid : Grid
///     Source wavelength axis.
/// source_values : ndarray
///     Per-bin source density. Length must equal the source bin count and
///     dtype must match ``source_grid``.
/// target_grid : Grid
///     Target wavelength axis. dtype must match ``source_grid``.
///
/// Returns
/// -------
/// ndarray
///     Rebinned values, length equal to the target bin count. dtype matches
///     the inputs.
///
/// Raises
/// ------
/// ValueError
///     If dtypes mismatch across inputs, or if ``source_values`` length does
///     not match the source bin count.
#[pyfunction]
#[pyo3(name = "rebin")]
#[pyo3(text_signature = "(source_grid, source_values, target_grid)")]
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

/// Variance propagation for ``rebin`` assuming independent source bins.
///
/// For each target bin ``i``, the output is
/// ``out[i] = sum_j (overlap[i, j] / target_width[i])^2 * source_variance[j]``.
/// As with ``rebin``, partial-coverage target bins are not errors; their
/// variance is computed against the partial sum and should usually be
/// masked by the caller using ``coverage``.
///
/// Parameters
/// ----------
/// source_grid : Grid
///     Source wavelength axis.
/// source_variance : ndarray
///     Per-bin source variance (1-sigma squared). Length must equal the
///     source bin count and dtype must match ``source_grid``.
/// target_grid : Grid
///     Target wavelength axis. dtype must match ``source_grid``.
///
/// Returns
/// -------
/// ndarray
///     Propagated variance per target bin. dtype matches the inputs.
///
/// Raises
/// ------
/// ValueError
///     If dtypes mismatch across inputs, or if ``source_variance`` length
///     does not match the source bin count.
///
/// Notes
/// -----
/// Independence is an assumption, not a property of the operator: if the
/// caller's source bins are correlated (for example because the spectrum
/// was previously upsampled), the output underestimates the true variance.
#[pyfunction]
#[pyo3(name = "rebin_variance")]
#[pyo3(text_signature = "(source_grid, source_variance, target_grid)")]
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

/// Geometric coverage fraction of each target bin by the source range.
///
/// For each target bin ``i``,
/// ``out[i] = (sum_j overlap[i, j]) / target_width[i]`` lies in ``[0, 1]``.
/// A target bin completely inside the source range yields 1.0; one
/// completely outside yields 0.0; a half-covered edge bin yields 0.5. This
/// is the standard mask companion for ``rebin`` and ``rebin_variance``.
///
/// Parameters
/// ----------
/// source_grid : Grid
///     Source wavelength axis.
/// target_grid : Grid
///     Target wavelength axis. dtype must match ``source_grid``.
///
/// Returns
/// -------
/// ndarray
///     Coverage fraction per target bin, in ``[0, 1]``. dtype matches the
///     inputs.
///
/// Raises
/// ------
/// ValueError
///     If dtypes mismatch across inputs.
#[pyfunction]
#[pyo3(name = "coverage")]
#[pyo3(text_signature = "(source_grid, target_grid)")]
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

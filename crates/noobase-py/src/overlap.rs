use ::noobase::Grid as CoreGrid;
use ::noobase::bins::overlap as core_overlap;
use ndarray::Array1;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{Scalar, typed_array1, with_grid_pair};
use crate::grid::PyGrid;

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
    with_grid_pair!(&source_grid.inner, &target_grid.inner, (source, target) =>
        rebin_impl(py, source, source_values, target))
}

fn rebin_impl<'py, T: Scalar>(
    py: Python<'py>,
    source: &CoreGrid<T>,
    source_values: &Bound<'py, PyAny>,
    target: &CoreGrid<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let values = typed_array1::<T>(source_values, "values", "source grid vs values")?;
    let source_bin_count = source.to_edges().len() - 1;
    if values.len() != source_bin_count {
        return Err(PyValueError::new_err(format!(
            "source_values length {} does not match source bin count {source_bin_count}",
            values.len()
        )));
    }
    let output: Array1<T> = core_overlap::rebin(source, values.view(), target);
    Ok(output.into_pyarray(py).into_any())
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
    with_grid_pair!(&source_grid.inner, &target_grid.inner, (source, target) =>
        rebin_variance_impl(py, source, source_variance, target))
}

fn rebin_variance_impl<'py, T: Scalar>(
    py: Python<'py>,
    source: &CoreGrid<T>,
    source_variance: &Bound<'py, PyAny>,
    target: &CoreGrid<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let variance = typed_array1::<T>(source_variance, "variance", "source grid vs variance")?;
    let source_bin_count = source.to_edges().len() - 1;
    if variance.len() != source_bin_count {
        return Err(PyValueError::new_err(format!(
            "source_variance length {} does not match source bin count {source_bin_count}",
            variance.len()
        )));
    }
    let output: Array1<T> = core_overlap::rebin_variance(source, variance.view(), target);
    Ok(output.into_pyarray(py).into_any())
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
    with_grid_pair!(&source_grid.inner, &target_grid.inner, (source, target) =>
        coverage_impl(py, source, target))
}

fn coverage_impl<'py, T: Scalar>(
    py: Python<'py>,
    source: &CoreGrid<T>,
    target: &CoreGrid<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let output: Array1<T> = core_overlap::coverage(source, target);
    Ok(output.into_pyarray(py).into_any())
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let overlap = PyModule::new(py, "noobase._core.overlap")?;
    overlap.setattr("__package__", "noobase._core")?;
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

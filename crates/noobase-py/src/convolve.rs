//! `noobase._core.convolve` — bare correlation kernels and the
//! NaN-as-missing renormalized variants, plus the Gaussian kernel
//! constructor.
//!
//! Mirrors the Rust core `noobase::convolve` module: a shared utility
//! layer used by both `image::convolve` and `spectroscopy::lsf`,
//! exposed in its own right for callers who want to compose kernels
//! and convolutions directly in Python.

use ::noobase::convolve::{
    self as core_convolve, Boundary, conv_axis, conv_axis_renorm, conv1d, conv2d, conv2d_renorm,
};
use ndarray::{Array1, Array2, Axis};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::convert::{
    Scalar, dispatch_array, is_float32_dtype, parse_axis, parse_boundary, parse_gaussian_sampling,
    parse_normalization, required_typed_array1, required_typed_array2,
};

const DEFAULT_TRUNCATE: f64 = 4.0;

/// Build a 1-D Gaussian correlation kernel of odd length
/// ``2 * ceil(truncate * sigma) + 1``, centered at index ``len // 2``.
///
/// Parameters
/// ----------
/// sigma : float
///     Gaussian sigma in pixels. Must be positive.
/// truncate : float, optional
///     Half-support in units of ``sigma``. Must be positive. Default
///     ``4.0``.
/// sampling : {"erf_integrated", "point_sampled"}, optional
///     ``erf_integrated`` (default) integrates the unit Gaussian over
///     each tap bin (accurate for narrow ``sigma``);
///     ``point_sampled`` evaluates the unnormalized Gaussian at integer
///     offsets (biased for narrow ``sigma``).
/// normalization : {"sum", "l2", "none"}, optional
///     Kernel normalization. ``sum`` (default) is flux-conserving;
///     ``l2`` is matched-filter S/N optimal; ``none`` leaves the kernel
///     unscaled.
/// dtype : numpy dtype, optional
///     Either ``numpy.float32`` or ``numpy.float64``. Default
///     ``numpy.float64``.
///
/// Returns
/// -------
/// ndarray
///     1-D Gaussian kernel of odd length.
///
/// Raises
/// ------
/// ValueError
///     If ``sigma`` or ``truncate`` is not positive, if ``sampling`` /
///     ``normalization`` is invalid, or if ``dtype`` is not
///     ``float32`` / ``float64``.
#[pyfunction]
#[pyo3(name = "gaussian1d")]
#[pyo3(signature = (sigma, *, truncate=DEFAULT_TRUNCATE, sampling="erf_integrated", normalization="sum", dtype=None))]
#[pyo3(
    text_signature = "(sigma, *, truncate=4.0, sampling=\"erf_integrated\", normalization=\"sum\", dtype=None)"
)]
fn gaussian1d_function<'py>(
    py: Python<'py>,
    sigma: f64,
    truncate: f64,
    sampling: &str,
    normalization: &str,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "sigma must be positive, got {sigma}"
        )));
    }
    if !truncate.is_finite() || truncate <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "truncate must be positive, got {truncate}"
        )));
    }
    let sampling_enum = parse_gaussian_sampling(sampling)?;
    let normalization_enum = parse_normalization(normalization)?;
    let use_f32 = is_float32_dtype(py, dtype)?;
    if use_f32 {
        let kernel: Array1<f32> =
            core_convolve::gaussian1d(sigma, truncate, sampling_enum, normalization_enum);
        Ok(kernel.into_pyarray(py).into_any())
    } else {
        let kernel: Array1<f64> =
            core_convolve::gaussian1d(sigma, truncate, sampling_enum, normalization_enum);
        Ok(kernel.into_pyarray(py).into_any())
    }
}

/// 1-D correlation of ``signal`` with ``kernel`` (the kernel is *not*
/// flipped — for a symmetric kernel this coincides with convolution).
/// NaN-naive: a non-finite tap propagates into its output element.
///
/// Parameters
/// ----------
/// signal : ndarray
///     1-D signal, dtype ``float32`` or ``float64``.
/// kernel : ndarray
///     1-D kernel, same dtype as ``signal``. The center tap is
///     ``len(kernel) // 2``.
/// boundary : {"zero", "reflect", "nearest"}, optional
///     Out-of-bounds tap handling. Default ``"zero"``.
///
/// Returns
/// -------
/// ndarray
///     Same shape and dtype as ``signal``.
///
/// Raises
/// ------
/// ValueError
///     If ``signal`` is not a 1-D float32/float64 array, ``kernel`` is
///     empty, ``kernel`` dtype does not match ``signal``, or
///     ``boundary`` is invalid.
#[pyfunction]
#[pyo3(name = "conv1d")]
#[pyo3(signature = (signal, kernel, *, boundary="zero"))]
#[pyo3(text_signature = "(signal, kernel, *, boundary=\"zero\")")]
fn conv1d_function<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
    kernel: &Bound<'py, PyAny>,
    boundary: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let boundary_enum = parse_boundary(boundary)?;
    dispatch_array!(signal, 1, "signal", conv1d_impl, py, kernel, boundary_enum)
}

fn conv1d_impl<'py, T: Scalar>(
    signal: Array1<T>,
    py: Python<'py>,
    kernel: &Bound<'py, PyAny>,
    boundary: Boundary,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel_array = required_typed_array1::<T>(kernel, "signal vs kernel")?;
    if kernel_array.is_empty() {
        return Err(PyValueError::new_err("kernel must be non-empty"));
    }
    let output = conv1d::<T>(signal.view(), kernel_array.view(), boundary);
    Ok(output.into_pyarray(py).into_any())
}

/// Apply [`conv1d`] independently along ``axis`` of a 2-D image.
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``.
/// kernel : ndarray
///     1-D kernel, same dtype as ``image``.
/// axis : int, optional
///     ``0`` correlates down each column, ``1`` along each row. Default
///     ``0``.
/// boundary : {"zero", "reflect", "nearest"}, optional
///     Out-of-bounds tap handling. Default ``"zero"``.
///
/// Returns
/// -------
/// ndarray
///     Same shape and dtype as ``image``.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32/float64 array, ``kernel`` is
///     empty or has a mismatched dtype, ``axis`` is not 0 or 1, or
///     ``boundary`` is invalid.
#[pyfunction]
#[pyo3(name = "conv_axis")]
#[pyo3(signature = (image, kernel, *, axis=0, boundary="zero"))]
#[pyo3(text_signature = "(image, kernel, *, axis=0, boundary=\"zero\")")]
fn conv_axis_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    kernel: &Bound<'py, PyAny>,
    axis: usize,
    boundary: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let axis_enum = parse_axis(axis)?;
    let boundary_enum = parse_boundary(boundary)?;
    dispatch_array!(
        image,
        2,
        "image",
        conv_axis_impl,
        py,
        kernel,
        axis_enum,
        boundary_enum
    )
}

fn conv_axis_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    kernel: &Bound<'py, PyAny>,
    axis: Axis,
    boundary: Boundary,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel_array = required_typed_array1::<T>(kernel, "image vs kernel")?;
    if kernel_array.is_empty() {
        return Err(PyValueError::new_err("kernel must be non-empty"));
    }
    let output = conv_axis::<T>(image.view(), kernel_array.view(), axis, boundary);
    Ok(output.into_pyarray(py).into_any())
}

/// 2-D correlation of ``image`` with ``kernel`` (the kernel is *not*
/// flipped). NaN-naive.
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``.
/// kernel : ndarray
///     2-D kernel, same dtype as ``image``. Centered at
///     ``(nrows // 2, ncols // 2)``.
/// boundary : {"zero", "reflect", "nearest"}, optional
///     Out-of-bounds tap handling. Default ``"zero"``.
///
/// Returns
/// -------
/// ndarray
///     Same shape and dtype as ``image``.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32/float64 array, ``kernel`` is
///     empty or has a mismatched dtype, or ``boundary`` is invalid.
#[pyfunction]
#[pyo3(name = "conv2d")]
#[pyo3(signature = (image, kernel, *, boundary="zero"))]
#[pyo3(text_signature = "(image, kernel, *, boundary=\"zero\")")]
fn conv2d_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    kernel: &Bound<'py, PyAny>,
    boundary: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let boundary_enum = parse_boundary(boundary)?;
    dispatch_array!(image, 2, "image", conv2d_impl, py, kernel, boundary_enum)
}

fn conv2d_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    kernel: &Bound<'py, PyAny>,
    boundary: Boundary,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel_array = required_typed_array2::<T>(kernel, "image vs kernel")?;
    if kernel_array.nrows() == 0 || kernel_array.ncols() == 0 {
        return Err(PyValueError::new_err("kernel must be non-empty"));
    }
    let output = conv2d::<T>(image.view(), kernel_array.view(), boundary);
    Ok(output.into_pyarray(py).into_any())
}

/// 2-D renormalized correlation (NaN-as-missing). Returns ``(value,
/// weight)`` where ``value = N / D`` over the finite taps and
/// ``weight = D`` is the kernel-weighted valid-tap sum. Out-of-bounds is
/// always missing; no ``boundary`` switch.
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``. NaN / +-inf are
///     treated as missing.
/// kernel : ndarray
///     2-D kernel, same dtype as ``image``. Should be non-negative for
///     the renorm semantics to make sense.
///
/// Returns
/// -------
/// tuple of (ndarray, ndarray)
///     ``(value, weight)``, both same shape and dtype as ``image``.
///     ``value`` is ``NaN`` where no valid tap contributed.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32/float64 array, or ``kernel`` is
///     empty or has a mismatched dtype.
#[pyfunction]
#[pyo3(name = "conv2d_renorm")]
#[pyo3(signature = (image, kernel))]
#[pyo3(text_signature = "(image, kernel)")]
fn conv2d_renorm_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    kernel: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_array!(image, 2, "image", conv2d_renorm_impl, py, kernel)
}

fn conv2d_renorm_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    kernel: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel_array = required_typed_array2::<T>(kernel, "image vs kernel")?;
    if kernel_array.nrows() == 0 || kernel_array.ncols() == 0 {
        return Err(PyValueError::new_err("kernel must be non-empty"));
    }
    let (value, weight) = conv2d_renorm::<T>(image.view(), kernel_array.view());
    let tuple = PyTuple::new(
        py,
        [
            value.into_pyarray(py).into_any(),
            weight.into_pyarray(py).into_any(),
        ],
    )?;
    Ok(tuple.into_any())
}

/// 1-D renormalized correlation along ``axis`` of a 2-D image
/// (NaN-as-missing). Returns ``(value, weight)`` with the same
/// semantics as [`conv2d_renorm`].
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``. NaN / +-inf are
///     treated as missing.
/// kernel : ndarray
///     1-D kernel, same dtype as ``image``. Should be non-negative.
/// axis : int, optional
///     ``0`` correlates down each column, ``1`` along each row. Default
///     ``0``.
///
/// Returns
/// -------
/// tuple of (ndarray, ndarray)
///     ``(value, weight)``, both same shape and dtype as ``image``.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32/float64 array, ``kernel`` is
///     empty or has a mismatched dtype, or ``axis`` is not 0 or 1.
#[pyfunction]
#[pyo3(name = "conv_axis_renorm")]
#[pyo3(signature = (image, kernel, *, axis=0))]
#[pyo3(text_signature = "(image, kernel, *, axis=0)")]
fn conv_axis_renorm_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    kernel: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let axis_enum = parse_axis(axis)?;
    dispatch_array!(
        image,
        2,
        "image",
        conv_axis_renorm_impl,
        py,
        kernel,
        axis_enum
    )
}

fn conv_axis_renorm_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    kernel: &Bound<'py, PyAny>,
    axis: Axis,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel_array = required_typed_array1::<T>(kernel, "image vs kernel")?;
    if kernel_array.is_empty() {
        return Err(PyValueError::new_err("kernel must be non-empty"));
    }
    let (value, weight) = conv_axis_renorm::<T>(image.view(), kernel_array.view(), axis);
    let tuple = PyTuple::new(
        py,
        [
            value.into_pyarray(py).into_any(),
            weight.into_pyarray(py).into_any(),
        ],
    )?;
    Ok(tuple.into_any())
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let convolve = PyModule::new(py, "noobase._core.convolve")?;
    convolve.setattr("__package__", "noobase._core")?;
    convolve.add_function(wrap_pyfunction!(gaussian1d_function, &convolve)?)?;
    convolve.add_function(wrap_pyfunction!(conv1d_function, &convolve)?)?;
    convolve.add_function(wrap_pyfunction!(conv_axis_function, &convolve)?)?;
    convolve.add_function(wrap_pyfunction!(conv2d_function, &convolve)?)?;
    convolve.add_function(wrap_pyfunction!(conv2d_renorm_function, &convolve)?)?;
    convolve.add_function(wrap_pyfunction!(conv_axis_renorm_function, &convolve)?)?;
    parent.add_submodule(&convolve)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.convolve", &convolve)?;
    Ok(())
}

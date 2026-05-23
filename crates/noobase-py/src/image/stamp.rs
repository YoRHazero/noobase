//! Thin PyO3 binding for the `image::stamp` leaf -- the
//! `build_stamp` / `StampResult` pair.
//!
//! Mirrors the Rust core path: lives at the `noobase._core.image`
//! level (alongside `convolve_psf` / `reproject_exact`), not inside
//! the nested `image.psf` submodule.

use ::noobase::image::stamp as core_stamp;
use ::noobase::image::stamp::StampResult;
use ndarray::{Array2, ArrayView2};
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::convert::{
    Scalar, dispatch_array, optional_bool_array2, optional_companion_array2, to_value_error,
};

/// Result of ``build_stamp``: the integer-window stamp plus the recorded
/// sub-pixel offset. Constructed by ``build_stamp``; not instantiable
/// directly.
#[pyclass(name = "StampResult", module = "noobase._core.image", frozen)]
pub struct PyStampResult {
    inner: StampResult,
}

#[pymethods]
impl PyStampResult {
    /// The extracted stamp of shape ``(stamp_size, stamp_size)``, dtype
    /// ``float64``. Carries the original cutout values; no background is
    /// removed (the centroid background subtraction is internal only).
    #[getter]
    fn stamp<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.stamp.to_pyarray(py).into_any()
    }

    /// The windowed 1-sigma error of shape ``(stamp_size, stamp_size)``,
    /// dtype ``float64``, or ``None`` when no ``error`` was supplied.
    #[getter]
    fn error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.inner
            .error
            .as_ref()
            .map(|array| array.to_pyarray(py).into_any())
    }

    /// Per-pixel validity of shape ``(stamp_size, stamp_size)``, dtype
    /// ``bool``. ``True`` means valid (positive polarity -- the opposite
    /// of the input ``mask`` whose ``True`` means invalid).
    #[getter]
    fn valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.valid.to_pyarray(py).into_any()
    }

    /// The recorded sub-pixel offset ``(delta_row, delta_col) =
    /// centroid - round(centroid)``; each component in ``[-0.5, 0.5)``.
    #[getter]
    fn delta(&self) -> (f64, f64) {
        self.inner.delta
    }

    /// ``(row, col)`` of the window's top-left corner in cutout-local
    /// indices. Add the caller's own cutout origin to map back to the
    /// source frame.
    #[getter]
    fn origin(&self) -> (i64, i64) {
        self.inner.origin
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = (self.inner.stamp.shape()[0], self.inner.stamp.shape()[1]);
        let (delta_row, delta_col) = self.inner.delta;
        format!(
            "StampResult(stamp=({rows}, {cols}), delta=({delta_row}, {delta_col}), origin={:?})",
            self.inner.origin
        )
    }
}

/// Locate an init-quality centroid in a rough cutout, pick an
/// integer-aligned odd square window, and slice a fixed-size stamp.
///
/// Single-image leaf for scheme C: the caller loops over N stars, cuts a
/// rough cutout per star (via its own WCS / distortion), calls
/// ``build_stamp`` on each, filters out the ``None`` returns, and stacks
/// the survivors for ``build_epsf`` / ``build_extended_psf``. The
/// sub-pixel offset is *recorded only*, never applied (a sub-pixel shift
/// would correlate the noise).
///
/// Parameters
/// ----------
/// cutout : ndarray
///     Rough cutout of any shape ``(rows, cols)``, dtype ``float32`` or
///     ``float64``. Determines the dtype channel; ``error`` must match.
///     Non-finite values are excluded from the centroid and marked
///     invalid.
/// stamp_size : int
///     Edge length of the square output stamp. Must be odd.
/// error : ndarray, optional
///     1-sigma error, same shape and dtype as ``cutout``. Default
///     ``None`` (error-level equal weighting downstream).
/// mask : ndarray of bool, optional
///     ``True`` marks an *invalid* (excluded) pixel -- the noobase-wide
///     polarity. Same shape as ``cutout``. Default ``None``.
/// weight_fwhm : float, optional
///     FWHM (pixels) of the Gaussian centroid window. Default ``3.0``.
/// max_iter : int, optional
///     Centroid iteration cap. Default ``10``.
/// tol : float, optional
///     Centroid convergence tolerance (pixels). Default ``1e-3``.
///
/// Returns
/// -------
/// StampResult or None
///     ``None`` when no usable stamp can be produced (the odd window
///     cannot fit fully inside the cutout, or there are too few valid
///     pixels to centroid) -- an algorithmic skip, not an error. The
///     caller filters these out before stacking.
///
/// Raises
/// ------
/// ValueError
///     If ``stamp_size`` is even, if ``stamp_size`` exceeds a cutout
///     dimension, if ``error`` / ``mask`` shape does not equal the
///     cutout shape, or if an array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "build_stamp")]
#[pyo3(signature = (cutout, stamp_size, *, error=None, mask=None, weight_fwhm=core_stamp::DEFAULT_WEIGHT_FWHM, max_iter=core_stamp::DEFAULT_CENTROID_MAX_ITER, tol=core_stamp::DEFAULT_CENTROID_TOL))]
#[pyo3(
    text_signature = "(cutout, stamp_size, *, error=None, mask=None, weight_fwhm=3.0, max_iter=10, tol=1e-3)"
)]
#[allow(clippy::too_many_arguments)]
fn build_stamp_function(
    cutout: &Bound<'_, PyAny>,
    stamp_size: usize,
    error: Option<&Bound<'_, PyAny>>,
    mask: Option<&Bound<'_, PyAny>>,
    weight_fwhm: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Option<PyStampResult>> {
    let mask_owned = optional_bool_array2(mask, "mask")?;
    let mask_view = mask_owned.as_ref().map(|array| array.view());
    dispatch_array!(
        cutout,
        2,
        "cutout",
        build_stamp_impl,
        stamp_size,
        error,
        mask_view,
        weight_fwhm,
        max_iter,
        tol
    )
}

#[allow(clippy::too_many_arguments)]
fn build_stamp_impl<T: Scalar>(
    cutout: Array2<T>,
    stamp_size: usize,
    error: Option<&Bound<'_, PyAny>>,
    mask_view: Option<ArrayView2<'_, bool>>,
    weight_fwhm: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Option<PyStampResult>> {
    let error_owned = optional_companion_array2::<T>(error, "error")?;
    let error_view = error_owned.as_ref().map(|array| array.view());
    let result = core_stamp::build_stamp::<T>(
        cutout.view(),
        stamp_size,
        error_view,
        mask_view,
        weight_fwhm,
        max_iter,
        tol,
    )
    .map_err(to_value_error)?;
    Ok(result.map(|inner| PyStampResult { inner }))
}

pub(crate) fn register_into(image: &Bound<'_, PyModule>) -> PyResult<()> {
    image.add_function(wrap_pyfunction!(build_stamp_function, image)?)?;
    image.add_class::<PyStampResult>()?;
    Ok(())
}

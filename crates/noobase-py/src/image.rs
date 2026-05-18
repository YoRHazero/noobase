use ::noobase::convolve::{Boundary, Normalization};
use ::noobase::image as core_image;
use ndarray::{Array2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::convert::{
    Scalar, dispatch_array, parse_boundary, parse_normalization, required_typed_array2,
    to_value_error,
};

/// Surface-brightness-conserving exact reprojection of a 2-D image.
///
/// The caller supplies an input image on its own pixel grid plus a corner
/// field that gives, for every node of the *output* pixel grid, the
/// corresponding location in the input image's pixel coordinates. Each
/// output pixel is then computed by polygon-clipping the input pixels
/// that fall under the quadrilateral defined by the four output corners
/// and taking the area-weighted mean of the contributing input values.
///
/// This package intentionally does NOT compute the corner field itself:
/// WCS handling belongs in astropy.wcs / gwcs / equivalent. The caller is
/// expected to map output-pixel corners through the two WCSs (output →
/// sky → input) once and hand the resulting array to this function. The
/// reprojection then runs entirely in input-pixel coordinates and does
/// not require any spherical-geometry primitive.
///
/// Parameters
/// ----------
/// image_in : ndarray
///     Input image of shape ``(H_in, W_in)``, dtype ``float32`` or
///     ``float64``. NaN values are treated as masked input pixels: they
///     are excluded from the numerator and from the valid weight, but
///     they still count toward the geometric ``footprint``.
/// pixel_corners : ndarray
///     Corner field of shape ``(H_out + 1, W_out + 1, 2)``, dtype
///     ``float64``. The last dimension stores ``[x_in, y_in]`` — the
///     location of each output-pixel corner in the input image's
///     continuous pixel coordinates, using the astropy convention that
///     integer ``(x, y)`` is the *center* of pixel ``(row=y, column=x)``.
///     Adjacent output pixels share corners by construction, which makes
///     the partition watertight. NaN corners propagate to ``(NaN, 0, 0)``
///     in the output for any output pixel that touches them.
///
/// Returns
/// -------
/// tuple of (ndarray, ndarray, ndarray)
///     ``(image, footprint, weight)``, all of shape ``(H_out, W_out)``
///     and dtype ``float64``.
///
///     - ``image`` is the surface-brightness-conserving reprojection:
///       ``sum(A_ij * image_in[i, j]) / sum(A_ij_valid)`` over the input
///       pixels overlapping each output pixel, where ``A_ij`` is the
///       polygon-intersection area in input-pixel units and the
///       ``_valid`` sum excludes NaN input pixels. Output pixels with no
///       valid contribution are ``NaN``.
///     - ``footprint`` is the pure geometric overlap fraction in
///       ``[0, 1]``: how much of the output pixel falls inside the input
///       image's pixel grid, independent of whether the covered pixels
///       are NaN. Useful to distinguish "no data because we are outside
///       the input footprint" from "no data because the inputs were
///       masked".
///     - ``weight`` is the same numerator restricted to non-NaN input
///       pixels, also in ``[0, 1]``. Invariant: ``weight <= footprint``;
///       they are equal when no input pixel inside the overlap was NaN.
///       The ratio ``weight / footprint`` (where ``footprint > 0``) is
///       the fraction of the covered region that was free of NaN.
///
/// Raises
/// ------
/// ValueError
///     If ``image_in`` is not a 2-D float32 / float64 array, if
///     ``pixel_corners`` is not a float64 array with last dimension 2 and
///     both grid dimensions at least 2.
///
/// Notes
/// -----
/// Spherical curvature is not corrected: this is a planar polygon clip
/// in input-pixel space. The residual error is the curvature of the
/// input projection over a single output pixel, which is negligible for
/// any reasonable astronomical image. Callers who need an explicit
/// spherical treatment should pre-correct ``pixel_corners`` accordingly.
#[pyfunction]
#[pyo3(name = "reproject_exact")]
#[pyo3(signature = (image_in, pixel_corners))]
#[pyo3(text_signature = "(image_in, pixel_corners)")]
fn reproject_exact_function<'py>(
    py: Python<'py>,
    image_in: &Bound<'py, PyAny>,
    pixel_corners: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let corners_readonly = pixel_corners
        .extract::<PyReadonlyArray3<'_, f64>>()
        .map_err(|_| {
            PyValueError::new_err(
                "pixel_corners must be a 3-D numpy array of dtype float64 with shape (H_out + 1, W_out + 1, 2)",
            )
        })?;
    let corners_owned = corners_readonly.as_array().to_owned();

    dispatch_array!(
        image_in,
        2,
        "image_in",
        reproject_exact_impl,
        py,
        corners_owned.view()
    )
}

fn reproject_exact_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    corners: ArrayView3<'_, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let output = core_image::reproject_exact::<T>(image.view(), corners).map_err(to_value_error)?;
    let tuple = PyTuple::new(
        py,
        [
            output.image.into_pyarray(py).into_any(),
            output.footprint.into_pyarray(py).into_any(),
            output.weight.into_pyarray(py).into_any(),
        ],
    )?;
    Ok(tuple.into_any())
}

/// Convolve a 2-D image with a fixed, centered PSF (true convolution).
///
/// The PSF is flipped (both axes reversed) before correlation, so this
/// is genuine convolution: an asymmetric ePSF keeps its orientation.
/// NaN / +-inf pixels are treated as missing and excluded from both the
/// weighted sum and its normalizer, and image edges are likewise
/// missing — this NaN-as-missing renormalization is the image
/// subsystem's single convention (same as ``reproject_exact``), so there
/// is deliberately no boundary or renormalize switch.
///
/// PSF normalization is the caller's responsibility: a sum-normalized
/// PSF gives an interior identity with the renorm acting only at NaN /
/// edges (flux conserving); an unnormalized PSF yields a local weighted
/// mean instead.
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``. NaN / +-inf are
///     treated as missing.
/// psf : ndarray
///     2-D PSF with odd dimensions, same dtype as ``image``.
///
/// Returns
/// -------
/// ndarray
///     The convolved image, same shape and dtype as ``image``.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32 / float64 array, ``psf`` dtype
///     does not match ``image``, or ``psf`` does not have odd
///     dimensions.
#[pyfunction]
#[pyo3(name = "convolve_psf")]
#[pyo3(signature = (image, psf))]
#[pyo3(text_signature = "(image, psf)")]
fn convolve_psf_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    psf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_array!(image, 2, "image", convolve_psf_impl, py, psf)
}

fn convolve_psf_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    psf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let psf_array = required_typed_array2::<T>(psf, "image vs psf")?;
    if psf_array.nrows() % 2 != 1 || psf_array.ncols() % 2 != 1 {
        return Err(PyValueError::new_err(format!(
            "psf must have odd dimensions (a centered PSF); got {}x{}",
            psf_array.nrows(),
            psf_array.ncols()
        )));
    }
    let output = core_image::convolve_psf::<T>(image.view(), psf_array.view());
    Ok(output.into_pyarray(py).into_any())
}

/// Correlate every lane along ``axis`` with a 1-D Gaussian — the grism
/// emission-line matched filter.
///
/// The kernel is built with the requested ``normalization`` (``"l2"``
/// makes the peak response the matched-filter S/N-optimal statistic).
/// This is a correlation; a Gaussian is symmetric so it coincides with
/// convolution. ``renormalize=False`` (the matched filter's natural
/// mode) uses ``boundary`` for out-of-bounds taps; ``renormalize=True``
/// uses NaN-as-missing renormalization and ``boundary`` is ignored.
///
/// Parameters
/// ----------
/// image : ndarray
///     2-D image, dtype ``float32`` or ``float64``.
/// sigma : float
///     Gaussian sigma in pixels along ``axis``. Must be positive.
/// axis : int, optional
///     ``0`` correlates down each column, ``1`` along each row. Default
///     ``0``.
/// normalization : {"sum", "l2", "none"}, optional
///     Kernel normalization. Default ``"l2"`` (matched-filter optimal).
/// boundary : {"zero", "reflect", "nearest"}, optional
///     Out-of-bounds handling, used only when ``renormalize=False``.
///     Default ``"zero"``.
/// renormalize : bool, optional
///     When ``True``, use NaN-as-missing renormalization (out-of-bounds
///     is missing and ``boundary`` is ignored). Default ``False``.
///
/// Returns
/// -------
/// ndarray
///     The filtered image, same shape and dtype as ``image``.
///
/// Raises
/// ------
/// ValueError
///     If ``image`` is not a 2-D float32 / float64 array, ``sigma`` is
///     not positive, ``axis`` is not 0 or 1, or ``normalization`` /
///     ``boundary`` is invalid.
#[pyfunction]
#[pyo3(name = "convolve_gaussian_axis")]
#[pyo3(signature = (image, *, sigma, axis=0, normalization="l2", boundary="zero", renormalize=false))]
#[pyo3(
    text_signature = "(image, *, sigma, axis=0, normalization=\"l2\", boundary=\"zero\", renormalize=False)"
)]
fn convolve_gaussian_axis_function<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    sigma: f64,
    axis: usize,
    normalization: &str,
    boundary: &str,
    renormalize: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "sigma must be positive, got {sigma}"
        )));
    }
    let axis_enum = match axis {
        0 => Axis(0),
        1 => Axis(1),
        other => {
            return Err(PyValueError::new_err(format!(
                "axis must be 0 or 1, got {other}"
            )));
        }
    };
    let normalization_enum = parse_normalization(normalization)?;
    let boundary_enum = parse_boundary(boundary)?;
    dispatch_array!(
        image,
        2,
        "image",
        convolve_gaussian_axis_impl,
        py,
        sigma,
        axis_enum,
        normalization_enum,
        boundary_enum,
        renormalize
    )
}

fn convolve_gaussian_axis_impl<'py, T: Scalar>(
    image: Array2<T>,
    py: Python<'py>,
    sigma: f64,
    axis: Axis,
    normalization: Normalization,
    boundary: Boundary,
    renormalize: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let output = core_image::convolve_gaussian_axis::<T>(
        image.view(),
        sigma,
        axis,
        normalization,
        boundary,
        renormalize,
    );
    Ok(output.into_pyarray(py).into_any())
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let image = PyModule::new(py, "image")?;
    image.add_function(wrap_pyfunction!(reproject_exact_function, &image)?)?;
    image.add_function(wrap_pyfunction!(convolve_psf_function, &image)?)?;
    image.add_function(wrap_pyfunction!(convolve_gaussian_axis_function, &image)?)?;
    parent.add_submodule(&image)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.image", &image)?;
    // Phase 8: build_stamp/StampResult (image::stamp) at the image
    // level + the nested noobase._core.image.psf submodule (image::psf),
    // mirroring the Rust core module paths.
    crate::psf::register(py, &image)?;
    Ok(())
}

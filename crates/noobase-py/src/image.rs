use ::noobase::image as core_image;
use ::noobase::image::ReprojectError;
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn map_reproject_error(err: ReprojectError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

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
    let corners_owned: Array3<f64> = corners_readonly.as_array().to_owned();

    let output = if let Ok(image_readonly) = image_in.extract::<PyReadonlyArray2<'_, f64>>() {
        let image_owned: Array2<f64> = image_readonly.as_array().to_owned();
        core_image::reproject_exact::<f64>(image_owned.view(), corners_owned.view())
            .map_err(map_reproject_error)?
    } else if let Ok(image_readonly) = image_in.extract::<PyReadonlyArray2<'_, f32>>() {
        let image_owned: Array2<f32> = image_readonly.as_array().to_owned();
        core_image::reproject_exact::<f32>(image_owned.view(), corners_owned.view())
            .map_err(map_reproject_error)?
    } else {
        return Err(PyValueError::new_err(
            "image_in must be a 2-D numpy array of dtype float32 or float64",
        ));
    };

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

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let image = PyModule::new(py, "image")?;
    image.add_function(wrap_pyfunction!(reproject_exact_function, &image)?)?;
    parent.add_submodule(&image)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.image", &image)?;
    // Phase 8: build_stamp/StampResult (image::stamp) at the image
    // level + the nested noobase._core.image.psf submodule (image::psf),
    // mirroring the Rust core module paths.
    crate::psf::register(py, &image)?;
    Ok(())
}

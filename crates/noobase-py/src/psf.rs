//! Thin PyO3 binding for the point-source PSF pipeline (Phase 8).
//!
//! This module carries no business logic (decision 9): every solver,
//! reducer, and stitch already lives in the Rust core. The binding only
//! translates the boundary -- numpy <-> ndarray views, flat kwargs ->
//! the core `*Params` structs, string literals -> the core enums,
//! `*Error` -> `ValueError`, and the per-star / per-pixel sentinels
//! through unchanged.
//!
//! Surface (forks of Phase 8):
//! - `noobase.image.build_stamp` + `StampResult` (the `image::stamp`
//!   leaf; mirrors the Rust core path, hence the image submodule level).
//! - `noobase.image.psf.{robust_combine, solve_flux_background,
//!   build_epsf, stitch_psf, build_extended_psf}` + their getter
//!   pyclasses (the `image::psf` orchestrators / leaves).
//!
//! `render` / `accumulate` / `refine_nuisance` stay Rust-only: they are
//! pure SR internals with no standalone Python use case and are already
//! pinned by the core tests.

use ::noobase::image::psf as core_psf;
use ::noobase::image::psf::{
    BuildEpsf, BuildEpsfParams, CombineMethod, ExtendedPsf, ExtendedPsfBuilt, ExtendedPsfParams,
    FluxBackground, ResidualReweight, RobustCombined, StitchParams,
};
use ::noobase::image::stamp as core_stamp;
use ::noobase::image::stamp::StampResult;
use ndarray::{Array2, Array3};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Boundary translation helpers
// ---------------------------------------------------------------------------

/// Every core `*Error` is a hard precondition; its `Display` is the
/// authoritative message. Map it to `ValueError` verbatim (the noobase-py
/// convention shared with grid/spectrum/photometry/image).
fn to_value_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Parse the flat `combine` string + companion scalars into the core
/// `CombineMethod` (fork 2). Mirrors the `parse_spacing` convention:
/// unknown literal -> `ValueError`. `median` ignores the companion
/// scalars; `clipped_mean`'s `kappa`/`max_iter` are validated by the core
/// (`ClippedMeanInvalidParams` -> `ValueError`), not pre-checked here.
fn parse_combine_method(
    method: &str,
    combine_kappa: f64,
    combine_max_iter: usize,
) -> PyResult<CombineMethod> {
    match method {
        "clipped_mean" => Ok(CombineMethod::ClippedMean {
            kappa: combine_kappa,
            max_iter: combine_max_iter,
        }),
        "median" => Ok(CombineMethod::Median),
        other => Err(PyValueError::new_err(format!(
            "invalid combine {other:?}; expected one of \"clipped_mean\", \"median\""
        ))),
    }
}

/// Parse the flat `residual_reweight` string + companion `reweight_c`
/// into the core `ResidualReweight` (fork 3). `none` ignores `reweight_c`;
/// `huber`/`tukey`'s `c` is validated by the core (`ParamsInvalid` ->
/// `ValueError`), not pre-checked here.
fn parse_residual_reweight(reweight: &str, reweight_c: f64) -> PyResult<ResidualReweight> {
    match reweight {
        "none" => Ok(ResidualReweight::None),
        "huber" => Ok(ResidualReweight::Huber { c: reweight_c }),
        "tukey" => Ok(ResidualReweight::Tukey { c: reweight_c }),
        other => Err(PyValueError::new_err(format!(
            "invalid residual_reweight {other:?}; expected one of \"none\", \"huber\", \"tukey\""
        ))),
    }
}

/// Extract an optional same-dtype companion 3-D array (`weight` /
/// `wing_weight`). `None` / Python `None` -> `None`; a dtype mismatch
/// with the primary data array is a `ValueError`.
fn extract_optional_array3<T>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
    expected_dtype: &'static str,
) -> PyResult<Option<Array3<T>>>
where
    T: numpy::Element + Clone,
{
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray3<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 3-D numpy array of dtype {expected_dtype} (matching the data array)"
        ))),
    }
}

/// Extract an optional same-dtype companion 2-D array (`error`). Same
/// contract as [`extract_optional_array3`].
fn extract_optional_array2<T>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
    expected_dtype: &'static str,
) -> PyResult<Option<Array2<T>>>
where
    T: numpy::Element + Clone,
{
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype {expected_dtype} (matching the cutout)"
        ))),
    }
}

/// Extract a required `float64` 2-D array (the non-generic model / delta
/// arrays: `epsf` / `core` / `wing` / `delta` / `delta_init` /
/// `wing_delta`; decision 12 -- psi / delta are always f64).
fn extract_required_f64_array2(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Array2<f64>> {
    match value.extract::<PyReadonlyArray2<'_, f64>>() {
        Ok(array) => Ok(array.as_array().to_owned()),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype float64"
        ))),
    }
}

/// Extract an optional `float64` 2-D array (`psi_init` /
/// `wing_confidence`; always f64).
fn extract_optional_f64_array2(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array2<f64>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, f64>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype float64"
        ))),
    }
}

/// Extract an optional `bool` 2-D mask (`true = invalid`; decision 7).
fn extract_optional_bool_array2(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Array2<bool>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray2<'_, bool>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(format!(
            "{name} must be a 2-D numpy array of dtype bool"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Output pyclasses (fork 3: a small getter pyclass per core output struct;
// numpy via zero-copy `to_pyarray`; sentinels passed through unchanged)
// ---------------------------------------------------------------------------

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

/// Result of ``robust_combine``: three ``(h, w)`` planes. A pixel with
/// no surviving sample carries ``combined = NaN``, ``weight = 0``,
/// ``count = 0`` (passed through unchanged, never raised).
#[pyclass(name = "RobustCombined", module = "noobase._core.image.psf", frozen)]
pub struct PyRobustCombined {
    inner: RobustCombined,
}

#[pymethods]
impl PyRobustCombined {
    /// The combined value per pixel, shape ``(h, w)``, dtype
    /// ``float64``. ``NaN`` for a no-survivor pixel.
    #[getter]
    fn combined<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.combined.to_pyarray(py).into_any()
    }

    /// The effective combined weight per pixel (sum of surviving
    /// weights; equals ``count`` when ``weight`` was ``None``), shape
    /// ``(h, w)``, dtype ``float64``. ``0`` for a no-survivor pixel.
    #[getter]
    fn weight<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.weight.to_pyarray(py).into_any()
    }

    /// The number of surviving samples per pixel, shape ``(h, w)``,
    /// dtype ``uint32``. ``0`` for a no-survivor pixel.
    #[getter]
    fn count<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.count.to_pyarray(py).into_any()
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = (
            self.inner.combined.shape()[0],
            self.inner.combined.shape()[1],
        );
        format!("RobustCombined(shape=({rows}, {cols}))")
    }
}

/// Result of ``solve_flux_background``: the per-star ``(flux,
/// background)`` at the supplied fixed ``delta``. An unsolved star
/// carries ``flux = NaN``, ``background = NaN``, ``ok = False`` (passed
/// through unchanged, never raised).
#[pyclass(name = "FluxBackground", module = "noobase._core.image.psf", frozen)]
pub struct PyFluxBackground {
    inner: FluxBackground,
}

#[pymethods]
impl PyFluxBackground {
    /// ``(N,)`` per-star flux, dtype ``float64``. ``NaN`` where
    /// ``ok = False``.
    #[getter]
    fn flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.flux.to_pyarray(py).into_any()
    }

    /// ``(N,)`` per-star scalar background, dtype ``float64``. ``NaN``
    /// where ``ok = False``.
    #[getter]
    fn background<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.background.to_pyarray(py).into_any()
    }

    /// ``(N,)`` per-star solvability, dtype ``bool``. ``False`` means
    /// too few valid pixels or a singular 2x2 normal matrix.
    #[getter]
    fn ok<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.ok.to_pyarray(py).into_any()
    }

    fn __repr__(&self) -> String {
        format!("FluxBackground(n_stars={})", self.inner.flux.len())
    }
}

/// Result of ``build_epsf``: the solved oversampled effective PSF plus
/// the refined per-star table and the global fixed-point flags. A star
/// that could not be solved carries the ``nuisance`` sentinel and
/// ``ok = False`` (passed through unchanged, never raised).
#[pyclass(name = "BuildEpsf", module = "noobase._core.image.psf", frozen)]
pub struct PyBuildEpsf {
    inner: BuildEpsf,
}

#[pymethods]
impl PyBuildEpsf {
    /// ``(os*s, os*s)`` solved oversampled effective PSF, unit-volume
    /// gauged (``sum(epsf) / os^2 = 1``), dtype ``float64``.
    #[getter]
    fn epsf<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.epsf.to_pyarray(py).into_any()
    }

    /// ``(N,)`` refined per-star flux, dtype ``float64``. ``NaN`` where
    /// ``ok = False``.
    #[getter]
    fn flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.flux.to_pyarray(py).into_any()
    }

    /// ``(N,)`` refined per-star scalar background, dtype ``float64``.
    /// ``NaN`` where ``ok = False``.
    #[getter]
    fn background<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.background.to_pyarray(py).into_any()
    }

    /// ``(N, 2)`` refined per-star centroid ``(delta_row, delta_col)``,
    /// dtype ``float64``.
    #[getter]
    fn delta<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.delta.to_pyarray(py).into_any()
    }

    /// ``(N,)`` per-star usability, dtype ``bool``. A ``False`` star did
    /// not enter the psi accumulate.
    #[getter]
    fn ok<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.ok.to_pyarray(py).into_any()
    }

    /// Number of outer SR iterations actually run.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Global convergence: ``‖Δpsi‖_rel < tol`` was reached strictly
    /// before ``max_iter`` (distinct from any per-star flag).
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = (self.inner.epsf.shape()[0], self.inner.epsf.shape()[1]);
        format!(
            "BuildEpsf(epsf=({rows}, {cols}), n_stars={}, iterations={}, converged={})",
            self.inner.flux.len(),
            self.inner.iterations,
            self.inner.converged
        )
    }
}

/// The hybrid / multi-resolution extended PSF: an oversampled core plus
/// a native wing plus the meta-info linking the two grids.
///
/// Sampling contract: ``recon(r) = f_core(r) * core_sample(r) +
/// wing_sample(r)`` with the raised-cosine ``f_core = 1 - f_wing``. The
/// stored ``wing`` already has ``f_wing`` and the scalar match baked in
/// (so it is ~0 in the core region); both planes are encircled-energy
/// normalized.
#[pyclass(name = "ExtendedPsf", module = "noobase._core.image.psf", frozen)]
pub struct PyExtendedPsf {
    inner: ExtendedPsf,
}

#[pymethods]
impl PyExtendedPsf {
    /// ``(os*s, os*s)`` oversampled core, EE-normalized, dtype
    /// ``float64``.
    #[getter]
    fn core<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.core.to_pyarray(py).into_any()
    }

    /// The oversample factor ``os`` (the meta-info linking the two
    /// grids).
    #[getter]
    fn oversample(&self) -> usize {
        self.inner.oversample
    }

    /// ``(H, W)`` native wing: scalar-matched at ``match_radius``,
    /// raised-cosine feathered (so ~0 in the core region), and
    /// EE-normalized, dtype ``float64``.
    #[getter]
    fn wing<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.wing.to_pyarray(py).into_any()
    }

    /// The native-pixel core<->wing crossover radius.
    #[getter]
    fn match_radius(&self) -> f64 {
        self.inner.match_radius
    }

    /// The native-pixel feather full width.
    #[getter]
    fn feather_width(&self) -> f64 {
        self.inner.feather_width
    }

    /// The native-pixel encircled-energy aperture radius.
    #[getter]
    fn ee_aperture_radius(&self) -> f64 {
        self.inner.ee_aperture_radius
    }

    fn __repr__(&self) -> String {
        let (core_rows, core_cols) = (self.inner.core.shape()[0], self.inner.core.shape()[1]);
        let (wing_rows, wing_cols) = (self.inner.wing.shape()[0], self.inner.wing.shape()[1]);
        format!(
            "ExtendedPsf(core=({core_rows}, {core_cols}), oversample={}, wing=({wing_rows}, {wing_cols}))",
            self.inner.oversample
        )
    }
}

/// Result of ``build_extended_psf``: the stitched ``ExtendedPsf`` plus
/// the per-bright-star table. A star that cannot be calibrated carries
/// ``star_ok = False`` and ``NaN`` scale/background (passed through
/// unchanged, never raised).
#[pyclass(name = "ExtendedPsfBuilt", module = "noobase._core.image.psf", frozen)]
pub struct PyExtendedPsfBuilt {
    inner: ExtendedPsfBuilt,
}

#[pymethods]
impl PyExtendedPsfBuilt {
    /// The stitched hybrid extended PSF (an ``ExtendedPsf``).
    #[getter]
    fn extended(&self) -> PyExtendedPsf {
        PyExtendedPsf {
            inner: self.inner.extended.clone(),
        }
    }

    /// ``(M,)`` per-stamp scale actually used (core-solve flux or
    /// aperture-photometry fallback), dtype ``float64``. ``NaN`` where
    /// ``star_ok = False``.
    #[getter]
    fn star_flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.star_flux.to_pyarray(py).into_any()
    }

    /// ``(M,)`` per-stamp robust sky actually subtracted, dtype
    /// ``float64``. ``NaN`` where ``star_ok = False``.
    #[getter]
    fn star_background<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.star_background.to_pyarray(py).into_any()
    }

    /// ``(M,)`` per-star usability, dtype ``bool``. ``False`` means
    /// saturated and the aperture fallback was also unusable.
    #[getter]
    fn star_ok<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.star_ok.to_pyarray(py).into_any()
    }

    /// ``(M,)`` scale provenance, dtype ``bool``: ``True`` = the core
    /// ``solve_flux_background``, ``False`` = the aperture-photometry
    /// fallback. Meaningless where ``star_ok = False``.
    #[getter]
    fn star_scale_from_core<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.star_scale_from_core.to_pyarray(py).into_any()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtendedPsfBuilt(n_stars={}, core=({}, {}), wing=({}, {}))",
            self.inner.star_flux.len(),
            self.inner.extended.core.shape()[0],
            self.inner.extended.core.shape()[1],
            self.inner.extended.wing.shape()[0],
            self.inner.extended.wing.shape()[1],
        )
    }
}

// ---------------------------------------------------------------------------
// Bound entry points
// ---------------------------------------------------------------------------

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
    let mask_owned = extract_optional_bool_array2(mask, "mask")?;
    let mask_view = mask_owned.as_ref().map(|array| array.view());

    let result = if let Ok(cutout_f64) = cutout.extract::<PyReadonlyArray2<'_, f64>>() {
        let cutout_owned: Array2<f64> = cutout_f64.as_array().to_owned();
        let error_owned = extract_optional_array2::<f64>(error, "error", "float64")?;
        let error_view = error_owned.as_ref().map(|array| array.view());
        core_stamp::build_stamp::<f64>(
            cutout_owned.view(),
            stamp_size,
            error_view,
            mask_view,
            weight_fwhm,
            max_iter,
            tol,
        )
        .map_err(to_value_error)?
    } else if let Ok(cutout_f32) = cutout.extract::<PyReadonlyArray2<'_, f32>>() {
        let cutout_owned: Array2<f32> = cutout_f32.as_array().to_owned();
        let error_owned = extract_optional_array2::<f32>(error, "error", "float32")?;
        let error_view = error_owned.as_ref().map(|array| array.view());
        core_stamp::build_stamp::<f32>(
            cutout_owned.view(),
            stamp_size,
            error_view,
            mask_view,
            weight_fwhm,
            max_iter,
            tol,
        )
        .map_err(to_value_error)?
    } else {
        return Err(PyValueError::new_err(
            "cutout must be a 2-D numpy array of dtype float32 or float64",
        ));
    };

    Ok(result.map(|inner| PyStampResult { inner }))
}

/// Robustly combine an aligned ``(N, h, w)`` stack along the stamp axis
/// with sign-agnostic outlier rejection.
///
/// A pure cross-N reducer: the caller passes an already-aligned native
/// stack plus the complete per-pixel weight. Used for the extended-PSF
/// native wing stack.
///
/// Parameters
/// ----------
/// stack : ndarray
///     ``(N, h, w)`` caller-aligned native stamp stack, dtype
///     ``float32`` or ``float64``. Determines the dtype channel;
///     ``weight`` must match. Non-finite samples are excluded.
/// weight : ndarray, optional
///     ``(N, h, w)`` complete per-pixel weight (``valid * 1/sigma^2 *
///     ...``, formed by the caller). ``None`` is equal weight. A sample
///     with non-finite or non-positive weight is excluded. Default
///     ``None``.
/// method : {"clipped_mean", "median"}, optional
///     ``clipped_mean`` is a symmetric sigma-clipped inverse-variance
///     weighted mean; ``median`` is the unweighted median of the
///     inclusion-gated samples. Default ``"clipped_mean"``.
/// combine_kappa : float, optional
///     ``clipped_mean`` clip threshold in robust sigmas. Must be ``>
///     0``. Ignored for ``median``. Default ``3.0``.
/// combine_max_iter : int, optional
///     ``clipped_mean`` clip-iteration cap. Must be ``> 0``. Ignored for
///     ``median``. Default ``5``.
///
/// Returns
/// -------
/// RobustCombined
///     Three ``(h, w)`` planes. A pixel with no surviving sample carries
///     ``combined = NaN``, ``weight = 0``, ``count = 0`` -- a per-pixel
///     sentinel, not an error.
///
/// Raises
/// ------
/// ValueError
///     If ``weight`` is given with a shape different from ``stack``, if
///     ``method`` is invalid, if ``method == "clipped_mean"`` with
///     ``combine_kappa <= 0`` or ``combine_max_iter == 0``, or if an
///     array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "robust_combine")]
#[pyo3(signature = (stack, *, weight=None, method="clipped_mean", combine_kappa=3.0, combine_max_iter=5))]
#[pyo3(
    text_signature = "(stack, *, weight=None, method=\"clipped_mean\", combine_kappa=3.0, combine_max_iter=5)"
)]
fn robust_combine_function(
    stack: &Bound<'_, PyAny>,
    weight: Option<&Bound<'_, PyAny>>,
    method: &str,
    combine_kappa: f64,
    combine_max_iter: usize,
) -> PyResult<PyRobustCombined> {
    let combine_method = parse_combine_method(method, combine_kappa, combine_max_iter)?;

    let inner = if let Ok(stack_f64) = stack.extract::<PyReadonlyArray3<'_, f64>>() {
        let stack_owned: Array3<f64> = stack_f64.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f64>(weight, "weight", "float64")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::robust_combine::<f64>(stack_owned.view(), weight_view, combine_method)
            .map_err(to_value_error)?
    } else if let Ok(stack_f32) = stack.extract::<PyReadonlyArray3<'_, f32>>() {
        let stack_owned: Array3<f32> = stack_f32.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f32>(weight, "weight", "float32")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::robust_combine::<f32>(stack_owned.view(), weight_view, combine_method)
            .map_err(to_value_error)?
    } else {
        return Err(PyValueError::new_err(
            "stack must be a 3-D numpy array of dtype float32 or float64",
        ));
    };

    Ok(PyRobustCombined { inner })
}

/// Solve the per-star ``(flux, background)`` at a fixed ``delta`` by an
/// exact 2x2 weighted linear least squares (no iteration).
///
/// Parameters
/// ----------
/// epsf : ndarray
///     ``(os*s, os*s)`` oversampled effective PSF model, dtype
///     ``float64`` (the model is always f64).
/// oversample : int
///     The oversample factor ``os``; forced odd. ``s`` is recovered from
///     ``epsf`` / ``os``.
/// data : ndarray
///     ``(N, s, s)`` per-stamp detector data, dtype ``float32`` or
///     ``float64``. Determines the dtype channel; ``weight`` must match.
/// delta : ndarray
///     ``(N, 2)`` fixed per-stamp sub-pixel offset, dtype ``float64``.
/// weight : ndarray, optional
///     ``(N, s, s)`` complete per-pixel weight, same dtype as ``data``.
///     ``None`` is equal weight. Default ``None``.
///
/// Returns
/// -------
/// FluxBackground
///     ``(N,)`` ``flux`` / ``background`` / ``ok``. An unsolved star
///     carries ``NaN`` flux/background and ``ok = False`` -- a per-star
///     sentinel, not an error.
///
/// Raises
/// ------
/// ValueError
///     If ``epsf`` is not square, ``oversample`` is not odd, the epsf
///     side is not a multiple of ``oversample``, the derived ``s`` is
///     even, the batch dimensions disagree, ``weight`` shape does not
///     equal ``data``, or an array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "solve_flux_background")]
#[pyo3(signature = (epsf, oversample, data, delta, *, weight=None))]
#[pyo3(text_signature = "(epsf, oversample, data, delta, *, weight=None)")]
fn solve_flux_background_function(
    epsf: &Bound<'_, PyAny>,
    oversample: usize,
    data: &Bound<'_, PyAny>,
    delta: &Bound<'_, PyAny>,
    weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyFluxBackground> {
    let epsf_owned = extract_required_f64_array2(epsf, "epsf")?;
    let delta_owned = extract_required_f64_array2(delta, "delta")?;

    let inner = if let Ok(data_f64) = data.extract::<PyReadonlyArray3<'_, f64>>() {
        let data_owned: Array3<f64> = data_f64.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f64>(weight, "weight", "float64")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::solve_flux_background::<f64>(
            epsf_owned.view(),
            oversample,
            data_owned.view(),
            weight_view,
            delta_owned.view(),
        )
        .map_err(to_value_error)?
    } else if let Ok(data_f32) = data.extract::<PyReadonlyArray3<'_, f32>>() {
        let data_owned: Array3<f32> = data_f32.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f32>(weight, "weight", "float32")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::solve_flux_background::<f32>(
            epsf_owned.view(),
            oversample,
            data_owned.view(),
            weight_view,
            delta_owned.view(),
        )
        .map_err(to_value_error)?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 3-D numpy array of dtype float32 or float64",
        ));
    };

    Ok(PyFluxBackground { inner })
}

/// Solve the core oversampled effective PSF from a caller-assembled
/// stamp stack by projected-Landweber super-resolution, jointly refining
/// the per-star nuisance table.
///
/// Parameters
/// ----------
/// data : ndarray
///     ``(N, s, s)`` caller-assembled stamp stack (from ``build_stamp``,
///     scheme C), dtype ``float32`` or ``float64``. Determines the dtype
///     channel; ``weight`` must match.
/// delta_init : ndarray
///     ``(N, 2)`` per-stamp ``delta`` initial value (from
///     ``build_stamp``; in-basin), dtype ``float64``.
/// oversample : int
///     The oversample factor ``os``; forced odd. ``s`` is recovered from
///     the ``data`` stamp dimension.
/// weight : ndarray, optional
///     ``(N, s, s)`` complete per-pixel base weight, same dtype as
///     ``data``. ``None`` is unit weight. The residual reweight
///     multiplies an internal copy; this array is never mutated. Default
///     ``None``.
/// psi_init : ndarray, optional
///     ``(os*s, os*s)`` warm-start psi, dtype ``float64``. ``None``
///     triggers the internal seed. Default ``None``.
/// max_iter : int, optional
///     Outer SR iteration cap. Must be ``> 0``. Default ``50``.
/// tol : float, optional
///     ``‖Δpsi‖_rel < tol`` convergence threshold. Finite, ``> 0``.
///     Default ``1e-4``.
/// step : float, optional
///     Landweber relaxation (operator-norm adaptive, so ``1.0`` is a
///     safe full step regardless of flux scale). Finite, ``> 0``.
///     Default ``1.0``.
/// residual_reweight : {"none", "huber", "tukey"}, optional
///     Per-pixel residual M-estimator down-weighting. Symmetric /
///     sign-agnostic. Default ``"none"``.
/// reweight_c : float, optional
///     The Huber / Tukey transition in robust sigmas. Must be finite and
///     ``> 0`` for ``"huber"`` / ``"tukey"``. Ignored for ``"none"``.
///     Default ``4.0``.
/// nuisance_max_iter : int, optional
///     Per-round ``refine_nuisance`` inner-iteration cap. Must be ``>
///     0``; ``1`` = one block-coordinate step. Default ``3``.
/// nuisance_tol : float, optional
///     Per-round ``refine_nuisance`` ``‖Δdelta‖`` threshold. Finite,
///     ``> 0``. Default ``1e-4``.
///
/// Returns
/// -------
/// BuildEpsf
///     The unit-volume-gauged ``epsf``, the per-star
///     ``flux`` / ``background`` / ``delta`` / ``ok``, and the global
///     ``iterations`` / ``converged``. A star that cannot be solved
///     carries the sentinel and ``ok = False`` -- a per-star sentinel,
///     not an error.
///
/// Raises
/// ------
/// ValueError
///     If ``oversample`` is not odd, the recovered ``s`` is even, the
///     batch dimensions disagree, ``weight`` / ``psi_init`` shapes are
///     wrong, any scalar parameter is invalid, ``residual_reweight`` is
///     invalid, or an array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "build_epsf")]
#[pyo3(signature = (
    data, delta_init, oversample, *, weight=None, psi_init=None,
    max_iter=core_psf::build_epsf::DEFAULT_MAX_ITER,
    tol=core_psf::build_epsf::DEFAULT_TOL,
    step=core_psf::build_epsf::DEFAULT_STEP,
    residual_reweight="none", reweight_c=4.0,
    nuisance_max_iter=core_psf::build_epsf::DEFAULT_NUISANCE_MAX_ITER,
    nuisance_tol=core_psf::build_epsf::DEFAULT_NUISANCE_TOL
))]
#[pyo3(
    text_signature = "(data, delta_init, oversample, *, weight=None, psi_init=None, max_iter=50, tol=1e-4, step=1.0, residual_reweight=\"none\", reweight_c=4.0, nuisance_max_iter=3, nuisance_tol=1e-4)"
)]
#[allow(clippy::too_many_arguments)]
fn build_epsf_function(
    data: &Bound<'_, PyAny>,
    delta_init: &Bound<'_, PyAny>,
    oversample: usize,
    weight: Option<&Bound<'_, PyAny>>,
    psi_init: Option<&Bound<'_, PyAny>>,
    max_iter: usize,
    tol: f64,
    step: f64,
    residual_reweight: &str,
    reweight_c: f64,
    nuisance_max_iter: usize,
    nuisance_tol: f64,
) -> PyResult<PyBuildEpsf> {
    let delta_init_owned = extract_required_f64_array2(delta_init, "delta_init")?;
    let psi_init_owned = extract_optional_f64_array2(psi_init, "psi_init")?;
    let psi_init_view = psi_init_owned.as_ref().map(|array| array.view());
    let params = BuildEpsfParams {
        max_iter,
        tol,
        step,
        residual_reweight: parse_residual_reweight(residual_reweight, reweight_c)?,
        nuisance_max_iter,
        nuisance_tol,
    };

    let inner = if let Ok(data_f64) = data.extract::<PyReadonlyArray3<'_, f64>>() {
        let data_owned: Array3<f64> = data_f64.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f64>(weight, "weight", "float64")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::build_epsf::<f64>(
            data_owned.view(),
            weight_view,
            delta_init_owned.view(),
            oversample,
            psi_init_view,
            params,
        )
        .map_err(to_value_error)?
    } else if let Ok(data_f32) = data.extract::<PyReadonlyArray3<'_, f32>>() {
        let data_owned: Array3<f32> = data_f32.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f32>(weight, "weight", "float32")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::build_epsf::<f32>(
            data_owned.view(),
            weight_view,
            delta_init_owned.view(),
            oversample,
            psi_init_view,
            params,
        )
        .map_err(to_value_error)?
    } else {
        return Err(PyValueError::new_err(
            "data must be a 3-D numpy array of dtype float32 or float64",
        ));
    };

    Ok(PyBuildEpsf { inner })
}

/// Stitch a Phase-6 oversampled core and a native wing into a hybrid
/// encircled-energy-normalized extended PSF. The core is not re-solved.
///
/// Parameters
/// ----------
/// core : ndarray
///     ``(os*s, os*s)`` Phase-6 oversampled ePSF (unit-volume gauged),
///     dtype ``float64`` (the model is always f64).
/// oversample : int
///     The oversample factor ``os``; forced odd. The core native edge
///     ``s`` is recovered from ``core`` / ``os``.
/// wing : ndarray
///     ``(H, W)`` native large-support robust wing, centered, ``H`` /
///     ``W`` odd, dtype ``float64``.
/// wing_confidence : ndarray, optional
///     ``(H, W)`` per-pixel confidence (the ``robust_combine`` weight /
///     count), dtype ``float64``. Only weights the seam scalar match;
///     not stored. ``None`` is uniform. Default ``None``.
/// match_radius : float, optional
///     Native-pixel core<->wing crossover (the feather ring center).
///     Finite, ``> 0``. Default ``8.0``.
/// feather_width : float, optional
///     Native-pixel raised-cosine feather full width. Finite, ``> 0``.
///     Default ``4.0``.
/// ee_aperture_radius : float, optional
///     Native-pixel circular aperture the result is encircled-energy
///     normalized to (integral within == 1). Finite, ``> 0``. Default
///     ``15.0``.
///
/// Returns
/// -------
/// ExtendedPsf
///     The hybrid / multi-resolution extended PSF. A degenerate
///     all-sentinel wing yields a pure-core EE-normalized result, not an
///     error.
///
/// Raises
/// ------
/// ValueError
///     If ``core`` is not square, ``oversample`` is not odd, the core
///     side is not a multiple of ``oversample``, the derived ``s`` is
///     even, ``wing`` does not have odd dimensions, ``wing_confidence``
///     shape does not equal ``wing``, the stitch params are infeasible,
///     or an array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "stitch_psf")]
#[pyo3(signature = (
    core, oversample, wing, *, wing_confidence=None,
    match_radius=core_psf::extended::DEFAULT_MATCH_RADIUS,
    feather_width=core_psf::extended::DEFAULT_FEATHER_WIDTH,
    ee_aperture_radius=core_psf::extended::DEFAULT_EE_APERTURE_RADIUS
))]
#[pyo3(
    text_signature = "(core, oversample, wing, *, wing_confidence=None, match_radius=8.0, feather_width=4.0, ee_aperture_radius=15.0)"
)]
fn stitch_psf_function(
    core: &Bound<'_, PyAny>,
    oversample: usize,
    wing: &Bound<'_, PyAny>,
    wing_confidence: Option<&Bound<'_, PyAny>>,
    match_radius: f64,
    feather_width: f64,
    ee_aperture_radius: f64,
) -> PyResult<PyExtendedPsf> {
    let core_owned = extract_required_f64_array2(core, "core")?;
    let wing_owned = extract_required_f64_array2(wing, "wing")?;
    let confidence_owned = extract_optional_f64_array2(wing_confidence, "wing_confidence")?;
    let confidence_view = confidence_owned.as_ref().map(|array| array.view());
    let params = StitchParams {
        match_radius,
        feather_width,
        ee_aperture_radius,
    };

    let inner = core_psf::stitch_psf(
        core_owned.view(),
        oversample,
        wing_owned.view(),
        confidence_view,
        params,
    )
    .map_err(to_value_error)?;

    Ok(PyExtendedPsf { inner })
}

/// Build the native wing from a bright-star stack (per-stamp robust sky
/// + per-stamp scale + robust combine) and stitch it onto a Phase-6
/// core.
///
/// Parameters
/// ----------
/// wing_data : ndarray
///     ``(M, H, W)`` caller-aligned bright-star large-support native
///     stamps, dtype ``float32`` or ``float64``. Determines the dtype
///     channel; ``wing_weight`` must match.
/// wing_delta : ndarray
///     ``(M, 2)`` bright-star ``build_stamp`` delta, dtype ``float64``.
///     Used only for integer re-centering bookkeeping (the sub-pixel
///     part is intentionally not applied -- the wing is sky-dominated).
/// core : ndarray
///     ``(os*s, os*s)`` Phase-6 ePSF, dtype ``float64`` (the core is not
///     re-solved here).
/// oversample : int
///     The oversample factor ``os``; forced odd.
/// wing_weight : ndarray, optional
///     ``(M, H, W)`` complete per-pixel base weight, same dtype as
///     ``wing_data``. ``None`` is unit weight. Default ``None``.
/// match_radius : float, optional
///     Native-pixel core<->wing crossover. Finite, ``> 0``. Default
///     ``8.0``.
/// feather_width : float, optional
///     Native-pixel raised-cosine feather full width. Finite, ``> 0``.
///     Default ``4.0``.
/// ee_aperture_radius : float, optional
///     Native-pixel encircled-energy aperture radius. Finite, ``> 0``.
///     Default ``15.0``.
/// combine : {"clipped_mean", "median"}, optional
///     The ``robust_combine`` method for the cross-M wing stack. Default
///     ``"clipped_mean"``.
/// combine_kappa : float, optional
///     ``clipped_mean`` clip threshold in robust sigmas. Must be ``>
///     0``. Ignored for ``median``. Default ``3.0``.
/// combine_max_iter : int, optional
///     ``clipped_mean`` clip-iteration cap. Must be ``> 0``. Ignored for
///     ``median``. Default ``5``.
/// scale_aperture_radius : float, optional
///     Native-pixel aperture radius for the aperture-photometry fallback
///     scale (used when the core solve is saturated). Finite, ``> 0``.
///     Default ``6.0``.
/// scale_background_annulus : tuple of (float, float), optional
///     Native-pixel ``(r_in, r_out)`` ring for the per-stamp robust sky.
///     Finite, ``0 <= r_in < r_out``. Default ``(18.0, 24.0)``.
///
/// Returns
/// -------
/// ExtendedPsfBuilt
///     The stitched ``ExtendedPsf`` plus the per-bright-star table. A
///     star that cannot be calibrated carries ``star_ok = False`` -- a
///     per-star sentinel, not an error.
///
/// Raises
/// ------
/// ValueError
///     If ``core`` is not square, ``oversample`` is not odd, the core
///     side is not a multiple of ``oversample``, the derived ``s`` is
///     even, ``wing_data`` does not have odd spatial dimensions, the
///     batch dimensions disagree, ``wing_weight`` shape does not equal
///     ``wing_data``, any param is invalid, ``combine`` is invalid, or
///     an array has an unsupported dtype.
#[pyfunction]
#[pyo3(name = "build_extended_psf")]
#[pyo3(signature = (
    wing_data, wing_delta, core, oversample, *, wing_weight=None,
    match_radius=core_psf::extended::DEFAULT_MATCH_RADIUS,
    feather_width=core_psf::extended::DEFAULT_FEATHER_WIDTH,
    ee_aperture_radius=core_psf::extended::DEFAULT_EE_APERTURE_RADIUS,
    combine="clipped_mean", combine_kappa=3.0, combine_max_iter=5,
    scale_aperture_radius=core_psf::extended::DEFAULT_SCALE_APERTURE_RADIUS,
    scale_background_annulus=core_psf::extended::DEFAULT_SCALE_BACKGROUND_ANNULUS
))]
#[pyo3(
    text_signature = "(wing_data, wing_delta, core, oversample, *, wing_weight=None, match_radius=8.0, feather_width=4.0, ee_aperture_radius=15.0, combine=\"clipped_mean\", combine_kappa=3.0, combine_max_iter=5, scale_aperture_radius=6.0, scale_background_annulus=(18.0, 24.0))"
)]
#[allow(clippy::too_many_arguments)]
fn build_extended_psf_function(
    wing_data: &Bound<'_, PyAny>,
    wing_delta: &Bound<'_, PyAny>,
    core: &Bound<'_, PyAny>,
    oversample: usize,
    wing_weight: Option<&Bound<'_, PyAny>>,
    match_radius: f64,
    feather_width: f64,
    ee_aperture_radius: f64,
    combine: &str,
    combine_kappa: f64,
    combine_max_iter: usize,
    scale_aperture_radius: f64,
    scale_background_annulus: (f64, f64),
) -> PyResult<PyExtendedPsfBuilt> {
    let wing_delta_owned = extract_required_f64_array2(wing_delta, "wing_delta")?;
    let core_owned = extract_required_f64_array2(core, "core")?;
    let params = ExtendedPsfParams {
        stitch: StitchParams {
            match_radius,
            feather_width,
            ee_aperture_radius,
        },
        combine: parse_combine_method(combine, combine_kappa, combine_max_iter)?,
        scale_aperture_radius,
        scale_background_annulus,
    };

    let inner = if let Ok(wing_data_f64) = wing_data.extract::<PyReadonlyArray3<'_, f64>>() {
        let wing_data_owned: Array3<f64> = wing_data_f64.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f64>(wing_weight, "wing_weight", "float64")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::build_extended_psf::<f64>(
            wing_data_owned.view(),
            weight_view,
            wing_delta_owned.view(),
            core_owned.view(),
            oversample,
            params,
        )
        .map_err(to_value_error)?
    } else if let Ok(wing_data_f32) = wing_data.extract::<PyReadonlyArray3<'_, f32>>() {
        let wing_data_owned: Array3<f32> = wing_data_f32.as_array().to_owned();
        let weight_owned = extract_optional_array3::<f32>(wing_weight, "wing_weight", "float32")?;
        let weight_view = weight_owned.as_ref().map(|array| array.view());
        core_psf::build_extended_psf::<f32>(
            wing_data_owned.view(),
            weight_view,
            wing_delta_owned.view(),
            core_owned.view(),
            oversample,
            params,
        )
        .map_err(to_value_error)?
    } else {
        return Err(PyValueError::new_err(
            "wing_data must be a 3-D numpy array of dtype float32 or float64",
        ));
    };

    Ok(PyExtendedPsfBuilt { inner })
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/// Register the Phase 8 surface onto the `image` submodule, mirroring
/// the Rust core paths: `build_stamp` / `StampResult` are
/// `image::stamp`, so they live at the image level; the five `psf`
/// orchestrators / leaves live in a nested `noobase._core.image.psf`
/// submodule (mirroring `image::psf`).
pub(crate) fn register<'py>(py: Python<'py>, image: &Bound<'py, PyModule>) -> PyResult<()> {
    image.add_function(wrap_pyfunction!(build_stamp_function, image)?)?;
    image.add_class::<PyStampResult>()?;

    let psf = PyModule::new(py, "psf")?;
    psf.add_function(wrap_pyfunction!(robust_combine_function, &psf)?)?;
    psf.add_function(wrap_pyfunction!(solve_flux_background_function, &psf)?)?;
    psf.add_function(wrap_pyfunction!(build_epsf_function, &psf)?)?;
    psf.add_function(wrap_pyfunction!(stitch_psf_function, &psf)?)?;
    psf.add_function(wrap_pyfunction!(build_extended_psf_function, &psf)?)?;
    psf.add_class::<PyRobustCombined>()?;
    psf.add_class::<PyFluxBackground>()?;
    psf.add_class::<PyBuildEpsf>()?;
    psf.add_class::<PyExtendedPsf>()?;
    psf.add_class::<PyExtendedPsfBuilt>()?;
    image.add_submodule(&psf)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.image.psf", &psf)?;
    Ok(())
}

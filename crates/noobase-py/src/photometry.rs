use ::noobase::axis::Grid as CoreGrid;
use ::noobase::spectroscopy::synthetic_photometry as core_photometry;
use ::noobase::spectroscopy::synthetic_photometry::PhotometryConvention;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{
    GridChannel, Scalar, band_pair, band_triple, coerce_to_grid, dispatch_array, grid_channel,
    parse_convention, required_typed_array1, to_value_error, with_inner,
};
use crate::grid::{GridInner, PyGrid};

/// Extract a required channel-`T` array, or the photometry-family
/// "`<what>` dtype must match `<reference>` (`<dtype>`)" `ValueError`.
fn typed_channel<T: Scalar>(
    value: &Bound<'_, PyAny>,
    what: &str,
    reference: &str,
) -> PyResult<Array1<T>> {
    value
        .extract::<PyReadonlyArray1<'_, T>>()
        .map(|array| array.as_array().to_owned())
        .map_err(|_| {
            PyValueError::new_err(format!(
                "{what} dtype must match {reference} ({})",
                T::DTYPE_NAME
            ))
        })
}

/// Optional companion of [`typed_channel`]. Absent / Python `None` ->
/// `None`; a non-`T` dtype yields the same message form.
fn optional_typed_channel<T: Scalar>(
    value: Option<&Bound<'_, PyAny>>,
    what: &str,
    reference: &str,
) -> PyResult<Option<Array1<T>>> {
    match value {
        Some(bound) if !bound.is_none() => typed_channel::<T>(bound, what, reference).map(Some),
        _ => Ok(None),
    }
}

/// Compute synthetic photometry of a spectrum through a transmission curve.
///
/// Returns the band-averaged flux density, the propagated 1-sigma
/// uncertainty (only when ``spectrum_error`` is given), and the geometric
/// coverage of the filter by the spectrum.
///
/// All arguments are keyword-only. The dtype channel (``float32`` or
/// ``float64``) is determined by ``spectrum_flux``; every other array
/// argument must match.
///
/// Parameters
/// ----------
/// spectrum_grid : Grid or ndarray
///     Spectrum wavelength axis. If an ndarray is passed, its length must
///     equal ``len(spectrum_flux)`` (centers) or ``len(spectrum_flux) + 1``
///     (edges); spacing is assumed linear and ``kind`` is inferred from the
///     length match. For log-spaced spectra, pass a pre-built Grid.
/// spectrum_flux : ndarray
///     Flux density per bin. Determines the dtype channel for the whole
///     call; all other arrays must match.
/// spectrum_error : ndarray, optional
///     1-sigma uncertainty per bin. Same length and dtype as
///     ``spectrum_flux``. Default is ``None``.
/// transmission_grid : Grid or ndarray
///     Filter wavelength axis. Same dtype and length-match rules as
///     ``spectrum_grid``, paired with ``transmission_values``.
/// transmission_values : ndarray
///     Filter transmission per bin. Same dtype as ``spectrum_flux``.
/// convention : {"photon_counting", "energy_weighted"}, optional
///     Photon-counting (default) is appropriate for photon-counting
///     detectors such as CCDs and HgCdTe arrays (including JWST NIRCam).
///     Energy-weighted is appropriate for bolometric / energy-integrating
///     detectors.
///
/// Returns
/// -------
/// tuple of (float, float or None, float)
///     ``(band_flux, band_error, coverage)``. ``band_error`` is ``None``
///     when ``spectrum_error`` was not provided. ``coverage`` is the
///     fraction of the filter's transmission integral probed by the
///     spectrum (1.0 means full coverage; a value below 1.0 means the
///     spectrum does not span the full filter and ``band_flux`` is biased
///     low).
///
/// Raises
/// ------
/// ValueError
///     If dtypes mismatch across inputs, array lengths are inconsistent,
///     ``convention`` is invalid, or the spectrum does not overlap the
///     filter in wavelength.
///
/// Notes
/// -----
/// The error is propagated assuming spectrum bins are statistically
/// independent. The transmission curve's bin density does not affect this
/// assumption — it only enters the deterministic weights. The assumption
/// is violated if the spectrum was previously upsampled (for example via
/// ``Spectrum.rebin`` onto a finer grid), in which case the returned
/// ``band_error`` underestimates the true uncertainty. See
/// ``Spectrum.rebin`` for the same caveat.
#[pyfunction]
#[pyo3(name = "synthetic")]
#[pyo3(signature = (*, spectrum_grid, spectrum_flux, spectrum_error=None, transmission_grid, transmission_values, convention="photon_counting"))]
#[pyo3(
    text_signature = "(*, spectrum_grid, spectrum_flux, spectrum_error=None, transmission_grid, transmission_values, convention=\"photon_counting\")"
)]
fn synthetic_function<'py>(
    py: Python<'py>,
    spectrum_grid: &Bound<'py, PyAny>,
    spectrum_flux: &Bound<'py, PyAny>,
    spectrum_error: Option<&Bound<'py, PyAny>>,
    transmission_grid: &Bound<'py, PyAny>,
    transmission_values: &Bound<'py, PyAny>,
    convention: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let convention_enum = parse_convention(convention)?;
    dispatch_array!(
        spectrum_flux,
        1,
        "spectrum_flux",
        synthetic_impl,
        py,
        spectrum_grid,
        spectrum_error,
        transmission_grid,
        transmission_values,
        convention_enum
    )
}

fn synthetic_impl<'py, T: Scalar + GridChannel>(
    flux: Array1<T>,
    py: Python<'py>,
    spectrum_grid: &Bound<'py, PyAny>,
    spectrum_error: Option<&Bound<'py, PyAny>>,
    transmission_grid: &Bound<'py, PyAny>,
    transmission_values: &Bound<'py, PyAny>,
    convention: PhotometryConvention,
) -> PyResult<Bound<'py, PyAny>> {
    let transmission =
        typed_channel::<T>(transmission_values, "transmission_values", "spectrum_flux")?;
    let error = optional_typed_channel::<T>(spectrum_error, "spectrum_error", "spectrum_flux")?;

    let spectrum_grid_py = coerce_to_grid(spectrum_grid, flux.len(), T::IS_F32, "spectrum")?;
    let transmission_grid_py = coerce_to_grid(
        transmission_grid,
        transmission.len(),
        T::IS_F32,
        "transmission",
    )?;
    let spectrum_core =
        grid_channel::<T>(spectrum_grid_py, "spectrum").expect("coerce_to_grid enforces dtype");
    let transmission_core = grid_channel::<T>(transmission_grid_py, "transmission")
        .expect("coerce_to_grid enforces dtype");

    let error_view = error.as_ref().map(|array| array.view());
    let result = core_photometry::synthetic::<T>(
        &spectrum_core,
        flux.view(),
        error_view,
        &transmission_core,
        transmission.view(),
        convention,
    )
    .map_err(to_value_error)?;

    band_triple(
        py,
        result.band_flux.promote(),
        result.band_error.map(|value| value.promote()),
        result.coverage.promote(),
    )
}

pub(crate) enum SyntheticOperatorInner {
    F32(core_photometry::SyntheticOperator<f32>),
    F64(core_photometry::SyntheticOperator<f64>),
}

/// Re-wrap a channel-`T` core operator into the dtype-erased
/// [`SyntheticOperatorInner`] so the generic constructor stays
/// single-bodied.
trait IntoOperatorInner {
    fn into_operator_inner(self) -> SyntheticOperatorInner;
}

impl IntoOperatorInner for core_photometry::SyntheticOperator<f32> {
    fn into_operator_inner(self) -> SyntheticOperatorInner {
        SyntheticOperatorInner::F32(self)
    }
}

impl IntoOperatorInner for core_photometry::SyntheticOperator<f64> {
    fn into_operator_inner(self) -> SyntheticOperatorInner {
        SyntheticOperatorInner::F64(self)
    }
}

/// Pre-built synthetic photometry operator with cached weights.
///
/// Amortizes the cost of grid intersection and transmission weighting for
/// repeated evaluation against many spectra that share the same wavelength
/// axis and transmission curve. ``apply`` then becomes a small inner
/// product. Useful for SED-fitting inner loops.
///
/// The operator caches both the deterministic weights and the geometric
/// coverage (available via the ``coverage`` property), so per-spectrum
/// calls do not recompute either.
#[pyclass(name = "SyntheticOperator", module = "noobase._core.photometry")]
pub struct PySyntheticOperator {
    inner: SyntheticOperatorInner,
}

#[pymethods]
impl PySyntheticOperator {
    /// Construct an operator from a spectrum wavelength axis and a
    /// transmission curve.
    ///
    /// All arguments are keyword-only. The dtype channel (``float32`` or
    /// ``float64``) is determined by ``spectrum_grid``; ``transmission_values``
    /// and ``transmission_grid`` must match.
    ///
    /// Parameters
    /// ----------
    /// spectrum_grid : Grid
    ///     Spectrum wavelength axis. Must be a Grid (an ndarray is not
    ///     accepted here because there is no paired array at construction
    ///     time from which to infer centers vs edges).
    /// transmission_grid : Grid or ndarray
    ///     Filter wavelength axis. If an ndarray is passed, its length
    ///     must equal ``len(transmission_values)`` (centers) or
    ///     ``len(transmission_values) + 1`` (edges).
    /// transmission_values : ndarray
    ///     Filter transmission per bin. Determines the operator's dtype
    ///     paired with ``spectrum_grid``.
    /// convention : {"photon_counting", "energy_weighted"}, optional
    ///     Photon-counting (default) is appropriate for photon-counting
    ///     detectors such as CCDs and HgCdTe arrays (including JWST
    ///     NIRCam). Energy-weighted is appropriate for bolometric /
    ///     energy-integrating detectors.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``spectrum_grid`` is not a Grid, dtypes mismatch across
    ///     inputs, array lengths are inconsistent, ``convention`` is
    ///     invalid, or the spectrum grid does not overlap the filter in
    ///     wavelength.
    #[new]
    #[pyo3(signature = (*, spectrum_grid, transmission_grid, transmission_values, convention="photon_counting"))]
    #[pyo3(
        text_signature = "(*, spectrum_grid, transmission_grid, transmission_values, convention=\"photon_counting\")"
    )]
    fn new(
        spectrum_grid: &Bound<'_, PyAny>,
        transmission_grid: &Bound<'_, PyAny>,
        transmission_values: &Bound<'_, PyAny>,
        convention: &str,
    ) -> PyResult<Self> {
        let convention_enum = parse_convention(convention)?;

        // spectrum_grid MUST be a Grid: there is no paired array at
        // construction time from which to infer centers vs edges.
        let spectrum_grid_py = spectrum_grid.extract::<PyGrid>().map_err(|_| {
            PyValueError::new_err(
                "SyntheticOperator requires spectrum_grid to be a Grid (ndarray is not supported \
                 here because there is no paired array to disambiguate centers vs edges)",
            )
        })?;

        let inner = with_inner!(&spectrum_grid_py.inner, GridInner, grid => operator_new_impl(
            grid,
            transmission_grid,
            transmission_values,
            convention_enum,
        )?);
        Ok(Self { inner })
    }

    /// Apply the operator to a spectrum flux array.
    ///
    /// Returns the band-averaged flux density and, optionally, the
    /// propagated 1-sigma uncertainty. The geometric coverage is fixed at
    /// construction time and accessible via the ``coverage`` property; it
    /// is not returned here.
    ///
    /// Parameters
    /// ----------
    /// spectrum_flux : ndarray
    ///     Flux density per bin. Length must equal the operator's spectrum
    ///     bin count and dtype must match the operator's dtype.
    /// spectrum_error : ndarray, optional
    ///     1-sigma uncertainty per bin. Same length and dtype as
    ///     ``spectrum_flux``. Default is ``None``.
    ///
    /// Returns
    /// -------
    /// tuple of (float, float or None)
    ///     ``(band_flux, band_error)``. ``band_error`` is ``None`` when
    ///     ``spectrum_error`` was not provided.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If dtypes mismatch with the operator, or if array lengths do not
    ///     match the operator's spectrum bin count.
    ///
    /// Notes
    /// -----
    /// As with ``photometry.synthetic``, the error is propagated assuming
    /// spectrum bins are statistically independent; the assumption is
    /// violated for spectra that were previously upsampled via
    /// ``Spectrum.rebin``. See ``Spectrum.rebin`` for the same caveat.
    #[pyo3(signature = (spectrum_flux, *, spectrum_error=None))]
    #[pyo3(text_signature = "(self, spectrum_flux, *, spectrum_error=None)")]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        spectrum_flux: &Bound<'py, PyAny>,
        spectrum_error: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        with_inner!(&self.inner, SyntheticOperatorInner, operator =>
            apply_impl(py, operator, spectrum_flux, spectrum_error))
    }

    /// Geometric coverage of the filter by the operator's spectrum grid.
    ///
    /// Returns
    /// -------
    /// float
    ///     Fraction of the filter's transmission integral probed by the
    ///     spectrum grid, in ``[0, 1]``. A value below 1.0 means the
    ///     spectrum grid does not span the full filter and band fluxes
    ///     produced by ``apply`` are biased low. SED-fitting callers should
    ///     threshold on this (for example, require coverage > 0.999).
    #[getter]
    fn coverage(&self) -> f64 {
        with_inner!(&self.inner, SyntheticOperatorInner, operator => operator.coverage().promote())
    }

    /// The numpy dtype of the operator's cached weights.
    ///
    /// Returns
    /// -------
    /// numpy.dtype
    ///     Either ``numpy.dtype('float32')`` or ``numpy.dtype('float64')``.
    ///     Arrays passed to ``apply`` must match this dtype.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            SyntheticOperatorInner::F32(_) => dtype::<f32>(py),
            SyntheticOperatorInner::F64(_) => dtype::<f64>(py),
        }
    }
}

fn operator_new_impl<T: Scalar + GridChannel>(
    spectrum_grid: &CoreGrid<T>,
    transmission_grid: &Bound<'_, PyAny>,
    transmission_values: &Bound<'_, PyAny>,
    convention: PhotometryConvention,
) -> PyResult<SyntheticOperatorInner>
where
    core_photometry::SyntheticOperator<T>: IntoOperatorInner,
{
    let transmission =
        required_typed_array1::<T>(transmission_values, "spectrum_grid vs transmission_values")?;
    let transmission_grid_py = coerce_to_grid(
        transmission_grid,
        transmission.len(),
        T::IS_F32,
        "transmission",
    )?;
    let transmission_core = grid_channel::<T>(transmission_grid_py, "transmission")
        .expect("coerce_to_grid enforces dtype");
    let operator = core_photometry::SyntheticOperator::<T>::new(
        spectrum_grid,
        &transmission_core,
        transmission.view(),
        convention,
    )
    .map_err(to_value_error)?;
    Ok(operator.into_operator_inner())
}

fn apply_impl<'py, T: Scalar>(
    py: Python<'py>,
    operator: &core_photometry::SyntheticOperator<T>,
    spectrum_flux: &Bound<'py, PyAny>,
    spectrum_error: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let flux = typed_channel::<T>(spectrum_flux, "spectrum_flux", "operator dtype")?;
    if flux.len() != operator.spectrum_bin_count() {
        return Err(PyValueError::new_err(format!(
            "spectrum_flux length {} does not match operator's spectrum bin count {}",
            flux.len(),
            operator.spectrum_bin_count()
        )));
    }
    let error = optional_typed_channel::<T>(spectrum_error, "spectrum_error", "operator dtype")?;
    if let Some(array) = error.as_ref()
        && array.len() != operator.spectrum_bin_count()
    {
        return Err(PyValueError::new_err(format!(
            "spectrum_error length {} does not match operator's spectrum bin count {}",
            array.len(),
            operator.spectrum_bin_count()
        )));
    }
    let error_view = error.as_ref().map(|array| array.view());
    let applied = operator.apply(flux.view(), error_view);
    band_pair(
        py,
        applied.band_flux.promote(),
        applied.band_error.map(|value| value.promote()),
    )
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let photometry = PyModule::new(py, "noobase._core.photometry")?;
    photometry.setattr("__package__", "noobase._core")?;
    photometry.add_class::<PySyntheticOperator>()?;
    photometry.add_function(wrap_pyfunction!(synthetic_function, &photometry)?)?;
    parent.add_submodule(&photometry)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.photometry", &photometry)?;
    Ok(())
}

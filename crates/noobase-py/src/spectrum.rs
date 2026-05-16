use ::noobase::photometry as core_photometry;
use ::noobase::spectroscopy::Spectrum as CoreSpectrum;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, ToPyArray, dtype};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::grid::{GridInner, PyGrid};
use crate::helpers::{
    build_grid_from_any, coerce_to_grid, dtype_mismatch_error, grid_dtype_name, parse_convention,
    parse_kind, parse_spacing,
};

pub(crate) enum SpectrumInner {
    F32(CoreSpectrum<f32>),
    F64(CoreSpectrum<f64>),
}

/// A 1-D spectrum: a wavelength Grid plus per-bin flux, optional 1-sigma
/// error, and optional mask.
///
/// The mask convention is ``True = invalid`` (a ``True`` entry marks the bin
/// as masked / excluded), matching astropy's masked-array convention. All
/// per-bin arrays share the wavelength Grid's
/// dtype (``float32`` or ``float64``); ``flux`` determines the channel and
/// every other input must match it.
///
/// Use ``rebin`` to resample onto a new wavelength axis, ``to_f_nu`` /
/// ``to_f_lambda`` to convert between flux density conventions, and
/// ``synthetic_photometry`` to integrate through a transmission curve.
#[pyclass(name = "Spectrum", module = "noobase._core", skip_from_py_object)]
pub struct PySpectrum {
    inner: SpectrumInner,
}

impl PySpectrum {
    fn from_inner(inner: SpectrumInner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySpectrum {
    /// Construct a Spectrum from a wavelength axis and per-bin arrays.
    ///
    /// All arguments are keyword-only. The ``flux`` dtype (``float32`` or
    /// ``float64``) determines the Spectrum's dtype; every other array
    /// argument and the wavelength Grid must match.
    ///
    /// Parameters
    /// ----------
    /// wavelength : Grid or ndarray
    ///     Wavelength axis. If a Grid is passed, ``spacing`` and ``kind``
    ///     must not be supplied. If an ndarray is passed, the Grid is built
    ///     internally using the given ``spacing`` and ``kind`` (defaulting
    ///     to ``"linear"`` and ``"centers"``).
    /// flux : ndarray
    ///     1-D flux density per bin. Length must equal the wavelength bin
    ///     count: ``len(wavelength)`` for ``kind="centers"`` or
    ///     ``len(wavelength) - 1`` for ``kind="edges"``.
    /// error : ndarray, optional
    ///     1-sigma uncertainty per bin. Same length and dtype as ``flux``.
    ///     Default is ``None``.
    /// mask : ndarray of bool, optional
    ///     Per-bin mask flag (``True = invalid``: a ``True`` entry marks the
    ///     bin as masked / excluded, matching astropy's convention). Same
    ///     length as ``flux``. Default is ``None``.
    /// spacing : {"linear", "log"}, optional
    ///     Spacing convention for the wavelength Grid when ``wavelength`` is
    ///     an ndarray. Must be omitted when ``wavelength`` is a Grid.
    /// kind : {"centers", "edges"}, optional
    ///     Bin convention for the wavelength Grid when ``wavelength`` is an
    ///     ndarray. Must be omitted when ``wavelength`` is a Grid.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If dtypes mismatch across inputs, array lengths are inconsistent
    ///     with the wavelength bin count, ``spacing``/``kind`` are passed
    ///     together with a Grid wavelength, or the wavelength values fail
    ///     the Grid invariants (strictly increasing, positive under log).
    #[new]
    #[pyo3(signature = (*, wavelength, flux, error=None, mask=None, spacing=None, kind=None))]
    #[pyo3(
        text_signature = "(*, wavelength, flux, error=None, mask=None, spacing=None, kind=None)"
    )]
    fn new(
        wavelength: &Bound<'_, PyAny>,
        flux: &Bound<'_, PyAny>,
        error: Option<&Bound<'_, PyAny>>,
        mask: Option<&Bound<'_, PyAny>>,
        spacing: Option<&str>,
        kind: Option<&str>,
    ) -> PyResult<Self> {
        let wavelength_grid = if let Ok(grid) = wavelength.extract::<PyGrid>() {
            if spacing.is_some() || kind.is_some() {
                return Err(PyValueError::new_err(
                    "spacing/kind must not be passed when wavelength is a Grid",
                ));
            }
            grid
        } else {
            let spacing_str = spacing.unwrap_or("linear");
            let kind_str = kind.unwrap_or("centers");
            let spacing_enum = parse_spacing(spacing_str)?;
            let kind_enum = parse_kind(kind_str)?;
            let inner = build_grid_from_any(wavelength, spacing_enum, kind_enum)?;
            PyGrid::from_inner(inner)
        };

        // Flux determines the spectrum's dtype; wavelength and error must match.
        if let Ok(flux_f64) = flux.extract::<PyReadonlyArray1<'_, f64>>() {
            let wavelength_f64 = match &wavelength_grid.inner {
                GridInner::F64(grid) => grid.clone(),
                GridInner::F32(_) => {
                    return Err(dtype_mismatch_error(
                        "float32",
                        "float64",
                        "wavelength vs flux",
                    ));
                }
            };
            let flux_array: Array1<f64> = flux_f64.as_array().to_owned();
            let error_array = extract_optional_array::<f64>(error, "error", "float64")?;
            let mask_array = extract_optional_mask(mask)?;
            let spectrum =
                CoreSpectrum::<f64>::new(wavelength_f64, flux_array, error_array, mask_array)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Ok(Self::from_inner(SpectrumInner::F64(spectrum)))
        } else if let Ok(flux_f32) = flux.extract::<PyReadonlyArray1<'_, f32>>() {
            let wavelength_f32 = match &wavelength_grid.inner {
                GridInner::F32(grid) => grid.clone(),
                GridInner::F64(_) => {
                    return Err(dtype_mismatch_error(
                        "float64",
                        "float32",
                        "wavelength vs flux",
                    ));
                }
            };
            let flux_array: Array1<f32> = flux_f32.as_array().to_owned();
            let error_array = extract_optional_array::<f32>(error, "error", "float32")?;
            let mask_array = extract_optional_mask(mask)?;
            let spectrum =
                CoreSpectrum::<f32>::new(wavelength_f32, flux_array, error_array, mask_array)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Ok(Self::from_inner(SpectrumInner::F32(spectrum)))
        } else {
            Err(PyValueError::new_err(
                "flux must be a 1-D numpy array of dtype float32 or float64",
            ))
        }
    }

    /// The wavelength axis.
    ///
    /// Returns
    /// -------
    /// Grid
    ///     A clone of the wavelength Grid. dtype matches the Spectrum's
    ///     dtype.
    #[getter]
    fn wavelength(&self) -> PyGrid {
        let inner = match &self.inner {
            SpectrumInner::F32(spectrum) => GridInner::F32(spectrum.wavelength().clone()),
            SpectrumInner::F64(spectrum) => GridInner::F64(spectrum.wavelength().clone()),
        };
        PyGrid::from_inner(inner)
    }

    /// The flux density array.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A new copy of the per-bin flux density. dtype matches the
    ///     Spectrum's dtype.
    #[getter]
    fn flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.flux().to_pyarray(py).into_any(),
            SpectrumInner::F64(spectrum) => spectrum.flux().to_pyarray(py).into_any(),
        }
    }

    /// The 1-sigma uncertainty array, if present.
    ///
    /// Returns
    /// -------
    /// ndarray or None
    ///     A new copy of the per-bin 1-sigma uncertainty, or ``None`` if the
    ///     Spectrum was constructed without an error array. dtype matches
    ///     the Spectrum's dtype.
    #[getter]
    fn error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        match &self.inner {
            SpectrumInner::F32(spectrum) => {
                spectrum.error().map(|view| view.to_pyarray(py).into_any())
            }
            SpectrumInner::F64(spectrum) => {
                spectrum.error().map(|view| view.to_pyarray(py).into_any())
            }
        }
    }

    /// The per-bin mask, if present.
    ///
    /// Returns
    /// -------
    /// ndarray of bool or None
    ///     A new copy of the per-bin mask flags (``True = invalid``: a
    ///     ``True`` entry marks the bin as masked / excluded, matching
    ///     astropy's masked-array convention), or ``None`` if no mask was
    ///     supplied at construction.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        let mask_view = match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.mask(),
            SpectrumInner::F64(spectrum) => spectrum.mask(),
        };
        mask_view.map(|view| view.to_pyarray(py).into_any())
    }

    /// Number of bins in the spectrum.
    ///
    /// Returns
    /// -------
    /// int
    ///     The length of the ``flux`` array, equivalently the wavelength
    ///     Grid's bin count.
    #[getter]
    fn n_bins(&self) -> usize {
        match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.n_bins(),
            SpectrumInner::F64(spectrum) => spectrum.n_bins(),
        }
    }

    /// The numpy dtype of the Spectrum's arrays.
    ///
    /// Returns
    /// -------
    /// numpy.dtype
    ///     Either ``numpy.dtype('float32')`` or ``numpy.dtype('float64')``.
    ///     All per-bin arrays (flux, error, wavelength values) share this
    ///     dtype.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            SpectrumInner::F32(_) => dtype::<f32>(py),
            SpectrumInner::F64(_) => dtype::<f64>(py),
        }
    }

    /// Resample the spectrum onto a target wavelength axis.
    ///
    /// Flux is propagated via overlap-weighted averaging (density-conserving:
    /// the integral of ``flux * bin_width`` is preserved over the region of
    /// overlap). Error, if present, is propagated by squaring to variance,
    /// applying the same overlap operator assuming independent source bins,
    /// and taking the square root; the result is the marginal 1-sigma per
    /// target bin. Mask, if present, is propagated as logical OR (``True =
    /// invalid``): a target bin is invalid iff any source bin with non-zero
    /// overlap into it is invalid.
    ///
    /// Parameters
    /// ----------
    /// target : Grid or ndarray
    ///     Target wavelength axis. If an ndarray is passed, ``spacing`` and
    ///     ``kind`` must be supplied (each defaults to ``"linear"`` /
    ///     ``"centers"`` when omitted but the dtype is read from the array).
    ///     dtype must match the Spectrum's dtype.
    /// spacing : {"linear", "log"}, optional
    ///     Spacing convention for building a Grid from ``target``. Must be
    ///     omitted when ``target`` is already a Grid.
    /// kind : {"centers", "edges"}, optional
    ///     Bin convention for building a Grid from ``target``. Must be
    ///     omitted when ``target`` is already a Grid.
    ///
    /// Returns
    /// -------
    /// Spectrum
    ///     A new Spectrum on the target wavelength axis.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``target`` dtype does not match the Spectrum's dtype, or if
    ///     ``spacing``/``kind`` are passed together with a Grid ``target``.
    ///
    /// Notes
    /// -----
    /// When the target is finer than the source (upsampling), neighboring
    /// target bins drawing from the same source bin are strongly correlated.
    /// The per-bin sigma values returned here are still individually correct
    /// as marginals, but downstream operations that assume independent bins
    /// (for example summing under quadrature, or ``synthetic_photometry`` on
    /// the upsampled spectrum) will underestimate the true uncertainty. See
    /// ``Spectrum.synthetic_photometry`` and ``photometry.synthetic`` for
    /// the same caveat.
    #[pyo3(signature = (target, *, spacing=None, kind=None))]
    #[pyo3(text_signature = "(self, target, *, spacing=None, kind=None)")]
    fn rebin(
        &self,
        target: &Bound<'_, PyAny>,
        spacing: Option<&str>,
        kind: Option<&str>,
    ) -> PyResult<PySpectrum> {
        let target_grid = if let Ok(grid) = target.extract::<PyGrid>() {
            if spacing.is_some() || kind.is_some() {
                return Err(PyValueError::new_err(
                    "spacing/kind must not be passed when target is a Grid",
                ));
            }
            grid
        } else {
            let spacing_str = spacing.unwrap_or("linear");
            let kind_str = kind.unwrap_or("centers");
            let spacing_enum = parse_spacing(spacing_str)?;
            let kind_enum = parse_kind(kind_str)?;
            let inner = build_grid_from_any(target, spacing_enum, kind_enum)?;
            PyGrid::from_inner(inner)
        };

        let output_inner = match (&self.inner, &target_grid.inner) {
            (SpectrumInner::F64(spectrum), GridInner::F64(target_core)) => {
                SpectrumInner::F64(spectrum.rebin(target_core))
            }
            (SpectrumInner::F32(spectrum), GridInner::F32(target_core)) => {
                SpectrumInner::F32(spectrum.rebin(target_core))
            }
            (spectrum_inner, target_inner) => {
                let spectrum_dtype = match spectrum_inner {
                    SpectrumInner::F32(_) => "float32",
                    SpectrumInner::F64(_) => "float64",
                };
                let target_dtype = grid_dtype_name(target_inner);
                return Err(dtype_mismatch_error(
                    spectrum_dtype,
                    target_dtype,
                    "spectrum vs target",
                ));
            }
        };
        Ok(PySpectrum::from_inner(output_inner))
    }

    /// Convert flux density from f_lambda to f_nu.
    ///
    /// Applies ``f_nu = f_lambda * lambda^2 / c`` per bin, using each bin's
    /// center wavelength. The wavelength Grid and mask are preserved; the
    /// error array, if present, scales by the same factor element-wise.
    ///
    /// Parameters
    /// ----------
    /// speed_of_light : float
    ///     Speed of light expressed in the wavelength axis's length unit per
    ///     second. For example, with the wavelength in angstroms pass
    ///     ``2.998e18`` (Å/s). noobase does not track units; consistency is
    ///     the caller's responsibility.
    ///
    /// Returns
    /// -------
    /// Spectrum
    ///     A new Spectrum with the converted flux density.
    ///
    /// Notes
    /// -----
    /// This method does not check whether the input is in fact f_lambda; it
    /// unconditionally applies the conversion factor. The caller is
    /// responsible for tracking which density the Spectrum currently
    /// represents.
    #[pyo3(text_signature = "(self, speed_of_light)")]
    fn to_f_nu(&self, speed_of_light: f64) -> PySpectrum {
        let inner = match &self.inner {
            SpectrumInner::F64(spectrum) => SpectrumInner::F64(spectrum.to_f_nu(speed_of_light)),
            // Accept f64 from Python and cast to f32 to match the spectrum's
            // dtype. This is a single scalar, not array data, so the cast does
            // not affect bulk allocation or vectorization.
            SpectrumInner::F32(spectrum) => {
                SpectrumInner::F32(spectrum.to_f_nu(speed_of_light as f32))
            }
        };
        PySpectrum::from_inner(inner)
    }

    /// Convert flux density from f_nu to f_lambda.
    ///
    /// Applies ``f_lambda = f_nu * c / lambda^2`` per bin, using each bin's
    /// center wavelength. The wavelength Grid and mask are preserved; the
    /// error array, if present, scales by the same factor element-wise.
    ///
    /// Parameters
    /// ----------
    /// speed_of_light : float
    ///     Speed of light expressed in the wavelength axis's length unit per
    ///     second. For example, with the wavelength in angstroms pass
    ///     ``2.998e18`` (Å/s). noobase does not track units; consistency is
    ///     the caller's responsibility.
    ///
    /// Returns
    /// -------
    /// Spectrum
    ///     A new Spectrum with the converted flux density.
    ///
    /// Notes
    /// -----
    /// This method does not check whether the input is in fact f_nu; it
    /// unconditionally applies the conversion factor. The caller is
    /// responsible for tracking which density the Spectrum currently
    /// represents.
    #[pyo3(text_signature = "(self, speed_of_light)")]
    fn to_f_lambda(&self, speed_of_light: f64) -> PySpectrum {
        let inner = match &self.inner {
            SpectrumInner::F64(spectrum) => {
                SpectrumInner::F64(spectrum.to_f_lambda(speed_of_light))
            }
            // Accept f64 from Python and cast to f32 to match the spectrum's
            // dtype. This is a single scalar, not array data, so the cast does
            // not affect bulk allocation or vectorization.
            SpectrumInner::F32(spectrum) => {
                SpectrumInner::F32(spectrum.to_f_lambda(speed_of_light as f32))
            }
        };
        PySpectrum::from_inner(inner)
    }

    /// Compute synthetic photometry through a transmission curve.
    ///
    /// Convenience wrapper around ``photometry.synthetic`` that reuses the
    /// Spectrum's wavelength, flux, and (optional) error. Returns the
    /// band-averaged flux density, the propagated 1-sigma uncertainty (only
    /// when the Spectrum has an error array), and the geometric coverage of
    /// the filter by the spectrum.
    ///
    /// Parameters
    /// ----------
    /// transmission_grid : Grid or ndarray
    ///     Filter wavelength axis. If an ndarray is passed, its length must
    ///     equal ``len(transmission_values)`` (centers) or
    ///     ``len(transmission_values) + 1`` (edges); spacing is assumed
    ///     linear and ``kind`` is inferred from the length match. For
    ///     log-spaced filters, pass a pre-built Grid. dtype must match the
    ///     Spectrum's dtype.
    /// transmission_values : ndarray
    ///     Filter transmission per bin. dtype must match the Spectrum's
    ///     dtype.
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
    ///     when the Spectrum has no error array. ``coverage`` is the
    ///     fraction of the filter's transmission integral probed by the
    ///     spectrum (1.0 means full coverage; a value below 1.0 means the
    ///     spectrum does not span the full filter and ``band_flux`` is
    ///     biased low).
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
    /// independent. The transmission curve's bin density does not affect
    /// this assumption — it only enters the deterministic weights. The
    /// assumption is violated if the spectrum was previously upsampled (for
    /// example via ``Spectrum.rebin`` onto a finer grid), in which case the
    /// returned ``band_error`` underestimates the true uncertainty. See
    /// ``Spectrum.rebin`` for the same caveat.
    #[pyo3(signature = (*, transmission_grid, transmission_values, convention="photon_counting"))]
    #[pyo3(
        text_signature = "(self, *, transmission_grid, transmission_values, convention=\"photon_counting\")"
    )]
    fn synthetic_photometry<'py>(
        &self,
        py: Python<'py>,
        transmission_grid: &Bound<'py, PyAny>,
        transmission_values: &Bound<'py, PyAny>,
        convention: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let convention_enum = parse_convention(convention)?;
        match &self.inner {
            SpectrumInner::F64(spectrum) => {
                let transmission_readonly = transmission_values
                    .extract::<PyReadonlyArray1<'_, f64>>()
                    .map_err(|_| {
                        dtype_mismatch_error("float64", "other", "spectrum vs transmission_values")
                    })?;
                let transmission_owned: Array1<f64> = transmission_readonly.as_array().to_owned();
                let transmission_length = transmission_owned.len();
                let transmission_grid_py = coerce_to_grid(
                    transmission_grid,
                    transmission_length,
                    false,
                    "transmission",
                )?;
                let transmission_core = match &transmission_grid_py.inner {
                    GridInner::F64(grid) => grid,
                    GridInner::F32(_) => unreachable!("coerce_to_grid enforces dtype"),
                };
                let result = core_photometry::synthetic::<f64>(
                    spectrum.wavelength(),
                    spectrum.flux(),
                    spectrum.error(),
                    transmission_core,
                    transmission_owned.view(),
                    convention_enum,
                )
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let tuple = PyTuple::new(
                    py,
                    [
                        result.band_flux.into_py_any(py)?,
                        match result.band_error {
                            Some(value) => value.into_py_any(py)?,
                            None => py.None(),
                        },
                        result.coverage.into_py_any(py)?,
                    ],
                )?;
                Ok(tuple.into_any())
            }
            SpectrumInner::F32(spectrum) => {
                let transmission_readonly = transmission_values
                    .extract::<PyReadonlyArray1<'_, f32>>()
                    .map_err(|_| {
                        dtype_mismatch_error("float32", "other", "spectrum vs transmission_values")
                    })?;
                let transmission_owned: Array1<f32> = transmission_readonly.as_array().to_owned();
                let transmission_length = transmission_owned.len();
                let transmission_grid_py =
                    coerce_to_grid(transmission_grid, transmission_length, true, "transmission")?;
                let transmission_core = match &transmission_grid_py.inner {
                    GridInner::F32(grid) => grid,
                    GridInner::F64(_) => unreachable!("coerce_to_grid enforces dtype"),
                };
                let result = core_photometry::synthetic::<f32>(
                    spectrum.wavelength(),
                    spectrum.flux(),
                    spectrum.error(),
                    transmission_core,
                    transmission_owned.view(),
                    convention_enum,
                )
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let band_flux: f64 = result.band_flux as f64;
                let coverage: f64 = result.coverage as f64;
                let band_error: Option<f64> = result.band_error.map(|value| value as f64);
                let tuple = PyTuple::new(
                    py,
                    [
                        band_flux.into_py_any(py)?,
                        match band_error {
                            Some(value) => value.into_py_any(py)?,
                            None => py.None(),
                        },
                        coverage.into_py_any(py)?,
                    ],
                )?;
                Ok(tuple.into_any())
            }
        }
    }

    fn __repr__(&self) -> String {
        let (n, dtype_name) = match &self.inner {
            SpectrumInner::F32(spectrum) => (spectrum.n_bins(), "float32"),
            SpectrumInner::F64(spectrum) => (spectrum.n_bins(), "float64"),
        };
        format!("Spectrum(n_bins={n}, dtype={dtype_name})")
    }
}

fn extract_optional_array<T>(
    value: Option<&Bound<'_, PyAny>>,
    name: &str,
    expected_dtype: &'static str,
) -> PyResult<Option<Array1<T>>>
where
    T: numpy::Element + Clone,
{
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray1<'_, T>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(dtype_mismatch_error(
            expected_dtype,
            "other",
            &format!("flux vs {name}"),
        )),
    }
}

fn extract_optional_mask(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Array1<bool>>> {
    let Some(bound) = value else {
        return Ok(None);
    };
    if bound.is_none() {
        return Ok(None);
    }
    match bound.extract::<PyReadonlyArray1<'_, bool>>() {
        Ok(array) => Ok(Some(array.as_array().to_owned())),
        Err(_) => Err(PyValueError::new_err(
            "mask must be a 1-D numpy array of dtype bool",
        )),
    }
}

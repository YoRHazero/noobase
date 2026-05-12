use ::noobase::spectroscopy::Spectrum as CoreSpectrum;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, ToPyArray, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid::{GridInner, PyGrid};
use crate::helpers::{
    build_grid_from_any, dtype_mismatch_error, grid_dtype_name, parse_kind, parse_spacing,
};

pub(crate) enum SpectrumInner {
    F32(CoreSpectrum<f32>),
    F64(CoreSpectrum<f64>),
}

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
    #[new]
    #[pyo3(signature = (*, wavelength, flux, error=None, mask=None, spacing=None, kind=None))]
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

    #[getter]
    fn wavelength(&self) -> PyGrid {
        let inner = match &self.inner {
            SpectrumInner::F32(spectrum) => GridInner::F32(spectrum.wavelength().clone()),
            SpectrumInner::F64(spectrum) => GridInner::F64(spectrum.wavelength().clone()),
        };
        PyGrid::from_inner(inner)
    }

    #[getter]
    fn flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.flux().to_pyarray(py).into_any(),
            SpectrumInner::F64(spectrum) => spectrum.flux().to_pyarray(py).into_any(),
        }
    }

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

    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        let mask_view = match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.mask(),
            SpectrumInner::F64(spectrum) => spectrum.mask(),
        };
        mask_view.map(|view| view.to_pyarray(py).into_any())
    }

    #[getter]
    fn n_bins(&self) -> usize {
        match &self.inner {
            SpectrumInner::F32(spectrum) => spectrum.n_bins(),
            SpectrumInner::F64(spectrum) => spectrum.n_bins(),
        }
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            SpectrumInner::F32(_) => dtype::<f32>(py),
            SpectrumInner::F64(_) => dtype::<f64>(py),
        }
    }

    #[pyo3(signature = (target, *, spacing=None, kind=None))]
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

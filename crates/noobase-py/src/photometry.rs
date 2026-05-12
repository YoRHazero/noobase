use ::noobase::photometry as core_photometry;
use ::noobase::photometry::PhotometryError;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, dtype};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::grid::{GridInner, PyGrid};
use crate::helpers::{coerce_to_grid, parse_convention};

fn map_photometry_error(err: PhotometryError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Discover whether the given array is f32 (true) or f64 (false). Returns an
/// error if it is neither.
fn detect_float_dtype_is_f32(value: &Bound<'_, PyAny>, role: &str) -> PyResult<bool> {
    if value.extract::<PyReadonlyArray1<'_, f64>>().is_ok() {
        Ok(false)
    } else if value.extract::<PyReadonlyArray1<'_, f32>>().is_ok() {
        Ok(true)
    } else {
        Err(PyValueError::new_err(format!(
            "{role} must be a 1-D numpy array of dtype float32 or float64"
        )))
    }
}

#[pyfunction]
#[pyo3(name = "synthetic")]
#[pyo3(signature = (*, spectrum_grid, spectrum_flux, spectrum_error=None, transmission_grid, transmission_values, convention="photon_counting"))]
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
    let flux_is_f32 = detect_float_dtype_is_f32(spectrum_flux, "spectrum_flux")?;

    if flux_is_f32 {
        let flux_readonly = spectrum_flux.extract::<PyReadonlyArray1<'_, f32>>()?;
        let flux_owned: Array1<f32> = flux_readonly.as_array().to_owned();
        let flux_length = flux_owned.len();

        let transmission_readonly = transmission_values
            .extract::<PyReadonlyArray1<'_, f32>>()
            .map_err(|_| {
                PyValueError::new_err(
                    "transmission_values dtype must match spectrum_flux (float32)",
                )
            })?;
        let transmission_owned: Array1<f32> = transmission_readonly.as_array().to_owned();
        let transmission_length = transmission_owned.len();

        let error_owned = match spectrum_error {
            Some(bound) if !bound.is_none() => {
                let arr = bound.extract::<PyReadonlyArray1<'_, f32>>().map_err(|_| {
                    PyValueError::new_err("spectrum_error dtype must match spectrum_flux (float32)")
                })?;
                Some(arr.as_array().to_owned())
            }
            _ => None,
        };

        let spectrum_grid_py = coerce_to_grid(spectrum_grid, flux_length, true, "spectrum")?;
        let transmission_grid_py =
            coerce_to_grid(transmission_grid, transmission_length, true, "transmission")?;
        let spectrum_core = match &spectrum_grid_py.inner {
            GridInner::F32(grid) => grid,
            GridInner::F64(_) => unreachable!("coerce_to_grid enforces dtype"),
        };
        let transmission_core = match &transmission_grid_py.inner {
            GridInner::F32(grid) => grid,
            GridInner::F64(_) => unreachable!("coerce_to_grid enforces dtype"),
        };

        let error_view = error_owned.as_ref().map(|arr| arr.view());
        let result = core_photometry::synthetic::<f32>(
            spectrum_core,
            flux_owned.view(),
            error_view,
            transmission_core,
            transmission_owned.view(),
            convention_enum,
        )
        .map_err(map_photometry_error)?;

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
    } else {
        let flux_readonly = spectrum_flux.extract::<PyReadonlyArray1<'_, f64>>()?;
        let flux_owned: Array1<f64> = flux_readonly.as_array().to_owned();
        let flux_length = flux_owned.len();

        let transmission_readonly = transmission_values
            .extract::<PyReadonlyArray1<'_, f64>>()
            .map_err(|_| {
                PyValueError::new_err(
                    "transmission_values dtype must match spectrum_flux (float64)",
                )
            })?;
        let transmission_owned: Array1<f64> = transmission_readonly.as_array().to_owned();
        let transmission_length = transmission_owned.len();

        let error_owned = match spectrum_error {
            Some(bound) if !bound.is_none() => {
                let arr = bound.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
                    PyValueError::new_err("spectrum_error dtype must match spectrum_flux (float64)")
                })?;
                Some(arr.as_array().to_owned())
            }
            _ => None,
        };

        let spectrum_grid_py = coerce_to_grid(spectrum_grid, flux_length, false, "spectrum")?;
        let transmission_grid_py = coerce_to_grid(
            transmission_grid,
            transmission_length,
            false,
            "transmission",
        )?;
        let spectrum_core = match &spectrum_grid_py.inner {
            GridInner::F64(grid) => grid,
            GridInner::F32(_) => unreachable!("coerce_to_grid enforces dtype"),
        };
        let transmission_core = match &transmission_grid_py.inner {
            GridInner::F64(grid) => grid,
            GridInner::F32(_) => unreachable!("coerce_to_grid enforces dtype"),
        };

        let error_view = error_owned.as_ref().map(|arr| arr.view());
        let result = core_photometry::synthetic::<f64>(
            spectrum_core,
            flux_owned.view(),
            error_view,
            transmission_core,
            transmission_owned.view(),
            convention_enum,
        )
        .map_err(map_photometry_error)?;

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
}

pub(crate) enum SyntheticOperatorInner {
    F32(core_photometry::SyntheticOperator<f32>),
    F64(core_photometry::SyntheticOperator<f64>),
}

#[pyclass(name = "SyntheticOperator", module = "noobase._core.photometry")]
pub struct PySyntheticOperator {
    inner: SyntheticOperatorInner,
}

#[pymethods]
impl PySyntheticOperator {
    #[new]
    #[pyo3(signature = (*, spectrum_grid, transmission_grid, transmission_values, convention="photon_counting"))]
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
        let spectrum_is_f32 = matches!(spectrum_grid_py.inner, GridInner::F32(_));

        let transmission_is_f32 =
            detect_float_dtype_is_f32(transmission_values, "transmission_values")?;
        if spectrum_is_f32 != transmission_is_f32 {
            let spec_dtype = if spectrum_is_f32 {
                "float32"
            } else {
                "float64"
            };
            let trans_dtype = if transmission_is_f32 {
                "float32"
            } else {
                "float64"
            };
            return Err(PyValueError::new_err(format!(
                "dtype mismatch: spectrum_grid vs transmission_values ({spec_dtype} vs {trans_dtype}); align dtypes explicitly"
            )));
        }

        if spectrum_is_f32 {
            let transmission_readonly =
                transmission_values.extract::<PyReadonlyArray1<'_, f32>>()?;
            let transmission_owned: Array1<f32> = transmission_readonly.as_array().to_owned();
            let transmission_length = transmission_owned.len();
            let transmission_grid_py =
                coerce_to_grid(transmission_grid, transmission_length, true, "transmission")?;
            let spectrum_core = match &spectrum_grid_py.inner {
                GridInner::F32(grid) => grid,
                GridInner::F64(_) => unreachable!(),
            };
            let transmission_core = match &transmission_grid_py.inner {
                GridInner::F32(grid) => grid,
                GridInner::F64(_) => unreachable!(),
            };
            let operator = core_photometry::SyntheticOperator::<f32>::new(
                spectrum_core,
                transmission_core,
                transmission_owned.view(),
                convention_enum,
            )
            .map_err(map_photometry_error)?;
            Ok(Self {
                inner: SyntheticOperatorInner::F32(operator),
            })
        } else {
            let transmission_readonly =
                transmission_values.extract::<PyReadonlyArray1<'_, f64>>()?;
            let transmission_owned: Array1<f64> = transmission_readonly.as_array().to_owned();
            let transmission_length = transmission_owned.len();
            let transmission_grid_py = coerce_to_grid(
                transmission_grid,
                transmission_length,
                false,
                "transmission",
            )?;
            let spectrum_core = match &spectrum_grid_py.inner {
                GridInner::F64(grid) => grid,
                GridInner::F32(_) => unreachable!(),
            };
            let transmission_core = match &transmission_grid_py.inner {
                GridInner::F64(grid) => grid,
                GridInner::F32(_) => unreachable!(),
            };
            let operator = core_photometry::SyntheticOperator::<f64>::new(
                spectrum_core,
                transmission_core,
                transmission_owned.view(),
                convention_enum,
            )
            .map_err(map_photometry_error)?;
            Ok(Self {
                inner: SyntheticOperatorInner::F64(operator),
            })
        }
    }

    #[pyo3(signature = (spectrum_flux, *, spectrum_error=None))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        spectrum_flux: &Bound<'py, PyAny>,
        spectrum_error: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            SyntheticOperatorInner::F64(operator) => {
                let flux_readonly = spectrum_flux
                    .extract::<PyReadonlyArray1<'_, f64>>()
                    .map_err(|_| {
                        PyValueError::new_err(
                            "spectrum_flux dtype must match operator dtype (float64)",
                        )
                    })?;
                let flux_owned: Array1<f64> = flux_readonly.as_array().to_owned();
                if flux_owned.len() != operator.spectrum_bin_count() {
                    return Err(PyValueError::new_err(format!(
                        "spectrum_flux length {} does not match operator's spectrum bin count {}",
                        flux_owned.len(),
                        operator.spectrum_bin_count()
                    )));
                }
                let error_owned = match spectrum_error {
                    Some(bound) if !bound.is_none() => {
                        let arr = bound.extract::<PyReadonlyArray1<'_, f64>>().map_err(|_| {
                            PyValueError::new_err(
                                "spectrum_error dtype must match operator dtype (float64)",
                            )
                        })?;
                        let owned: Array1<f64> = arr.as_array().to_owned();
                        if owned.len() != operator.spectrum_bin_count() {
                            return Err(PyValueError::new_err(format!(
                                "spectrum_error length {} does not match operator's spectrum bin count {}",
                                owned.len(),
                                operator.spectrum_bin_count()
                            )));
                        }
                        Some(owned)
                    }
                    _ => None,
                };
                let error_view = error_owned.as_ref().map(|arr| arr.view());
                let applied = operator.apply(flux_owned.view(), error_view);
                let tuple = PyTuple::new(
                    py,
                    [
                        applied.band_flux.into_py_any(py)?,
                        match applied.band_error {
                            Some(value) => value.into_py_any(py)?,
                            None => py.None(),
                        },
                    ],
                )?;
                Ok(tuple.into_any())
            }
            SyntheticOperatorInner::F32(operator) => {
                let flux_readonly = spectrum_flux
                    .extract::<PyReadonlyArray1<'_, f32>>()
                    .map_err(|_| {
                        PyValueError::new_err(
                            "spectrum_flux dtype must match operator dtype (float32)",
                        )
                    })?;
                let flux_owned: Array1<f32> = flux_readonly.as_array().to_owned();
                if flux_owned.len() != operator.spectrum_bin_count() {
                    return Err(PyValueError::new_err(format!(
                        "spectrum_flux length {} does not match operator's spectrum bin count {}",
                        flux_owned.len(),
                        operator.spectrum_bin_count()
                    )));
                }
                let error_owned = match spectrum_error {
                    Some(bound) if !bound.is_none() => {
                        let arr = bound.extract::<PyReadonlyArray1<'_, f32>>().map_err(|_| {
                            PyValueError::new_err(
                                "spectrum_error dtype must match operator dtype (float32)",
                            )
                        })?;
                        let owned: Array1<f32> = arr.as_array().to_owned();
                        if owned.len() != operator.spectrum_bin_count() {
                            return Err(PyValueError::new_err(format!(
                                "spectrum_error length {} does not match operator's spectrum bin count {}",
                                owned.len(),
                                operator.spectrum_bin_count()
                            )));
                        }
                        Some(owned)
                    }
                    _ => None,
                };
                let error_view = error_owned.as_ref().map(|arr| arr.view());
                let applied = operator.apply(flux_owned.view(), error_view);
                // Promote f32 outputs to f64 for the Python tuple so the
                // surface is uniformly Python `float`.
                let band_flux: f64 = applied.band_flux as f64;
                let band_error: Option<f64> = applied.band_error.map(|value| value as f64);
                let tuple = PyTuple::new(
                    py,
                    [
                        band_flux.into_py_any(py)?,
                        match band_error {
                            Some(value) => value.into_py_any(py)?,
                            None => py.None(),
                        },
                    ],
                )?;
                Ok(tuple.into_any())
            }
        }
    }

    #[getter]
    fn coverage(&self) -> f64 {
        match &self.inner {
            SyntheticOperatorInner::F32(operator) => operator.coverage() as f64,
            SyntheticOperatorInner::F64(operator) => operator.coverage(),
        }
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            SyntheticOperatorInner::F32(_) => dtype::<f32>(py),
            SyntheticOperatorInner::F64(_) => dtype::<f64>(py),
        }
    }
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let photometry = PyModule::new(py, "photometry")?;
    photometry.add_class::<PySyntheticOperator>()?;
    photometry.add_function(wrap_pyfunction!(synthetic_function, &photometry)?)?;
    parent.add_submodule(&photometry)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.photometry", &photometry)?;
    Ok(())
}

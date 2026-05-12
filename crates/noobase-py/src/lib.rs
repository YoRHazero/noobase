use ::noobase::bins::overlap as core_overlap;
use ::noobase::spectroscopy::Spectrum as CoreSpectrum;
use ::noobase::{Grid as CoreGrid, GridKind, Spacing};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArrayDescr, PyReadonlyArray1, ToPyArray, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_spacing(spacing: &str) -> PyResult<Spacing> {
    match spacing {
        "linear" => Ok(Spacing::Linear),
        "log" => Ok(Spacing::Log),
        other => Err(PyValueError::new_err(format!(
            "invalid spacing {other:?}; expected one of \"linear\", \"log\""
        ))),
    }
}

fn parse_kind(kind: &str) -> PyResult<GridKind> {
    match kind {
        "centers" => Ok(GridKind::Centers),
        "edges" => Ok(GridKind::Edges),
        other => Err(PyValueError::new_err(format!(
            "invalid kind {other:?}; expected one of \"centers\", \"edges\""
        ))),
    }
}

fn spacing_to_str(spacing: Spacing) -> &'static str {
    match spacing {
        Spacing::Linear => "linear",
        Spacing::Log => "log",
    }
}

fn kind_to_str(kind: GridKind) -> &'static str {
    match kind {
        GridKind::Centers => "centers",
        GridKind::Edges => "edges",
    }
}

// ---------------------------------------------------------------------------
// Grid binding
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum GridInner {
    F32(CoreGrid<f32>),
    F64(CoreGrid<f64>),
}

#[pyclass(name = "Grid", module = "noobase._core", from_py_object)]
#[derive(Clone)]
pub struct PyGrid {
    inner: GridInner,
}

impl PyGrid {
    fn from_inner(inner: GridInner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGrid {
    #[new]
    #[pyo3(signature = (values, *, spacing="linear", kind="centers"))]
    fn new(values: &Bound<'_, PyAny>, spacing: &str, kind: &str) -> PyResult<Self> {
        let spacing_enum = parse_spacing(spacing)?;
        let kind_enum = parse_kind(kind)?;
        let inner = build_grid_from_any(values, spacing_enum, kind_enum)?;
        Ok(Self::from_inner(inner))
    }

    #[classmethod]
    #[pyo3(signature = (start, end, n, *, kind="centers", dtype=None))]
    fn linspace<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        start: f64,
        end: f64,
        n: usize,
        kind: &str,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let kind_enum = parse_kind(kind)?;
        if n < 2 {
            return Err(PyValueError::new_err("linspace requires n >= 2"));
        }
        let use_f32 = is_float32_dtype(py, dtype)?;
        let inner = if use_f32 {
            GridInner::F32(CoreGrid::<f32>::linspace(
                start as f32,
                end as f32,
                n,
                kind_enum,
            ))
        } else {
            GridInner::F64(CoreGrid::<f64>::linspace(start, end, n, kind_enum))
        };
        Ok(Self::from_inner(inner))
    }

    #[classmethod]
    #[pyo3(signature = (start, end, n, *, kind="centers", dtype=None))]
    fn logspace<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        start: f64,
        end: f64,
        n: usize,
        kind: &str,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let kind_enum = parse_kind(kind)?;
        if n < 2 {
            return Err(PyValueError::new_err("logspace requires n >= 2"));
        }
        if !(start > 0.0 && end > 0.0) {
            return Err(PyValueError::new_err(
                "logspace requires strictly positive endpoints",
            ));
        }
        let use_f32 = is_float32_dtype(py, dtype)?;
        let inner = if use_f32 {
            GridInner::F32(CoreGrid::<f32>::logspace(
                start as f32,
                end as f32,
                n,
                kind_enum,
            ))
        } else {
            GridInner::F64(CoreGrid::<f64>::logspace(start, end, n, kind_enum))
        };
        Ok(Self::from_inner(inner))
    }

    #[classmethod]
    #[pyo3(signature = (values, *, rel_tol=1e-9, kind="centers"))]
    fn from_array(
        _cls: &Bound<'_, PyType>,
        values: &Bound<'_, PyAny>,
        rel_tol: f64,
        kind: &str,
    ) -> PyResult<Self> {
        let kind_enum = parse_kind(kind)?;
        if let Ok(array_f64) = values.extract::<PyReadonlyArray1<'_, f64>>() {
            let owned: Array1<f64> = array_f64.as_array().to_owned();
            let grid = CoreGrid::<f64>::from_array(owned, rel_tol, kind_enum)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Ok(Self::from_inner(GridInner::F64(grid)))
        } else if let Ok(array_f32) = values.extract::<PyReadonlyArray1<'_, f32>>() {
            let owned: Array1<f32> = array_f32.as_array().to_owned();
            let grid = CoreGrid::<f32>::from_array(owned, rel_tol as f32, kind_enum)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Ok(Self::from_inner(GridInner::F32(grid)))
        } else {
            Err(PyValueError::new_err(
                "values must be a 1-D numpy array of dtype float32 or float64",
            ))
        }
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.inner {
            GridInner::F32(grid) => grid.values().to_pyarray(py).into_any(),
            GridInner::F64(grid) => grid.values().to_pyarray(py).into_any(),
        }
    }

    #[getter]
    fn spacing(&self) -> &'static str {
        let value = match &self.inner {
            GridInner::F32(grid) => grid.spacing(),
            GridInner::F64(grid) => grid.spacing(),
        };
        spacing_to_str(value)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        let value = match &self.inner {
            GridInner::F32(grid) => grid.kind(),
            GridInner::F64(grid) => grid.kind(),
        };
        kind_to_str(value)
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            GridInner::F32(_) => dtype::<f32>(py),
            GridInner::F64(_) => dtype::<f64>(py),
        }
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            GridInner::F32(grid) => grid.len(),
            GridInner::F64(grid) => grid.len(),
        }
    }

    fn to_edges(&self) -> PyGrid {
        let inner = match &self.inner {
            GridInner::F32(grid) => GridInner::F32(grid.to_edges()),
            GridInner::F64(grid) => GridInner::F64(grid.to_edges()),
        };
        PyGrid::from_inner(inner)
    }

    fn to_centers(&self) -> PyGrid {
        let inner = match &self.inner {
            GridInner::F32(grid) => GridInner::F32(grid.to_centers()),
            GridInner::F64(grid) => GridInner::F64(grid.to_centers()),
        };
        PyGrid::from_inner(inner)
    }

    #[pyo3(signature = (rel_tol=1e-9))]
    fn is_uniform(&self, rel_tol: f64) -> bool {
        match &self.inner {
            GridInner::F32(grid) => grid.is_uniform(rel_tol as f32),
            GridInner::F64(grid) => grid.is_uniform(rel_tol),
        }
    }

    fn __repr__(&self) -> String {
        let (n, spacing, kind, dtype_name) = match &self.inner {
            GridInner::F32(grid) => (
                grid.len(),
                spacing_to_str(grid.spacing()),
                kind_to_str(grid.kind()),
                "float32",
            ),
            GridInner::F64(grid) => (
                grid.len(),
                spacing_to_str(grid.spacing()),
                kind_to_str(grid.kind()),
                "float64",
            ),
        };
        format!("Grid(len={n}, spacing={spacing:?}, kind={kind:?}, dtype={dtype_name})")
    }
}

fn build_grid_from_any(
    values: &Bound<'_, PyAny>,
    spacing: Spacing,
    kind: GridKind,
) -> PyResult<GridInner> {
    if let Ok(array_f64) = values.extract::<PyReadonlyArray1<'_, f64>>() {
        let owned: Array1<f64> = array_f64.as_array().to_owned();
        let grid = CoreGrid::<f64>::new(owned, spacing, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(GridInner::F64(grid))
    } else if let Ok(array_f32) = values.extract::<PyReadonlyArray1<'_, f32>>() {
        let owned: Array1<f32> = array_f32.as_array().to_owned();
        let grid = CoreGrid::<f32>::new(owned, spacing, kind)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(GridInner::F32(grid))
    } else {
        Err(PyValueError::new_err(
            "values must be a 1-D numpy array of dtype float32 or float64",
        ))
    }
}

fn is_float32_dtype(py: Python<'_>, dtype_arg: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(value) = dtype_arg else {
        return Ok(false);
    };
    if value.is_none() {
        return Ok(false);
    }
    let numpy_module = py.import("numpy")?;
    let numpy_dtype_factory = numpy_module.getattr("dtype")?;
    let resolved = numpy_dtype_factory.call1((value,))?;
    let resolved_name: String = resolved.getattr("name")?.extract()?;
    match resolved_name.as_str() {
        "float32" => Ok(true),
        "float64" => Ok(false),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype {other:?}; expected float32 or float64"
        ))),
    }
}

// ---------------------------------------------------------------------------
// overlap submodule functions
// ---------------------------------------------------------------------------

fn dtype_mismatch_error(left: &'static str, right: &'static str, context: &str) -> PyErr {
    PyValueError::new_err(format!(
        "dtype mismatch: {context} ({left} vs {right}); align dtypes explicitly"
    ))
}

fn grid_dtype_name(inner: &GridInner) -> &'static str {
    match inner {
        GridInner::F32(_) => "float32",
        GridInner::F64(_) => "float64",
    }
}

#[pyfunction]
#[pyo3(name = "rebin")]
fn overlap_rebin<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    source_values: &Bound<'py, PyAny>,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let array = source_values
                .extract::<PyReadonlyArray1<'_, f64>>()
                .map_err(|_| {
                    dtype_mismatch_error("float64", "values not float64", "source grid vs values")
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_values length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f64> = core_overlap::rebin(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let array = source_values
                .extract::<PyReadonlyArray1<'_, f32>>()
                .map_err(|_| {
                    dtype_mismatch_error("float32", "values not float32", "source grid vs values")
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_values length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f32> = core_overlap::rebin(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "rebin_variance")]
fn overlap_rebin_variance<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    source_variance: &Bound<'py, PyAny>,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let array = source_variance
                .extract::<PyReadonlyArray1<'_, f64>>()
                .map_err(|_| {
                    dtype_mismatch_error(
                        "float64",
                        "variance not float64",
                        "source grid vs variance",
                    )
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_variance length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f64> = core_overlap::rebin_variance(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let array = source_variance
                .extract::<PyReadonlyArray1<'_, f32>>()
                .map_err(|_| {
                    dtype_mismatch_error(
                        "float32",
                        "variance not float32",
                        "source grid vs variance",
                    )
                })?;
            let view = array.as_array();
            let source_bin_count = source.to_edges().len() - 1;
            if view.len() != source_bin_count {
                return Err(PyValueError::new_err(format!(
                    "source_variance length {} does not match source bin count {source_bin_count}",
                    view.len()
                )));
            }
            let output: Array1<f32> = core_overlap::rebin_variance(source, view, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "coverage")]
fn overlap_coverage<'py>(
    py: Python<'py>,
    source_grid: &PyGrid,
    target_grid: &PyGrid,
) -> PyResult<Bound<'py, PyAny>> {
    match (&source_grid.inner, &target_grid.inner) {
        (GridInner::F64(source), GridInner::F64(target)) => {
            let output: Array1<f64> = core_overlap::coverage(source, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (GridInner::F32(source), GridInner::F32(target)) => {
            let output: Array1<f32> = core_overlap::coverage(source, target);
            Ok(output.into_pyarray(py).into_any())
        }
        (left, right) => Err(dtype_mismatch_error(
            grid_dtype_name(left),
            grid_dtype_name(right),
            "source grid vs target grid",
        )),
    }
}

// ---------------------------------------------------------------------------
// Spectrum binding
// ---------------------------------------------------------------------------

enum SpectrumInner {
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

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrid>()?;
    module.add_class::<PySpectrum>()?;

    let overlap = PyModule::new(py, "overlap")?;
    overlap.add_function(wrap_pyfunction!(overlap_rebin, &overlap)?)?;
    overlap.add_function(wrap_pyfunction!(overlap_rebin_variance, &overlap)?)?;
    overlap.add_function(wrap_pyfunction!(overlap_coverage, &overlap)?)?;
    module.add_submodule(&overlap)?;

    // Register the submodule under sys.modules so `from noobase.overlap import ...`
    // and `noobase._core.overlap` both resolve to the same module object.
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.overlap", &overlap)?;
    sys_modules.set_item("noobase.overlap", &overlap)?;

    Ok(())
}

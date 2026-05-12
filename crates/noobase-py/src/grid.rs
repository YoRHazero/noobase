use ::noobase::Grid as CoreGrid;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, ToPyArray, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::helpers::{
    build_grid_from_any, is_float32_dtype, kind_to_str, parse_kind, parse_spacing, spacing_to_str,
};

#[derive(Clone)]
pub(crate) enum GridInner {
    F32(CoreGrid<f32>),
    F64(CoreGrid<f64>),
}

#[pyclass(name = "Grid", module = "noobase._core", from_py_object)]
#[derive(Clone)]
pub struct PyGrid {
    pub(crate) inner: GridInner,
}

impl PyGrid {
    pub(crate) fn from_inner(inner: GridInner) -> Self {
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

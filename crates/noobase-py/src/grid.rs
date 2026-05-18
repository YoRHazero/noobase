use ::noobase::Grid as CoreGrid;
use ndarray::Array1;
use numpy::{PyArrayDescr, PyReadonlyArray1, ToPyArray, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::convert::{
    build_grid_from_any, is_float32_dtype, kind_to_str, map_inner, parse_kind, parse_spacing,
    spacing_to_str, with_inner,
};

#[derive(Clone)]
pub(crate) enum GridInner {
    F32(CoreGrid<f32>),
    F64(CoreGrid<f64>),
}

/// A 1-D monotonically increasing wavelength (or generic axis) grid.
///
/// Carries two orthogonal flags: ``spacing`` (linear vs log midpoint
/// convention) and ``kind`` (whether values represent bin centers or bin
/// edges). A Grid with ``kind="centers"`` and ``N`` values describes ``N``
/// bins; with ``kind="edges"`` and ``N`` values, ``N - 1`` bins. Use
/// ``to_edges()`` and ``to_centers()`` to convert between the two
/// representations.
///
/// Values are stored as either ``float32`` or ``float64``; the dtype is fixed
/// at construction and exposed via the ``dtype`` property. Most downstream
/// operations (``Spectrum.rebin``, ``photometry.synthetic``) require dtypes
/// to match across all inputs.
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
    /// Construct a Grid from an explicit 1-D array of values.
    ///
    /// Parameters
    /// ----------
    /// values : ndarray
    ///     1-D array of strictly increasing wavelengths. dtype must be
    ///     ``float32`` or ``float64`` and determines the Grid's dtype.
    /// spacing : {"linear", "log"}, optional
    ///     Midpoint convention. Default is ``"linear"``. Log spacing requires
    ///     all values to be strictly positive; the convention only affects
    ///     center/edge interpolation, the stored values are not transformed.
    /// kind : {"centers", "edges"}, optional
    ///     Whether ``values`` are bin centers or bin edges. Default is
    ///     ``"centers"``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``values`` has fewer than 2 entries, is not strictly increasing,
    ///     contains non-positive entries under log spacing, has an
    ///     unsupported dtype, or if ``spacing``/``kind`` is not one of the
    ///     accepted literals.
    #[new]
    #[pyo3(signature = (values, *, spacing="linear", kind="centers"))]
    #[pyo3(text_signature = "(values, *, spacing=\"linear\", kind=\"centers\")")]
    fn new(values: &Bound<'_, PyAny>, spacing: &str, kind: &str) -> PyResult<Self> {
        let spacing_enum = parse_spacing(spacing)?;
        let kind_enum = parse_kind(kind)?;
        let inner = build_grid_from_any(values, spacing_enum, kind_enum)?;
        Ok(Self::from_inner(inner))
    }

    /// Construct a linearly spaced Grid on ``[start, end]``.
    ///
    /// Parameters
    /// ----------
    /// start : float
    ///     First value. Inclusive.
    /// end : float
    ///     Last value. Inclusive.
    /// n : int
    ///     Number of values. Must be at least 2.
    /// kind : {"centers", "edges"}, optional
    ///     Whether the values are bin centers or bin edges. Default is
    ///     ``"centers"``.
    /// dtype : numpy dtype, optional
    ///     Either ``numpy.float32`` or ``numpy.float64``. Default is
    ///     ``numpy.float64``.
    ///
    /// Returns
    /// -------
    /// Grid
    ///     A new Grid with ``spacing="linear"``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``n < 2`` or ``dtype`` is neither ``float32`` nor ``float64``.
    #[classmethod]
    #[pyo3(signature = (start, end, n, *, kind="centers", dtype=None))]
    #[pyo3(text_signature = "(start, end, n, *, kind=\"centers\", dtype=None)")]
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

    /// Construct a logarithmically spaced Grid on ``[start, end]``.
    ///
    /// The endpoints are the actual axis values (not their logarithms); they
    /// must both be strictly positive.
    ///
    /// Parameters
    /// ----------
    /// start : float
    ///     First value. Inclusive. Must be > 0.
    /// end : float
    ///     Last value. Inclusive. Must be > 0.
    /// n : int
    ///     Number of values. Must be at least 2.
    /// kind : {"centers", "edges"}, optional
    ///     Whether the values are bin centers or bin edges. Default is
    ///     ``"centers"``.
    /// dtype : numpy dtype, optional
    ///     Either ``numpy.float32`` or ``numpy.float64``. Default is
    ///     ``numpy.float64``.
    ///
    /// Returns
    /// -------
    /// Grid
    ///     A new Grid with ``spacing="log"``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``n < 2``, either endpoint is non-positive, or ``dtype`` is
    ///     neither ``float32`` nor ``float64``.
    #[classmethod]
    #[pyo3(signature = (start, end, n, *, kind="centers", dtype=None))]
    #[pyo3(text_signature = "(start, end, n, *, kind=\"centers\", dtype=None)")]
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

    /// Construct a Grid from values and auto-detect the spacing convention.
    ///
    /// If the values are strictly positive and uniform in log-space within
    /// ``rel_tol``, the spacing is set to ``"log"``; otherwise it falls back
    /// to ``"linear"``.
    ///
    /// Parameters
    /// ----------
    /// values : ndarray
    ///     1-D array of strictly increasing values. dtype must be ``float32``
    ///     or ``float64``.
    /// rel_tol : float, optional
    ///     Maximum allowed relative deviation of per-step spacing when
    ///     deciding between log and linear. Default is ``1e-9``.
    /// kind : {"centers", "edges"}, optional
    ///     Whether the values are bin centers or bin edges. Default is
    ///     ``"centers"``.
    ///
    /// Returns
    /// -------
    /// Grid
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``values`` has fewer than 2 entries, is not strictly increasing,
    ///     or has an unsupported dtype.
    #[classmethod]
    #[pyo3(signature = (values, *, rel_tol=1e-9, kind="centers"))]
    #[pyo3(text_signature = "(values, *, rel_tol=1e-9, kind=\"centers\")")]
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

    /// The underlying 1-D array of values.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     A new copy of the Grid's values. dtype matches the Grid's dtype.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        with_inner!(&self.inner, GridInner, grid => grid.values().to_pyarray(py).into_any())
    }

    /// The spacing (midpoint) convention.
    ///
    /// Returns
    /// -------
    /// str
    ///     Either ``"linear"`` or ``"log"``.
    #[getter]
    fn spacing(&self) -> &'static str {
        let value = with_inner!(&self.inner, GridInner, grid => grid.spacing());
        spacing_to_str(value)
    }

    /// Whether the values represent bin centers or bin edges.
    ///
    /// Returns
    /// -------
    /// str
    ///     Either ``"centers"`` or ``"edges"``.
    #[getter]
    fn kind(&self) -> &'static str {
        let value = with_inner!(&self.inner, GridInner, grid => grid.kind());
        kind_to_str(value)
    }

    /// The numpy dtype of the underlying values.
    ///
    /// Returns
    /// -------
    /// numpy.dtype
    ///     Either ``numpy.dtype('float32')`` or ``numpy.dtype('float64')``.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.inner {
            GridInner::F32(_) => dtype::<f32>(py),
            GridInner::F64(_) => dtype::<f64>(py),
        }
    }

    /// Number of values in the grid (NOT the number of bins, which depends on ``kind``).
    fn __len__(&self) -> usize {
        with_inner!(&self.inner, GridInner, grid => grid.len())
    }

    /// Return a new Grid expressed as bin edges.
    ///
    /// If the Grid is already ``kind="edges"``, this returns a clone. If it
    /// is ``kind="centers"``, interior edges are computed via the spacing's
    /// midpoint convention (arithmetic mean for linear, geometric mean for
    /// log) and the two outer edges are extrapolated half a bin out from the
    /// first/last pair.
    ///
    /// Returns
    /// -------
    /// Grid
    ///     A new Grid with ``kind="edges"`` and the same ``spacing`` and
    ///     dtype. Length is ``len(self) + 1`` when converting from centers,
    ///     otherwise unchanged.
    fn to_edges(&self) -> PyGrid {
        let inner = map_inner!(&self.inner, GridInner, grid => grid.to_edges());
        PyGrid::from_inner(inner)
    }

    /// Return a new Grid expressed as bin centers.
    ///
    /// If the Grid is already ``kind="centers"``, this returns a clone. If
    /// it is ``kind="edges"``, centers are computed via the spacing's
    /// midpoint convention (arithmetic mean for linear, geometric mean for
    /// log).
    ///
    /// Returns
    /// -------
    /// Grid
    ///     A new Grid with ``kind="centers"`` and the same ``spacing`` and
    ///     dtype. Length is ``len(self) - 1`` when converting from edges,
    ///     otherwise unchanged.
    ///
    /// Notes
    /// -----
    /// ``to_edges`` followed by ``to_centers`` is an exact round-trip only
    /// when the original Grid is strictly uniform under its spacing
    /// convention. For irregular Grids the conversion is lossy.
    fn to_centers(&self) -> PyGrid {
        let inner = map_inner!(&self.inner, GridInner, grid => grid.to_centers());
        PyGrid::from_inner(inner)
    }

    /// Test whether the values are uniform under the Grid's spacing convention.
    ///
    /// For ``spacing="linear"`` this checks uniform per-step differences; for
    /// ``spacing="log"`` it checks uniform per-step differences in ln-space.
    /// Useful for selecting analytic fast paths in downstream code.
    ///
    /// Parameters
    /// ----------
    /// rel_tol : float, optional
    ///     Maximum allowed relative deviation of per-step spacing from the
    ///     mean step. Default is ``1e-9``.
    ///
    /// Returns
    /// -------
    /// bool
    ///     ``True`` if the values are uniform within ``rel_tol``.
    #[pyo3(signature = (rel_tol=1e-9))]
    #[pyo3(text_signature = "(self, rel_tol=1e-9)")]
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

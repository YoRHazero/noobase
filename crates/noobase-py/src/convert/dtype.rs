//! dtype-channel dispatch.
//!
//! The whole binding accepts exactly two numpy float channels (f32 /
//! f64). This module owns the machinery that picks the channel once and
//! keeps every per-binding implementation generic:
//!
//! - [`Scalar`] -- the f32 / f64 marker carrying the dtype name used in
//!   boundary error messages, the `is_f32` channel flag, and the
//!   uniform f32 -> f64 promotion that keeps the Python scalar surface a
//!   plain `float`.
//! - [`dispatch_array!`] -- probe a numpy array argument and forward the
//!   *owned* array to a generic `impl`, else the canonical `ValueError`.
//! - [`with_inner!`] / [`map_inner!`] -- collapse the symmetric
//!   `match &self.inner { F32 => .., F64 => .. }` arms over an
//!   already-typed channel enum (`GridInner` / `SpectrumInner`).
//! - [`with_grid_pair!`] -- collapse the same-channel `GridInner` pair
//!   match shared by every `overlap` function.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::axis::GridInner;

/// The f32 / f64 channel marker: the only two numpy float dtypes the
/// binding accepts. Carries the dtype name used verbatim in boundary
/// error messages, the `IS_F32` channel flag (for the few core APIs
/// that take a `bool` dtype selector), and the uniform f32 -> f64
/// promotion applied to scalar outputs so the Python surface is always
/// a plain `float`.
pub(crate) trait Scalar: ::noobase::Float + numpy::Element + Copy + 'static {
    /// The numpy dtype name (`"float32"` / `"float64"`).
    const DTYPE_NAME: &'static str;

    /// Whether this channel is `float32` (the `bool` selector some core
    /// helpers accept).
    const IS_F32: bool;

    /// Promote a channel scalar to the f64 the Python surface exposes
    /// uniformly (identity for f64).
    fn promote(self) -> f64;
}

impl Scalar for f32 {
    const DTYPE_NAME: &'static str = "float32";
    const IS_F32: bool = true;
    fn promote(self) -> f64 {
        self as f64
    }
}

impl Scalar for f64 {
    const DTYPE_NAME: &'static str = "float64";
    const IS_F32: bool = false;
    fn promote(self) -> f64 {
        self
    }
}

/// Map any core `*Error` to a `ValueError` verbatim. Every core error is
/// a hard precondition whose `Display` is the authoritative message;
/// this is the noobase-py-wide convention.
pub(crate) fn to_value_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Build the canonical cross-input dtype-mismatch `ValueError` shared by
/// grid / spectrum / overlap / photometry.
pub(crate) fn dtype_mismatch_error(left: &str, right: &str, context: &str) -> PyErr {
    PyValueError::new_err(format!(
        "dtype mismatch: {context} ({left} vs {right}); align dtypes explicitly"
    ))
}

/// The numpy dtype name of a [`GridInner`] channel.
pub(crate) fn grid_dtype_name(inner: &GridInner) -> &'static str {
    match inner {
        GridInner::F32(_) => "float32",
        GridInner::F64(_) => "float64",
    }
}

/// Resolve an optional Python ``dtype=`` argument to the f32 (`true`) or
/// f64 (`false`) channel. `None` / Python `None` defaults to f64;
/// anything other than float32 / float64 is a `ValueError`.
pub(crate) fn is_float32_dtype(
    py: Python<'_>,
    dtype_arg: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
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

/// Probe a numpy float array argument and forward it to a generic
/// implementation.
///
/// Attempts to extract `$any` as an `$ndim`-D `float64` then `float32`
/// numpy array (preserving the historical extraction-based contract: a
/// non-array or unsupported dtype yields the canonical `ValueError`),
/// then calls `$imp::<T>` with the matched [`Scalar`] channel, passing
/// the *owned* array followed by any extra arguments.
///
/// `$ndim` must be the literal `1`, `2` or `3`.
macro_rules! dispatch_array {
    ($any:expr, 1, $role:expr, $imp:path $(, $rest:expr)* $(,)?) => {{
        let __any = $any;
        if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray1<'_, f64>>()
        {
            let __owned: ::ndarray::Array1<f64> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray1<'_, f32>>()
        {
            let __owned: ::ndarray::Array1<f32> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else {
            ::std::result::Result::Err(::pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 1-D numpy array of dtype float32 or float64",
                $role
            )))
        }
    }};
    ($any:expr, 2, $role:expr, $imp:path $(, $rest:expr)* $(,)?) => {{
        let __any = $any;
        if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray2<'_, f64>>()
        {
            let __owned: ::ndarray::Array2<f64> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray2<'_, f32>>()
        {
            let __owned: ::ndarray::Array2<f32> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else {
            ::std::result::Result::Err(::pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 2-D numpy array of dtype float32 or float64",
                $role
            )))
        }
    }};
    ($any:expr, 3, $role:expr, $imp:path $(, $rest:expr)* $(,)?) => {{
        let __any = $any;
        if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray3<'_, f64>>()
        {
            let __owned: ::ndarray::Array3<f64> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else if let ::std::result::Result::Ok(__a) =
            __any.extract::<::numpy::PyReadonlyArray3<'_, f32>>()
        {
            let __owned: ::ndarray::Array3<f32> = __a.as_array().to_owned();
            $imp(__owned $(, $rest)*)
        } else {
            ::std::result::Result::Err(::pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 3-D numpy array of dtype float32 or float64",
                $role
            )))
        }
    }};
}

/// Run `$body` for both channels of an already-typed inner enum (e.g.
/// `GridInner` / `SpectrumInner`), binding the channel value as `$g`.
/// For getter-style passthrough where `$body` is valid for both
/// channels.
macro_rules! with_inner {
    ($e:expr, $ty:ident, $g:ident => $body:expr) => {
        match $e {
            $ty::F32($g) => $body,
            $ty::F64($g) => $body,
        }
    };
}

/// Like [`with_inner!`] but re-wraps the body result back into the same
/// channel variant (e.g. `Grid::to_edges` preserving the dtype).
macro_rules! map_inner {
    ($e:expr, $ty:ident, $g:ident => $body:expr) => {
        match $e {
            $ty::F32($g) => $ty::F32($body),
            $ty::F64($g) => $ty::F64($body),
        }
    };
}

/// Dispatch a same-channel pair of [`GridInner`]s, binding the two
/// channel grids as `$ga` / `$gb`. A cross-channel pair yields the
/// canonical "source grid vs target grid" dtype-mismatch `ValueError`
/// (the form shared by every `overlap` function).
macro_rules! with_grid_pair {
    ($a:expr, $b:expr, ($ga:ident, $gb:ident) => $body:expr) => {
        match ($a, $b) {
            ($crate::axis::GridInner::F64($ga), $crate::axis::GridInner::F64($gb)) => $body,
            ($crate::axis::GridInner::F32($ga), $crate::axis::GridInner::F32($gb)) => $body,
            (__left, __right) => ::std::result::Result::Err($crate::convert::dtype_mismatch_error(
                $crate::convert::grid_dtype_name(__left),
                $crate::convert::grid_dtype_name(__right),
                "source grid vs target grid",
            )),
        }
    };
}

pub(crate) use dispatch_array;
pub(crate) use map_inner;
pub(crate) use with_grid_pair;
pub(crate) use with_inner;
// `to_value_error` is re-exported from `mod.rs` alongside the other
// items; declared here only because it sits with the error builders.

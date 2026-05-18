//! String-literal <-> core-enum parsing.
//!
//! The Python surface uses string literals instead of exposing the core
//! enums; this module is the single home for those `&str` -> enum
//! parsers and the reverse `enum -> &'static str` getters. An unknown
//! literal is always a `ValueError` naming the accepted set (the
//! convention shared crate-wide).

use ::noobase::photometry::PhotometryConvention;
use ::noobase::{GridKind, Spacing};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn parse_spacing(spacing: &str) -> PyResult<Spacing> {
    match spacing {
        "linear" => Ok(Spacing::Linear),
        "log" => Ok(Spacing::Log),
        other => Err(PyValueError::new_err(format!(
            "invalid spacing {other:?}; expected one of \"linear\", \"log\""
        ))),
    }
}

pub(crate) fn parse_kind(kind: &str) -> PyResult<GridKind> {
    match kind {
        "centers" => Ok(GridKind::Centers),
        "edges" => Ok(GridKind::Edges),
        other => Err(PyValueError::new_err(format!(
            "invalid kind {other:?}; expected one of \"centers\", \"edges\""
        ))),
    }
}

pub(crate) fn spacing_to_str(spacing: Spacing) -> &'static str {
    match spacing {
        Spacing::Linear => "linear",
        Spacing::Log => "log",
    }
}

pub(crate) fn kind_to_str(kind: GridKind) -> &'static str {
    match kind {
        GridKind::Centers => "centers",
        GridKind::Edges => "edges",
    }
}

pub(crate) fn parse_convention(value: &str) -> PyResult<PhotometryConvention> {
    match value {
        "photon_counting" => Ok(PhotometryConvention::PhotonCounting),
        "energy_weighted" => Ok(PhotometryConvention::EnergyWeighted),
        other => Err(PyValueError::new_err(format!(
            "invalid convention {other:?}; expected one of \"photon_counting\", \"energy_weighted\""
        ))),
    }
}

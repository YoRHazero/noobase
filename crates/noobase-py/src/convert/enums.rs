//! String-literal <-> core-enum parsing.
//!
//! The Python surface uses string literals instead of exposing the core
//! enums; this module is the single home for those `&str` -> enum
//! parsers and the reverse `enum -> &'static str` getters. An unknown
//! literal is always a `ValueError` naming the accepted set (the
//! convention shared crate-wide).

use ::noobase::convolve::{Boundary, Normalization};
use ::noobase::image::psf::{CombineMethod, ResidualReweight};
use ::noobase::spectroscopy::synthetic_photometry::PhotometryConvention;
use ::noobase::spectroscopy::LsfSpec;
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

/// Parse the flat `combine` string + companion scalars into the core
/// `CombineMethod`. `median` ignores the companion scalars;
/// `clipped_mean`'s `kappa`/`max_iter` are validated by the core, not
/// pre-checked here.
pub(crate) fn parse_combine_method(
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

pub(crate) fn parse_normalization(value: &str) -> PyResult<Normalization> {
    match value {
        "sum" => Ok(Normalization::Sum),
        "l2" => Ok(Normalization::L2),
        "none" => Ok(Normalization::None),
        other => Err(PyValueError::new_err(format!(
            "invalid normalization {other:?}; expected one of \"sum\", \"l2\", \"none\""
        ))),
    }
}

pub(crate) fn parse_boundary(value: &str) -> PyResult<Boundary> {
    match value {
        "zero" => Ok(Boundary::Zero),
        "reflect" => Ok(Boundary::Reflect),
        "nearest" => Ok(Boundary::Nearest),
        other => Err(PyValueError::new_err(format!(
            "invalid boundary {other:?}; expected one of \"zero\", \"reflect\", \"nearest\""
        ))),
    }
}

/// Parse the flat `spec` string + the companion scalars into the core
/// `LsfSpec`. Each spec consumes a disjoint subset of the companions; a
/// missing required companion is a `ValueError` naming it (positivity is
/// validated by the core, not pre-checked here).
pub(crate) fn parse_lsf_spec(
    spec: &str,
    resolving_power: Option<f64>,
    sigma: Option<f64>,
    speed_of_light: Option<f64>,
) -> PyResult<LsfSpec> {
    match spec {
        "constant_r" => {
            let resolving_power = resolving_power.ok_or_else(|| {
                PyValueError::new_err("resolving_power is required when spec=\"constant_r\"")
            })?;
            Ok(LsfSpec::ConstantR(resolving_power))
        }
        "constant_velocity" => {
            let sigma = sigma.ok_or_else(|| {
                PyValueError::new_err("sigma is required when spec=\"constant_velocity\"")
            })?;
            let speed_of_light = speed_of_light.ok_or_else(|| {
                PyValueError::new_err("speed_of_light is required when spec=\"constant_velocity\"")
            })?;
            Ok(LsfSpec::ConstantVelocitySigma {
                sigma,
                speed_of_light,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "invalid spec {other:?}; expected one of \"constant_r\", \"constant_velocity\""
        ))),
    }
}

/// Parse the flat `residual_reweight` string + companion `reweight_c`
/// into the core `ResidualReweight`. `none` ignores `reweight_c`;
/// `huber`/`tukey`'s `c` is validated by the core, not pre-checked here.
pub(crate) fn parse_residual_reweight(
    reweight: &str,
    reweight_c: f64,
) -> PyResult<ResidualReweight> {
    match reweight {
        "none" => Ok(ResidualReweight::None),
        "huber" => Ok(ResidualReweight::Huber { c: reweight_c }),
        "tukey" => Ok(ResidualReweight::Tukey { c: reweight_c }),
        other => Err(PyValueError::new_err(format!(
            "invalid residual_reweight {other:?}; expected one of \"none\", \"huber\", \"tukey\""
        ))),
    }
}

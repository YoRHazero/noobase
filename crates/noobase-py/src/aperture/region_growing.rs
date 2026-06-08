//! PyO3 binding for `aperture::region_growing::grow_mask`.
//!
//! The Rust core is f64-only and takes a structured `GrowthConfig` plus
//! the optional `LabelInput` pair; this binding flattens both into a
//! single `grow_mask` function with keyword arguments and a fully
//! explicit signature. The thin Python wrapper at
//! `noobase.aperture.region_growing` layers the policy defaults on top
//! (auto-enable SNR when `err` is provided, accept a numpy `(N, 2)`
//! array as `seed_pixels`, default `detection` to `data`, and scale the
//! dimensionless `shape_weight` by the image noise level).

use ::noobase::aperture::region_growing::{
    GradientStop, GrowthConfig, GrowthResult, LabelInput, SnrStop, StopCriterion, StopReason,
    grow_mask,
};
use numpy::{PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::to_value_error;

use super::parse_connectivity;

/// Result of `grow_mask`: the boolean source mask plus the termination
/// reason and the heap-loop iteration count.
#[pyclass(name = "GrowthResult", module = "noobase._core.aperture", frozen)]
pub struct PyGrowthResult {
    inner: GrowthResult,
}

#[pymethods]
impl PyGrowthResult {
    /// The boolean source mask, shape ``(rows, cols)`` matching the
    /// input ``data``.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.mask.to_pyarray(py).into_any()
    }

    /// Why the growth terminated. One of ``"snr_below"``,
    /// ``"gradient_flip"``, or ``"filled"`` — the last is the failure
    /// outcome (mask reached the cutout edge before any stop fired).
    #[getter]
    fn stop_reason(&self) -> &'static str {
        match self.inner.stop_reason {
            StopReason::SnrBelow => "snr_below",
            StopReason::GradientFlip => "gradient_flip",
            StopReason::Filled => "filled",
        }
    }

    /// Number of pixels admitted by the heap loop after the seeds. The
    /// total number of ``True`` pixels in ``mask`` equals
    /// ``n_iterations`` plus the (de-duplicated) seed count.
    #[getter]
    fn n_iterations(&self) -> usize {
        self.inner.n_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "GrowthResult(mask=({}, {}), stop_reason={:?}, n_iterations={})",
            self.inner.mask.shape()[0],
            self.inner.mask.shape()[1],
            self.stop_reason(),
            self.inner.n_iterations,
        )
    }
}

/// Grow a boolean source mask outward from one or more seed pixels.
///
/// Thin binding around the Rust core. See the pure-Python wrapper at
/// ``noobase.aperture.region_growing.grow_mask`` for the docstring
/// describing parameters, defaults, and the
/// ``snr_threshold``-follows-``err`` auto-enable rule. The signature
/// here is fully explicit: pass ``None`` to disable a stop criterion,
/// pass a number to enable it.
///
/// Raises
/// ------
/// ValueError
///     For any invalid input — out-of-bounds seed, mismatched array
///     shapes, label / allowed pairing, missing or dangling ``err``
///     relative to ``snr_threshold``, no stop criterion enabled, etc.
///     The message mirrors the core ``GrowError`` variant verbatim.
#[pyfunction]
#[pyo3(name = "grow_mask")]
#[pyo3(signature = (
    data,
    seed_pixels,
    *,
    detection = None,
    err = None,
    label_map = None,
    label_allowed = None,
    connectivity = "eight",
    shape_weight = 0.0,
    min_neighbor_support = 1,
    min_pixels_before_shape_gate = 0,
    snr_threshold = None,
    snr_hysteresis = 3,
    gradient_ratio_threshold = None,
    gradient_hysteresis = 3,
    gradient_lo_percentile = 75.0,
    gradient_hi_percentile = 99.0,
    min_pixels_before_stop_check = 30,
    check_interval = 5,
    annulus_thickness = 2,
))]
#[allow(clippy::too_many_arguments)]
fn grow_mask_function(
    data: PyReadonlyArray2<'_, f64>,
    seed_pixels: Vec<(usize, usize)>,
    detection: Option<PyReadonlyArray2<'_, f64>>,
    err: Option<PyReadonlyArray2<'_, f64>>,
    label_map: Option<PyReadonlyArray2<'_, i32>>,
    label_allowed: Option<Vec<i32>>,
    connectivity: &str,
    shape_weight: f64,
    min_neighbor_support: usize,
    min_pixels_before_shape_gate: usize,
    snr_threshold: Option<f64>,
    snr_hysteresis: usize,
    gradient_ratio_threshold: Option<f64>,
    gradient_hysteresis: usize,
    gradient_lo_percentile: f64,
    gradient_hi_percentile: f64,
    min_pixels_before_stop_check: usize,
    check_interval: usize,
    annulus_thickness: usize,
) -> PyResult<PyGrowthResult> {
    // The (label_map, label_allowed) pair is bidirectionally bound:
    // both or neither. This binding-layer check is mirrored in the
    // pure-Python wrapper, but is asserted here too because the
    // _core function is reachable directly via `noobase._core.aperture`.
    let label = match (label_map.as_ref(), label_allowed) {
        (None, None) => None,
        (Some(map), Some(allowed)) => Some(LabelInput {
            map: map.as_array(),
            allowed,
        }),
        (Some(_), None) | (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "label_map and label_allowed must be both provided or both None",
            ));
        }
    };

    let config = GrowthConfig {
        connectivity: parse_connectivity(connectivity)?,
        stop: StopCriterion {
            snr: snr_threshold.map(|threshold| SnrStop {
                threshold,
                hysteresis: snr_hysteresis,
            }),
            gradient: gradient_ratio_threshold.map(|ratio_threshold| GradientStop {
                ratio_threshold,
                hysteresis: gradient_hysteresis,
                lo_percentile: gradient_lo_percentile,
                hi_percentile: gradient_hi_percentile,
            }),
        },
        shape_weight,
        min_neighbor_support,
        min_pixels_before_shape_gate,
        min_pixels_before_stop_check,
        check_interval,
        annulus_thickness,
    };

    let data_view = data.as_array();
    // The detection image drives the heap priority; when omitted it
    // defaults to the science image (plain brightest-pixel growth).
    let detection_view = match detection.as_ref() {
        Some(detection) => detection.as_array(),
        None => data_view,
    };
    let err_view = err.as_ref().map(|e| e.as_array());
    let result = grow_mask(
        detection_view,
        data_view,
        err_view,
        label,
        &seed_pixels,
        &config,
    )
    .map_err(to_value_error)?;

    Ok(PyGrowthResult { inner: result })
}

pub(super) fn register_into(aperture: &Bound<'_, PyModule>) -> PyResult<()> {
    aperture.add_function(wrap_pyfunction!(grow_mask_function, aperture)?)?;
    aperture.add_class::<PyGrowthResult>()?;
    Ok(())
}

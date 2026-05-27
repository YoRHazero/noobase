//! PyO3 bindings for the aperture mask construction subsystem.
//!
//! The Python surface is flattened: even though the Rust core nests
//! algorithms under `aperture::region_growing`, the Python user sees
//! `noobase.aperture.grow_mask` (and `noobase.aperture.GrowthResult`).
//! The `region_growing` re-export at the pure-Python layer mirrors the
//! core structure for callers who want it.

mod region_growing;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::noobase::aperture::region_growing::Connectivity;

/// Parse the connectivity string used on the Python surface.
pub(super) fn parse_connectivity(value: &str) -> PyResult<Connectivity> {
    match value {
        "four" => Ok(Connectivity::Four),
        "eight" => Ok(Connectivity::Eight),
        other => Err(PyValueError::new_err(format!(
            "connectivity must be \"four\" or \"eight\"; got {other:?}"
        ))),
    }
}

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let aperture = PyModule::new(py, "noobase._core.aperture")?;
    aperture.setattr("__package__", "noobase._core")?;
    region_growing::register_into(&aperture)?;
    parent.add_submodule(&aperture)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.aperture", &aperture)?;
    Ok(())
}

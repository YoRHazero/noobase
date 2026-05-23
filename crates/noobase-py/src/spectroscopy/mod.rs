//! `noobase._core.spectroscopy` — spectrum container plus the
//! `synthetic_photometry` nested submodule.
//!
//! Mirrors the Rust core `noobase::spectroscopy` layout: `Spectrum`
//! lives at the submodule level, and `synthetic_photometry` is a nested
//! submodule for the `synthetic` / `SyntheticOperator` surface.

mod spectrum;
pub mod synthetic_photometry;

use pyo3::prelude::*;

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let spectroscopy = PyModule::new(py, "noobase._core.spectroscopy")?;
    spectroscopy.setattr("__package__", "noobase._core")?;
    spectrum::register_into(&spectroscopy)?;
    synthetic_photometry::build_submodule(py, &spectroscopy)?;
    parent.add_submodule(&spectroscopy)?;

    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.spectroscopy", &spectroscopy)?;

    Ok(())
}

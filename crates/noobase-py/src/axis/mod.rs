//! `noobase._core.axis` — Grid type and overlap-based bin reductions.
//!
//! Mirrors the Rust core `noobase::axis` module: the [`PyGrid`] type
//! lives at the submodule level, and `axis.overlap` holds the
//! `rebin` / `rebin_variance` / `coverage` reductions.

mod grid;
pub mod overlap;

pub(crate) use grid::{GridInner, PyGrid};

use pyo3::prelude::*;

pub(crate) fn build_submodule<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>) -> PyResult<()> {
    let axis = PyModule::new(py, "noobase._core.axis")?;
    axis.setattr("__package__", "noobase._core")?;
    axis.add_class::<PyGrid>()?;
    overlap::build_submodule(py, &axis)?;
    parent.add_submodule(&axis)?;

    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("noobase._core.axis", &axis)?;

    Ok(())
}

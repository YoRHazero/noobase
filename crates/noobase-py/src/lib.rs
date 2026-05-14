use pyo3::prelude::*;

mod grid;
mod helpers;
mod image;
mod overlap;
mod photometry;
mod spectrum;

use crate::grid::PyGrid;
use crate::spectrum::PySpectrum;

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrid>()?;
    module.add_class::<PySpectrum>()?;
    overlap::build_submodule(py, module)?;
    photometry::build_submodule(py, module)?;
    image::build_submodule(py, module)?;
    Ok(())
}

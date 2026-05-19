use pyo3::prelude::*;

mod convert;
mod grid;
mod image;
mod overlap;
mod photometry;
mod psf;
mod spectrum;

use crate::grid::PyGrid;

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrid>()?;
    spectrum::build_submodule(py, module)?;
    overlap::build_submodule(py, module)?;
    photometry::build_submodule(py, module)?;
    image::build_submodule(py, module)?;
    Ok(())
}

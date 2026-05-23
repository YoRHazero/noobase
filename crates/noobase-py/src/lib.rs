use pyo3::prelude::*;

mod axis;
mod convert;
mod image;
mod photometry;
mod psf;
mod spectrum;

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    axis::build_submodule(py, module)?;
    spectrum::build_submodule(py, module)?;
    photometry::build_submodule(py, module)?;
    image::build_submodule(py, module)?;
    Ok(())
}

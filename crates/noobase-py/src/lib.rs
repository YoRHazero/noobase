use pyo3::prelude::*;

mod axis;
mod convert;
mod convolve;
mod image;
mod spectroscopy;

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    axis::build_submodule(py, module)?;
    convolve::build_submodule(py, module)?;
    spectroscopy::build_submodule(py, module)?;
    image::build_submodule(py, module)?;
    Ok(())
}

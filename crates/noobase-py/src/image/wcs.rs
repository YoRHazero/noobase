//! PyO3 binding for `image::wcs` -- the compiled WCS transform program.
//!
//! The Python side hands in a plain JSON-able *spec* dict (produced by a
//! compiler that walks a gwcs / astropy model tree); this module parses it
//! once into a [`Program`] and evaluates it on numpy arrays with the GIL
//! released.

use ::noobase::image::wcs::{
    BinaryKind, GrismAxis, GrismOrderBwd, GrismOrderFwd, GwaAxis, GwaStep, LabelKey,
    LogicalCondition, Op, OpInstr, Program, TPoly,
};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// A compiled coordinate-transform program.
///
/// Construct from a spec dict (see ``noobase.image.wcs`` for the schema),
/// then call it with numpy arrays (any shared shape) or python floats.
#[pyclass(name = "WcsProgram", module = "noobase._core.image", frozen)]
pub struct PyWcsProgram {
    inner: Program,
}

#[pymethods]
impl PyWcsProgram {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let inner = parse_program(spec)?;
        Ok(Self { inner })
    }

    /// Number of input arrays the program expects.
    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.n_inputs()
    }

    /// Number of output arrays the program produces.
    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs()
    }

    /// Evaluate the program.
    ///
    /// Accepts either all python scalars (returns a tuple of floats, no
    /// numpy round-trip) or numpy ``float64`` arrays of one shared shape
    /// (returns a tuple of arrays of that shape). The GIL is released
    /// while array math runs.
    #[pyo3(signature = (*inputs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // Scalar fast path: every argument extracts as f64.
        let mut scalars: Vec<f64> = Vec::with_capacity(inputs.len());
        let mut all_scalar = true;
        for item in inputs.iter() {
            match item.extract::<f64>() {
                Ok(v) => scalars.push(v),
                Err(_) => {
                    all_scalar = false;
                    break;
                }
            }
        }
        if all_scalar {
            let out = self.inner.eval_scalar(&scalars).map_err(to_value_error)?;
            return PyTuple::new(py, out);
        }

        let arrays: Vec<PyReadonlyArrayDyn<'py, f64>> = inputs
            .iter()
            .map(|item| {
                item.extract::<PyReadonlyArrayDyn<'py, f64>>().map_err(|_| {
                    PyTypeError::new_err(
                        "WcsProgram expects float64 numpy arrays (or all scalars)",
                    )
                })
            })
            .collect::<PyResult<_>>()?;
        let shape = arrays
            .first()
            .ok_or_else(|| PyValueError::new_err("WcsProgram called with no inputs"))?
            .as_array()
            .shape()
            .to_vec();

        // Flatten to standard-layout slices, copying (in logical order) only
        // when the array is not C-contiguous.
        let views: Vec<_> = arrays.iter().map(|a| a.as_array()).collect();
        let owned: Vec<Option<Vec<f64>>> = views
            .iter()
            .map(|v| match v.to_slice() {
                Some(_) => None,
                None => Some(v.iter().copied().collect()),
            })
            .collect();
        let slices: Vec<&[f64]> = views
            .iter()
            .zip(&owned)
            .map(|(v, copy)| match copy {
                Some(buffer) => buffer.as_slice(),
                None => v.to_slice().expect("standard layout checked above"),
            })
            .collect();

        let result = py
            .detach(|| self.inner.eval(&slices))
            .map_err(to_value_error)?;

        let out: Vec<Bound<'py, PyArrayDyn<f64>>> = result
            .into_iter()
            .map(|v| {
                v.to_pyarray(py)
                    .reshape(shape.clone())
                    .expect("output length equals input length")
            })
            .collect();
        PyTuple::new(py, out)
    }

    fn __repr__(&self) -> String {
        format!(
            "WcsProgram(n_inputs={}, n_outputs={})",
            self.inner.n_inputs(),
            self.inner.n_outputs()
        )
    }
}

fn to_value_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_program(spec: &Bound<'_, PyDict>) -> PyResult<Program> {
    let n_regs: usize = required(spec, "n_regs")?;
    let inputs: Vec<u16> = required(spec, "inputs")?;
    let outputs: Vec<u16> = required(spec, "outputs")?;
    let ops_list: Bound<'_, PyList> = required(spec, "ops")?;

    let mut ops = Vec::with_capacity(ops_list.len());
    for (index, item) in ops_list.iter().enumerate() {
        let d = item.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!("ops[{index}] is not a dict"))
        })?;
        ops.push(parse_op(&d, index)?);
    }
    Program::new(n_regs, inputs, outputs, ops).map_err(to_value_error)
}

fn parse_op(d: &Bound<'_, PyDict>, index: usize) -> PyResult<OpInstr> {
    let kind: String = required(d, "op")?;
    let inputs: Vec<u16> = required(d, "in")?;
    let outputs: Vec<u16> = required(d, "out")?;
    let op = match kind.as_str() {
        "shift" => Op::Shift {
            offset: required(d, "offset")?,
        },
        "scale" => Op::Scale {
            factor: required(d, "factor")?,
        },
        "const" => Op::Const {
            value: required(d, "value")?,
        },
        "poly1d" => Op::Poly1d {
            coeffs: required(d, "coeffs")?,
        },
        "poly2d" => Op::Poly2d {
            degree: required(d, "degree")?,
            coeffs: required(d, "coeffs")?,
        },
        "affine2" => Op::Affine2 {
            matrix: required(d, "matrix")?,
            translation: required(d, "translation")?,
        },
        "sph2cart" => Op::SphToCart,
        "cart2sph" => Op::CartToSph {
            wrap_lon_at: required(d, "wrap_lon_at")?,
        },
        "rot3" => Op::Rot3 {
            matrix: required(d, "matrix")?,
        },
        "tan_project" => Op::TanProject,
        "tan_deproject" => Op::TanDeproject,
        "grism_forward" => {
            let axis: String = required(d, "axis")?;
            let axis = match axis.as_str() {
                "row" => GrismAxis::Row,
                "column" => GrismAxis::Column,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "ops[{index}]: axis must be 'row' or 'column', got {other:?}"
                    )));
                }
            };
            let alongdisp: Vec<Bound<'_, PyDict>> = required(d, "alongdisp")?;
            let lmodels: Vec<Bound<'_, PyDict>> = required(d, "lmodels")?;
            let models = alongdisp
                .iter()
                .zip(&lmodels)
                .map(|(a, l)| {
                    Ok(GrismOrderFwd {
                        alongdisp: parse_tpoly(a, index)?,
                        lmodel: parse_tpoly(l, index)?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Op::GrismForward {
                axis,
                orders: required(d, "orders")?,
                models,
            }
        }
        "grism_backward" => {
            let lmodels: Vec<Bound<'_, PyDict>> = required(d, "lmodels")?;
            let xmodels: Vec<Bound<'_, PyDict>> = required(d, "xmodels")?;
            let ymodels: Vec<Bound<'_, PyDict>> = required(d, "ymodels")?;
            let models = lmodels
                .iter()
                .zip(&xmodels)
                .zip(&ymodels)
                .map(|((l, x), y)| {
                    Ok(GrismOrderBwd {
                        lmodel: parse_tpoly(l, index)?,
                        xmodel: parse_tpoly(x, index)?,
                        ymodel: parse_tpoly(y, index)?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Op::GrismBackward {
                orders: required(d, "orders")?,
                models,
            }
        }
        "binary" => {
            let kind: String = required(d, "kind")?;
            let kind = match kind.as_str() {
                "add" => BinaryKind::Add,
                "sub" => BinaryKind::Sub,
                "mul" => BinaryKind::Mul,
                "div" => BinaryKind::Div,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "ops[{index}]: binary kind must be add/sub/mul/div, got {other:?}"
                    )));
                }
            };
            Op::Binary { kind }
        }
        "rot3_gwa" => {
            let steps: Vec<Bound<'_, PyDict>> = required(d, "steps")?;
            let steps = steps
                .iter()
                .map(|step| {
                    let axis: String = required(step, "axis")?;
                    let axis = match axis.as_str() {
                        "x" => GwaAxis::X,
                        "y" => GwaAxis::Y,
                        "z" => GwaAxis::Z,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "ops[{index}]: rot3_gwa axis must be x/y/z, got {other:?}"
                            )));
                        }
                    };
                    let angle: f64 = required(step, "angle")?;
                    let (sin, cos) = angle.sin_cos();
                    Ok(GwaStep { axis, cos, sin })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Op::Rot3Gwa { steps }
        }
        "unitless2dircos" => Op::Unitless2DirCos,
        "dircos2unitless" => Op::DirCos2Unitless,
        "grating_wavelength" => Op::GratingWavelength {
            factor: required(d, "factor")?,
        },
        "grating_angles3d" => Op::GratingAngles3D {
            factor: required(d, "factor")?,
        },
        "tabular1d" => Op::Tabular1d {
            points: required(d, "points")?,
            values: required(d, "values")?,
            fill: required(d, "fill")?,
        },
        "logical" => {
            let condition: String = required(d, "condition")?;
            let condition = match condition.as_str() {
                "GT" => LogicalCondition::Gt,
                "LT" => LogicalCondition::Lt,
                "EQ" => LogicalCondition::Eq,
                "NE" => LogicalCondition::Ne,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "ops[{index}]: logical condition must be GT/LT/EQ/NE, got {other:?}"
                    )));
                }
            };
            Op::Logical {
                condition,
                compareto: required(d, "compareto")?,
                value: required(d, "value")?,
            }
        }
        "select" => {
            let label: Bound<'_, PyDict> = required(d, "label")?;
            let key = parse_label_key(&label, index)?;
            let case_list: Vec<Bound<'_, PyDict>> = required(d, "cases")?;
            let cases = case_list
                .iter()
                .map(|case| {
                    let label: i64 = required(case, "label")?;
                    let spec: Bound<'_, PyDict> = required(case, "program")?;
                    Ok((label, parse_program(&spec)?))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Op::Select { key, cases }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "ops[{index}]: unknown op {other:?}"
            )));
        }
    };
    Ok(OpInstr {
        op,
        inputs,
        outputs,
    })
}

fn parse_label_key(d: &Bound<'_, PyDict>, index: usize) -> PyResult<LabelKey> {
    let kind: String = required(d, "kind")?;
    match kind.as_str() {
        "array" => {
            let data: PyReadonlyArray2<'_, i64> = required(d, "data")?;
            let view = data.as_array();
            let (height, width) = (view.nrows(), view.ncols());
            Ok(LabelKey::Array {
                data: view.iter().copied().collect(),
                height,
                width,
            })
        }
        "dict" => Ok(LabelKey::Dict {
            keys: required(d, "keys")?,
            labels: required(d, "labels")?,
            key_input: required(d, "key_input")?,
            atol: required(d, "atol")?,
        }),
        other => Err(PyValueError::new_err(format!(
            "ops[{index}]: label kind must be 'array' or 'dict', got {other:?}"
        ))),
    }
}

fn parse_tpoly(d: &Bound<'_, PyDict>, index: usize) -> PyResult<TPoly> {
    let kind: String = required(d, "kind")?;
    match kind.as_str() {
        "t" => Ok(TPoly::TOnly {
            coeffs: required(d, "coeffs")?,
        }),
        "spatial" => Ok(TPoly::Spatial {
            degree: required(d, "degree")?,
            coeff_polys: required(d, "coeffs")?,
        }),
        other => Err(PyValueError::new_err(format!(
            "ops[{index}]: t-polynomial kind must be 't' or 'spatial', got {other:?}"
        ))),
    }
}

fn required<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py>,
{
    let value = d
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("spec missing key {key:?}")))?;
    value.extract::<T>().map_err(|err| {
        PyValueError::new_err(format!("spec key {key:?}: {}", err.into()))
    })
}

pub(crate) fn register_into(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyWcsProgram>()?;
    Ok(())
}

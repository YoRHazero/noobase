//! The compiled transform program and its evaluator.

use rayon::prelude::*;
use thiserror::Error;

use super::op::Op;

/// Elements processed per register-file chunk. Large enough to amortise the
/// per-chunk dispatch, small enough that a whole register file stays cache
/// resident (`n_regs * CHUNK * 8` bytes; 16 registers -> 512 KiB).
const CHUNK: usize = 4096;

/// Minimum elements before evaluation fans out across rayon workers.
const PAR_THRESHOLD: usize = 2 * CHUNK;

/// Errors raised when constructing or evaluating a [`Program`].
#[derive(Debug, Error)]
pub enum ProgramError {
    #[error("op {index} ({op}): expected {want_in} inputs / {want_out} outputs, got {got_in} / {got_out}")]
    Arity {
        index: usize,
        op: &'static str,
        want_in: usize,
        want_out: usize,
        got_in: usize,
        got_out: usize,
    },
    #[error("op {index}: register {reg} out of range (n_regs = {n_regs})")]
    Register {
        index: usize,
        reg: u16,
        n_regs: usize,
    },
    #[error("op {index}: {message}")]
    BadOp { index: usize, message: String },
    #[error("program input/output register {reg} out of range (n_regs = {n_regs})")]
    IoRegister { reg: u16, n_regs: usize },
    #[error("expected {want} input arrays, got {got}")]
    InputCount { want: usize, got: usize },
    #[error("input arrays must share one length: got {a} and {b}")]
    InputLength { a: usize, b: usize },
}

/// One instruction: an op plus its register wiring.
#[derive(Debug, Clone)]
pub struct OpInstr {
    pub op: Op,
    pub inputs: Vec<u16>,
    pub outputs: Vec<u16>,
}

/// A compiled coordinate-transform program (see the module docs).
#[derive(Debug, Clone)]
pub struct Program {
    n_regs: usize,
    inputs: Vec<u16>,
    outputs: Vec<u16>,
    ops: Vec<OpInstr>,
}

impl Program {
    /// Validate wiring and build a program.
    pub fn new(
        n_regs: usize,
        inputs: Vec<u16>,
        outputs: Vec<u16>,
        ops: Vec<OpInstr>,
    ) -> Result<Self, ProgramError> {
        for &reg in inputs.iter().chain(&outputs) {
            if reg as usize >= n_regs {
                return Err(ProgramError::IoRegister { reg, n_regs });
            }
        }
        for (index, instr) in ops.iter().enumerate() {
            let (want_in, want_out) = instr.op.arity();
            if instr.inputs.len() != want_in || instr.outputs.len() != want_out {
                return Err(ProgramError::Arity {
                    index,
                    op: op_name(&instr.op),
                    want_in,
                    want_out,
                    got_in: instr.inputs.len(),
                    got_out: instr.outputs.len(),
                });
            }
            for &reg in instr.inputs.iter().chain(&instr.outputs) {
                if reg as usize >= n_regs {
                    return Err(ProgramError::Register { index, reg, n_regs });
                }
            }
            instr
                .op
                .check()
                .map_err(|message| ProgramError::BadOp { index, message })?;
        }
        Ok(Self {
            n_regs,
            inputs,
            outputs,
            ops,
        })
    }

    /// Number of input registers (arrays expected by [`Program::eval`]).
    pub fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Number of output registers (arrays returned by [`Program::eval`]).
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Evaluate on flat `f64` slices of one shared length.
    ///
    /// Returns one `Vec<f64>` per program output. Parallelises over element
    /// chunks once the input length crosses a threshold; results are
    /// identical to the sequential path (each element is independent).
    ///
    /// Performance note: parallel workers reuse their register buffer
    /// across chunks *without re-zeroing* (compiled programs are SSA --
    /// every register is written before it is read -- so stale values from
    /// a previous chunk are never observable). Small inputs and nested
    /// sub-program calls allocate an exact-size register file instead.
    pub fn eval(&self, inputs: &[&[f64]]) -> Result<Vec<Vec<f64>>, ProgramError> {
        if inputs.len() != self.inputs.len() {
            return Err(ProgramError::InputCount {
                want: self.inputs.len(),
                got: inputs.len(),
            });
        }
        let n = inputs.first().map_or(0, |a| a.len());
        for a in inputs {
            if a.len() != n {
                return Err(ProgramError::InputLength { a: n, b: a.len() });
            }
        }
        if n == 0 {
            return Ok(self.outputs.iter().map(|_| Vec::new()).collect());
        }

        let run_chunk = |regs: &mut [f64], stride: usize, start: usize, len: usize| {
            for (&reg, input) in self.inputs.iter().zip(inputs) {
                let base = reg as usize * stride;
                regs[base..base + len].copy_from_slice(&input[start..start + len]);
            }
            for instr in &self.ops {
                instr
                    .op
                    .apply(regs, stride, len, &instr.inputs, &instr.outputs);
            }
        };

        if n < PAR_THRESHOLD {
            // One exact-size pass: keeps nested (select sub-program) calls
            // from paying full-chunk allocation for tiny groups.
            let mut regs = vec![0.0_f64; self.n_regs * n];
            run_chunk(&mut regs, n, 0, n);
            return Ok(self
                .outputs
                .iter()
                .map(|&reg| regs[reg as usize * n..(reg as usize + 1) * n].to_vec())
                .collect());
        }

        let n_chunks = n.div_ceil(CHUNK);
        let chunks: Vec<Vec<Vec<f64>>> = (0..n_chunks)
            .into_par_iter()
            .map_init(
                || vec![0.0_f64; self.n_regs * CHUNK],
                |regs, ci| {
                    let start = ci * CHUNK;
                    let len = CHUNK.min(n - start);
                    run_chunk(regs, CHUNK, start, len);
                    self.outputs
                        .iter()
                        .map(|&reg| {
                            let base = reg as usize * CHUNK;
                            regs[base..base + len].to_vec()
                        })
                        .collect()
                },
            )
            .collect();

        let mut out: Vec<Vec<f64>> = self
            .outputs
            .iter()
            .map(|_| Vec::with_capacity(n))
            .collect();
        for chunk in chunks {
            for (dst, src) in out.iter_mut().zip(chunk) {
                dst.extend_from_slice(&src);
            }
        }
        Ok(out)
    }

    /// Evaluate a single point without any array machinery.
    pub fn eval_scalar(&self, inputs: &[f64]) -> Result<Vec<f64>, ProgramError> {
        if inputs.len() != self.inputs.len() {
            return Err(ProgramError::InputCount {
                want: self.inputs.len(),
                got: inputs.len(),
            });
        }
        let mut regs = vec![0.0_f64; self.n_regs];
        for (&reg, &value) in self.inputs.iter().zip(inputs) {
            regs[reg as usize] = value;
        }
        for instr in &self.ops {
            instr.op.apply(&mut regs, 1, 1, &instr.inputs, &instr.outputs);
        }
        Ok(self.outputs.iter().map(|&reg| regs[reg as usize]).collect())
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Shift { .. } => "shift",
        Op::Scale { .. } => "scale",
        Op::Const { .. } => "const",
        Op::Poly1d { .. } => "poly1d",
        Op::Poly2d { .. } => "poly2d",
        Op::Affine2 { .. } => "affine2",
        Op::SphToCart => "sph2cart",
        Op::CartToSph { .. } => "cart2sph",
        Op::Rot3 { .. } => "rot3",
        Op::TanProject => "tan_project",
        Op::TanDeproject => "tan_deproject",
        Op::GrismForward { .. } => "grism_forward",
        Op::GrismBackward { .. } => "grism_backward",
        Op::Rot3Gwa { .. } => "rot3_gwa",
        Op::Binary { .. } => "binary",
        Op::Unitless2DirCos => "unitless2dircos",
        Op::DirCos2Unitless => "dircos2unitless",
        Op::GratingWavelength { .. } => "grating_wavelength",
        Op::GratingAngles3D { .. } => "grating_angles3d",
        Op::Tabular1d { .. } => "tabular1d",
        Op::Logical { .. } => "logical",
        Op::Select { .. } => "select",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// shift -> scale chain: out = (in + 1) * 2.
    fn simple_program() -> Program {
        Program::new(
            3,
            vec![0],
            vec![2],
            vec![
                OpInstr {
                    op: Op::Shift { offset: 1.0 },
                    inputs: vec![0],
                    outputs: vec![1],
                },
                OpInstr {
                    op: Op::Scale { factor: 2.0 },
                    inputs: vec![1],
                    outputs: vec![2],
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn eval_matches_scalar_across_chunk_boundaries() {
        let p = simple_program();
        let n = 3 * CHUNK + 17; // exercise the parallel path + ragged tail
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
        let out = p.eval(&[&xs]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), n);
        for (i, &x) in xs.iter().enumerate().step_by(997) {
            let s = p.eval_scalar(&[x]).unwrap();
            assert_eq!(out[0][i], s[0]);
            assert_eq!(s[0], (x + 1.0) * 2.0);
        }
    }

    #[test]
    fn wiring_validation_rejects_bad_registers() {
        let err = Program::new(
            1,
            vec![0],
            vec![5],
            vec![],
        );
        assert!(matches!(err, Err(ProgramError::IoRegister { reg: 5, .. })));
    }
}

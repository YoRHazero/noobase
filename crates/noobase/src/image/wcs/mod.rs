//! Compiled WCS transform programs.
//!
//! A [`Program`] is a flat register-machine encoding of a coordinate
//! transform chain (a gwcs / astropy compound-model pipeline compiled by the
//! caller). Evaluation is pure `f64` array math: no model tree, no per-call
//! dispatch overhead, rayon-parallel over element chunks.
//!
//! Design notes:
//!
//! - **No WCS parsing.** Like `reproject_exact`, this module never reads
//!   ASDF/FITS. The caller walks its favourite WCS object tree (gwcs,
//!   astropy) once, extracts plain coefficients, and hands them in as a
//!   program. There is exactly one op per mathematical primitive; anything
//!   that is pure plumbing in the source model tree (`Mapping`, `Identity`)
//!   must be resolved to register wiring at compile time.
//!
//! - **Register machine, SSA-style.** A program owns `n_regs` virtual
//!   registers of `f64`. Ops read `inputs` registers and write `outputs`
//!   registers in listed order. Compilers are expected to emit write-once
//!   (SSA) programs; evaluation is strictly sequential per element, so
//!   register reuse is *allowed* but on the compiler's own head.
//!
//! - **Angles are degrees.** Spherical ops ([`Op::SphToCart`],
//!   [`Op::CartToSph`], [`Op::TanProject`], [`Op::TanDeproject`]) take and
//!   return degrees, matching gwcs's `SphericalToCartesian` /
//!   `CartesianToSpherical` and FITS conventions.
//!
//! - **Grism ops follow stdatamodels semantics.** The JWST WFSS dispersion
//!   ops replicate `stdatamodels.jwst.transforms.models`
//!   `NIRCAM{Forward,Backward}GrismDispersion`, with one deliberate
//!   difference: where jwst inverts the trace polynomial by sampling
//!   `t in [0, 1]` on a coarse grid and interpolating, this module solves
//!   the (at most quadratic) polynomial exactly. Round-trips are therefore
//!   self-consistent to machine precision, and agreement with jwst is
//!   bounded by *jwst's* sampling error, not ours. The spectral-order input
//!   is read from the first element of its register (jwst semantics: one
//!   order per call); it must be constant across a call.
//!
//! - **Future piecewise transforms.** IFS modes (NIRSpec IFU slices, MIRI
//!   MRS regions) need a selector op that routes elements through different
//!   sub-programs. The op-code namespace reserves `select` for this; it is
//!   not implemented in v1.

mod op;
mod program;

pub use op::{
    BinaryKind, GrismAxis, GrismOrderBwd, GrismOrderFwd, GwaAxis, GwaStep, LabelKey,
    LogicalCondition, Op, TPoly,
};
pub use program::{OpInstr, Program, ProgramError};

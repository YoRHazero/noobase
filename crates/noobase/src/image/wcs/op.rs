//! Transform op-codes and their per-chunk evaluation kernels.

use super::program::Program;

/// Comparison used by [`Op::Logical`], matching
/// ``stdatamodels.jwst.transforms.models.Logical``.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalCondition {
    Gt,
    Lt,
    Eq,
    Ne,
}

/// Elementwise binary arithmetic between two already-computed registers,
/// matching astropy compound-model ``+ - * /`` operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// One step of a [`Op::Rot3Gwa`] rotation sequence: the axis and the
/// precomputed sine / cosine of the (verbatim, unit-unconverted) angle.
#[derive(Debug, Clone, Copy)]
pub struct GwaStep {
    pub axis: GwaAxis,
    pub cos: f64,
    pub sin: f64,
}

/// Rotation axis of a [`GwaStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwaAxis {
    X,
    Y,
    Z,
}

/// How [`Op::Select`] maps an element to a region label.
#[derive(Debug, Clone)]
pub enum LabelKey {
    /// Per-pixel label image (gwcs ``LabelMapperArray``): element label is
    /// ``data[floor(y + 0.5), floor(x + 0.5)]`` where ``x`` / ``y`` are the
    /// select op's first two inputs; out-of-bounds or non-finite
    /// coordinates give label 0 (= no label).
    Array {
        data: Vec<i64>,
        height: usize,
        width: usize,
    },
    /// Quantized-value lookup (gwcs ``LabelMapperDict``): the label of the
    /// first ``keys[i]`` with ``|input - keys[i]| <= atol`` (key spacing in
    /// practice far exceeds ``atol``, so first-match equals gwcs's
    /// last-match). No match (including NaN input) gives label 0.
    Dict {
        keys: Vec<f64>,
        labels: Vec<i64>,
        key_input: usize,
        atol: f64,
    },
}

impl LabelKey {
    /// Label of element `k` given the select op's gathered inputs.
    #[inline]
    fn label(&self, inputs: &[&[f64]], k: usize) -> i64 {
        match self {
            LabelKey::Array {
                data,
                height,
                width,
            } => {
                let (x, y) = (inputs[0][k], inputs[1][k]);
                if !x.is_finite() || !y.is_finite() {
                    return 0;
                }
                let ix = (x + 0.5).floor();
                let iy = (y + 0.5).floor();
                if ix < 0.0 || iy < 0.0 || ix >= *width as f64 || iy >= *height as f64 {
                    return 0;
                }
                data[iy as usize * width + ix as usize]
            }
            LabelKey::Dict {
                keys,
                labels,
                key_input,
                atol,
            } => {
                let value = inputs[*key_input][k];
                for (key, label) in keys.iter().zip(labels) {
                    if (value - key).abs() <= *atol {
                        return *label;
                    }
                }
                0
            }
        }
    }
}

/// Polynomial in the grism trace parameter `t`, in the two forms the JWST
/// specwcs reference files use (stdatamodels "guess form").
#[derive(Debug, Clone)]
pub enum TPoly {
    /// Depends only on `t`: `p(t) = sum_i coeffs[i] * t^i`.
    TOnly { coeffs: Vec<f64> },
    /// Coefficients are themselves 2-D polynomials of the source position:
    /// `p(t; x, y) = sum_i c_i(x, y) * t^i`, where `coeff_polys[i]` is the
    /// dense `(degree+1) x (degree+1)` matrix of `c_i` (see [`Op::Poly2d`]).
    Spatial {
        degree: usize,
        coeff_polys: Vec<Vec<f64>>,
    },
}

impl TPoly {
    /// Evaluate at trace parameter `t` and source position `(x, y)`.
    #[inline]
    pub fn eval(&self, t: f64, x: f64, y: f64) -> f64 {
        match self {
            TPoly::TOnly { coeffs } => poly1d(coeffs, t),
            TPoly::Spatial {
                degree,
                coeff_polys,
            } => {
                // Horner in t; each coefficient is a Poly2D of (x, y).
                let mut acc = 0.0;
                for c in coeff_polys.iter().rev() {
                    acc = acc * t + poly2d(*degree, c, x, y);
                }
                acc
            }
        }
    }

    /// Coefficients of the polynomial in `t` at a fixed `(x, y)`.
    ///
    /// Returns the number of coefficients written into `buf` (at most
    /// `buf.len()`); the polynomial degree is that count minus one.
    #[inline]
    fn t_coeffs(&self, x: f64, y: f64, buf: &mut [f64]) -> usize {
        match self {
            TPoly::TOnly { coeffs } => {
                let n = coeffs.len().min(buf.len());
                buf[..n].copy_from_slice(&coeffs[..n]);
                n
            }
            TPoly::Spatial {
                degree,
                coeff_polys,
            } => {
                let n = coeff_polys.len().min(buf.len());
                for (slot, c) in buf[..n].iter_mut().zip(coeff_polys) {
                    *slot = poly2d(*degree, c, x, y);
                }
                n
            }
        }
    }

    /// Solve `p(t; x, y) = value` for `t`, exactly.
    ///
    /// Linear and quadratic polynomials (all NIRCam WFSS trace models) are
    /// solved in closed form; for a quadratic the root closer to the centre
    /// of the physical trace window `t in [0, 1]` is returned (the other
    /// root sits at `~ -c1/c2`, far outside the window, because dispersion
    /// is near-linear). Degenerate / unsupported cases return NaN.
    #[inline]
    pub fn invert(&self, value: f64, x: f64, y: f64) -> f64 {
        let mut c = [0.0_f64; 8];
        let n = self.t_coeffs(x, y, &mut c);
        // Strip trailing (numerically) zero leading coefficients.
        let mut n = n;
        while n > 1 && c[n - 1] == 0.0 {
            n -= 1;
        }
        match n {
            0 => f64::NAN,
            1 => f64::NAN, // constant polynomial: no unique inverse
            2 => (value - c[0]) / c[1],
            3 => {
                // c2 t^2 + c1 t + (c0 - value) = 0, stable two-root form.
                let (c0, c1, c2) = (c[0] - value, c[1], c[2]);
                let disc = c1 * c1 - 4.0 * c2 * c0;
                if disc < 0.0 {
                    return f64::NAN;
                }
                let q = -0.5 * (c1 + c1.signum() * disc.sqrt());
                let r1 = q / c2;
                let r2 = c0 / q;
                if (r1 - 0.5).abs() <= (r2 - 0.5).abs() {
                    r1
                } else {
                    r2
                }
            }
            _ => f64::NAN, // cubic+ trace models do not occur in JWST WFSS
        }
    }
}

/// Dispersion axis of a JWST WFSS forward grism transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrismAxis {
    Row,
    Column,
}

/// Per-spectral-order models of a forward (dispersed -> direct image)
/// grism transform.
#[derive(Debug, Clone)]
pub struct GrismOrderFwd {
    /// Trace displacement along the dispersion axis as a function of `t`
    /// (jwst `xmodels` for row dispersion, `ymodels` for column).
    pub alongdisp: TPoly,
    /// Wavelength as a function of `t` (jwst `lmodels`).
    pub lmodel: TPoly,
}

/// Per-spectral-order models of a backward (direct image -> dispersed)
/// grism transform.
#[derive(Debug, Clone)]
pub struct GrismOrderBwd {
    /// Wavelength as a function of `t` (inverted for `t`).
    pub lmodel: TPoly,
    /// Trace x-displacement as a function of `t`.
    pub xmodel: TPoly,
    /// Trace y-displacement as a function of `t`.
    pub ymodel: TPoly,
}

/// A single transform op: reads its input registers element-wise, writes its
/// output registers. Arities are fixed per variant (see `arity`).
#[derive(Debug, Clone)]
pub enum Op {
    /// `out = in + offset` (1 -> 1).
    Shift { offset: f64 },
    /// `out = in * factor` (1 -> 1).
    Scale { factor: f64 },
    /// `out = value` (0 -> 1).
    Const { value: f64 },
    /// `out = sum_i coeffs[i] * in^i` (1 -> 1).
    Poly1d { coeffs: Vec<f64> },
    /// `out = sum_{i+j<=degree} c[i][j] * x^i * y^j` (2 -> 1). `coeffs` is
    /// the dense `(degree+1) x (degree+1)` matrix flattened row-major
    /// (row = x power, column = y power); entries with `i+j > degree` are
    /// ignored and should be zero.
    Poly2d { degree: usize, coeffs: Vec<f64> },
    /// `(x', y') = m @ (x, y) + t` (2 -> 2).
    Affine2 {
        matrix: [[f64; 2]; 2],
        translation: [f64; 2],
    },
    /// Unit-sphere `(lon, lat)` degrees -> cartesian `(x, y, z)` (2 -> 3).
    SphToCart,
    /// Cartesian `(x, y, z)` -> `(lon, lat)` degrees (3 -> 2). `wrap_lon_at`
    /// is 360 (lon in `[0, 360)`) or 180 (lon in `(-180, 180]`), matching
    /// `gwcs.geometry.CartesianToSpherical`.
    CartToSph { wrap_lon_at: u16 },
    /// `(x, y, z)' = matrix @ (x, y, z)` (3 -> 3).
    Rot3 { matrix: [[f64; 3]; 3] },
    /// Gnomonic (TAN) projection, native spherical `(lon, lat)` degrees ->
    /// tangent-plane `(x, y)` degrees (2 -> 2). FITS convention: the
    /// projection point is the native pole `lat = 90`.
    TanProject,
    /// Inverse gnomonic: tangent-plane `(x, y)` degrees -> native spherical
    /// `(lon, lat)` degrees (2 -> 2).
    TanDeproject,
    /// JWST WFSS forward row/column dispersion (5 -> 1):
    /// `(x, y, x0, y0, order) -> wavelength`. Pass-through outputs of the
    /// jwst model (`x0`, `y0`, `order`) are compile-time register wiring.
    GrismForward {
        axis: GrismAxis,
        orders: Vec<i32>,
        models: Vec<GrismOrderFwd>,
    },
    /// JWST WFSS backward dispersion (4 -> 2):
    /// `(x0, y0, wavelength, order) -> (x_dispersed, y_dispersed)`.
    GrismBackward {
        orders: Vec<i32>,
        models: Vec<GrismOrderBwd>,
    },
    /// Elementwise arithmetic of two registers (2 -> 1).
    Binary { kind: BinaryKind },
    /// `(x, y) -> (x, y, 1) / |(x, y, 1)|` direction cosines (2 -> 3),
    /// matching jwst `Unitless2DirCos`.
    Unitless2DirCos,
    /// `(x, y, z) -> (x / z, y / z)` (3 -> 2), matching jwst
    /// `DirCos2Unitless`.
    DirCos2Unitless,
    /// Grating equation solved for wavelength (2 -> 1):
    /// `(alpha_in, alpha_out) -> (alpha_in + alpha_out) / factor` where
    /// `factor = groove_density * spectral_order`
    /// (gwcs `WavelengthFromGratingEquation`).
    GratingWavelength { factor: f64 },
    /// Grating equation solved for the refracted direction (3 -> 3):
    /// `(lam, alpha_in, beta_in) -> (alpha_in - factor * lam, -beta_in,
    /// sqrt(1 - a^2 - b^2))` (gwcs `AnglesFromGratingEquation3D`).
    GratingAngles3D { factor: f64 },
    /// 1-D linear interpolation over ascending `points` (1 -> 1); outside
    /// the domain the output is `fill` (astropy `Tabular1D`,
    /// `method='linear'`, `bounds_error=False`).
    Tabular1d {
        points: Vec<f64>,
        values: Vec<f64>,
        fill: f64,
    },
    /// Conditional substitution (1 -> 1): where the input is non-NaN and
    /// compares true against `compareto`, output `value`; otherwise pass
    /// the input through (jwst `Logical` with scalar operands).
    Logical {
        condition: LogicalCondition,
        compareto: f64,
        value: f64,
    },
    /// NIRSpec GWA rotation sequence (3 -> 3), matching jwst
    /// `Rotation3DToGWA`: each step rotates two components and then
    /// *renormalizes* `z = sqrt(1 - x^2 - y^2)` (NaN where the direction
    /// cosines leave the unit disc), so this is NOT a linear rotation.
    /// Angles enter `cos` / `sin` verbatim, exactly as jwst applies them.
    Rot3Gwa { steps: Vec<GwaStep> },
    /// Piecewise transform (gwcs `RegionsSelector`): each element is
    /// routed by its label to one of `cases`' sub-programs; elements with
    /// no label / no matching case yield NaN on every output. Sub-program
    /// arities must all equal this op's wiring arity.
    Select {
        key: LabelKey,
        cases: Vec<(i64, Program)>,
    },
}

impl Op {
    /// `(n_inputs, n_outputs)` of this op.
    pub fn arity(&self) -> (usize, usize) {
        match self {
            Op::Shift { .. }
            | Op::Scale { .. }
            | Op::Poly1d { .. }
            | Op::Tabular1d { .. }
            | Op::Logical { .. } => (1, 1),
            Op::Const { .. } => (0, 1),
            Op::Poly2d { .. } | Op::Binary { .. } | Op::GratingWavelength { .. } => (2, 1),
            Op::Affine2 { .. } | Op::TanProject | Op::TanDeproject => (2, 2),
            Op::SphToCart | Op::Unitless2DirCos => (2, 3),
            Op::CartToSph { .. } | Op::DirCos2Unitless => (3, 2),
            Op::Rot3 { .. } | Op::GratingAngles3D { .. } | Op::Rot3Gwa { .. } => (3, 3),
            Op::GrismForward { .. } => (5, 1),
            Op::GrismBackward { .. } => (4, 2),
            Op::Select { cases, .. } => cases.first().map_or((0, 0), |(_, program)| {
                (program.n_inputs(), program.n_outputs())
            }),
        }
    }

    /// Extra validation beyond arity (coefficient shapes, order tables).
    pub fn check(&self) -> Result<(), String> {
        let check_tpoly = |p: &TPoly, what: &str| -> Result<(), String> {
            match p {
                TPoly::TOnly { coeffs } if coeffs.is_empty() => {
                    Err(format!("{what}: empty t-polynomial"))
                }
                TPoly::Spatial {
                    degree,
                    coeff_polys,
                } => {
                    if coeff_polys.is_empty() {
                        return Err(format!("{what}: empty t-polynomial"));
                    }
                    let want = (degree + 1) * (degree + 1);
                    for c in coeff_polys {
                        if c.len() != want {
                            return Err(format!(
                                "{what}: coefficient matrix has {} entries, expected {want}",
                                c.len()
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        };
        match self {
            Op::Poly1d { coeffs } if coeffs.is_empty() => Err("poly1d: empty coefficients".into()),
            Op::Poly2d { degree, coeffs } => {
                let want = (degree + 1) * (degree + 1);
                if coeffs.len() != want {
                    Err(format!(
                        "poly2d: coefficient matrix has {} entries, expected {want}",
                        coeffs.len()
                    ))
                } else {
                    Ok(())
                }
            }
            Op::CartToSph { wrap_lon_at } if *wrap_lon_at != 180 && *wrap_lon_at != 360 => Err(
                format!("cart2sph: wrap_lon_at must be 180 or 360, got {wrap_lon_at}"),
            ),
            Op::GrismForward { orders, models, .. } => {
                if orders.len() != models.len() || orders.is_empty() {
                    return Err("grism_forward: orders/models length mismatch or empty".into());
                }
                for m in models {
                    check_tpoly(&m.alongdisp, "grism_forward.alongdisp")?;
                    check_tpoly(&m.lmodel, "grism_forward.lmodel")?;
                }
                Ok(())
            }
            Op::GrismBackward { orders, models } => {
                if orders.len() != models.len() || orders.is_empty() {
                    return Err("grism_backward: orders/models length mismatch or empty".into());
                }
                for m in models {
                    check_tpoly(&m.lmodel, "grism_backward.lmodel")?;
                    check_tpoly(&m.xmodel, "grism_backward.xmodel")?;
                    check_tpoly(&m.ymodel, "grism_backward.ymodel")?;
                }
                Ok(())
            }
            Op::Tabular1d { points, values, .. } => {
                if points.len() != values.len() || points.len() < 2 {
                    return Err(format!(
                        "tabular1d: needs >= 2 points with matching values, got {} / {}",
                        points.len(),
                        values.len()
                    ));
                }
                // `all(<)` is false for a non-ascending pair *or* a NaN
                // (NaN comparisons are false), so this rejects both.
                if !points.windows(2).all(|w| w[0] < w[1]) {
                    return Err("tabular1d: points must be strictly ascending".into());
                }
                Ok(())
            }
            Op::Select { key, cases } => {
                let Some((n_in, n_out)) = cases
                    .first()
                    .map(|(_, program)| (program.n_inputs(), program.n_outputs()))
                else {
                    return Err("select: no cases".into());
                };
                for (label, program) in cases {
                    if program.n_inputs() != n_in || program.n_outputs() != n_out {
                        return Err(format!(
                            "select: case {label} arity ({}, {}) differs from ({n_in}, {n_out})",
                            program.n_inputs(),
                            program.n_outputs()
                        ));
                    }
                }
                match key {
                    LabelKey::Array {
                        data,
                        height,
                        width,
                    } => {
                        if data.len() != height * width {
                            return Err(format!(
                                "select: label image has {} entries, expected {}x{}",
                                data.len(),
                                height,
                                width
                            ));
                        }
                        if n_in < 2 {
                            return Err("select: array label key needs >= 2 inputs (x, y)".into());
                        }
                    }
                    LabelKey::Dict {
                        keys,
                        labels,
                        key_input,
                        ..
                    } => {
                        if keys.len() != labels.len() || keys.is_empty() {
                            return Err("select: dict keys/labels length mismatch or empty".into());
                        }
                        if *key_input >= n_in {
                            return Err(format!(
                                "select: key_input {key_input} out of range (n_in = {n_in})"
                            ));
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Apply this op to one chunk of the register file.
    ///
    /// `regs` is the flat register buffer of the chunk (`n_regs * chunk`
    /// elements, register `r` occupying `regs[r*chunk .. r*chunk + n]`);
    /// `n <= chunk` is the number of live elements. Register indices were
    /// validated at program construction.
    #[inline]
    pub(crate) fn apply(
        &self,
        regs: &mut [f64],
        chunk: usize,
        n: usize,
        ins: &[u16],
        outs: &[u16],
    ) {
        let r = |reg: u16| reg as usize * chunk;
        match self {
            Op::Shift { offset } => {
                let (i, o) = (r(ins[0]), r(outs[0]));
                for k in 0..n {
                    regs[o + k] = regs[i + k] + offset;
                }
            }
            Op::Scale { factor } => {
                let (i, o) = (r(ins[0]), r(outs[0]));
                for k in 0..n {
                    regs[o + k] = regs[i + k] * factor;
                }
            }
            Op::Const { value } => {
                let o = r(outs[0]);
                regs[o..o + n].fill(*value);
            }
            Op::Poly1d { coeffs } => {
                let (i, o) = (r(ins[0]), r(outs[0]));
                for k in 0..n {
                    regs[o + k] = poly1d(coeffs, regs[i + k]);
                }
            }
            Op::Poly2d { degree, coeffs } => {
                let (ix, iy, o) = (r(ins[0]), r(ins[1]), r(outs[0]));
                for k in 0..n {
                    regs[o + k] = poly2d(*degree, coeffs, regs[ix + k], regs[iy + k]);
                }
            }
            Op::Affine2 {
                matrix,
                translation,
            } => {
                let (ix, iy, ox, oy) = (r(ins[0]), r(ins[1]), r(outs[0]), r(outs[1]));
                for k in 0..n {
                    let (x, y) = (regs[ix + k], regs[iy + k]);
                    regs[ox + k] = matrix[0][0] * x + matrix[0][1] * y + translation[0];
                    regs[oy + k] = matrix[1][0] * x + matrix[1][1] * y + translation[1];
                }
            }
            Op::SphToCart => {
                let (ilon, ilat) = (r(ins[0]), r(ins[1]));
                let (ox, oy, oz) = (r(outs[0]), r(outs[1]), r(outs[2]));
                for k in 0..n {
                    let lon = regs[ilon + k].to_radians();
                    let lat = regs[ilat + k].to_radians();
                    let (slon, clon) = lon.sin_cos();
                    let (slat, clat) = lat.sin_cos();
                    regs[ox + k] = clat * clon;
                    regs[oy + k] = clat * slon;
                    regs[oz + k] = slat;
                }
            }
            Op::CartToSph { wrap_lon_at } => {
                let (ix, iy, iz) = (r(ins[0]), r(ins[1]), r(ins[2]));
                let (olon, olat) = (r(outs[0]), r(outs[1]));
                for k in 0..n {
                    let (x, y, z) = (regs[ix + k], regs[iy + k], regs[iz + k]);
                    let h = x.hypot(y);
                    let mut lon = y.atan2(x).to_degrees();
                    if h == 0.0 {
                        lon = 0.0;
                    }
                    if *wrap_lon_at != 180 && lon.is_finite() {
                        lon = lon.rem_euclid(360.0);
                    }
                    regs[olon + k] = lon;
                    regs[olat + k] = z.atan2(h).to_degrees();
                }
            }
            Op::Rot3Gwa { steps } => {
                let (ix, iy, iz) = (r(ins[0]), r(ins[1]), r(ins[2]));
                let (ox, oy, oz) = (r(outs[0]), r(outs[1]), r(outs[2]));
                for k in 0..n {
                    let (mut x, mut y, mut z) = (regs[ix + k], regs[iy + k], regs[iz + k]);
                    for step in steps {
                        let (nx, ny) = match step.axis {
                            GwaAxis::X => (x, y * step.cos + z * step.sin),
                            GwaAxis::Y => (x * step.cos - z * step.sin, y),
                            GwaAxis::Z => {
                                (x * step.cos + y * step.sin, -x * step.sin + y * step.cos)
                            }
                        };
                        x = nx;
                        y = ny;
                        z = (1.0 - x * x - y * y).sqrt();
                    }
                    regs[ox + k] = x;
                    regs[oy + k] = y;
                    regs[oz + k] = z;
                }
            }
            Op::Rot3 { matrix } => {
                let (ix, iy, iz) = (r(ins[0]), r(ins[1]), r(ins[2]));
                let (ox, oy, oz) = (r(outs[0]), r(outs[1]), r(outs[2]));
                for k in 0..n {
                    let v = [regs[ix + k], regs[iy + k], regs[iz + k]];
                    regs[ox + k] = matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2];
                    regs[oy + k] = matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2];
                    regs[oz + k] = matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2];
                }
            }
            Op::TanProject => {
                // Native spherical -> plane: R = 180/pi * cot(lat),
                // x = R sin(lon), y = -R cos(lon) (FITS WCS Paper II).
                // The gnomonic projection is undefined on the far
                // hemisphere (native latitude <= 0, i.e. > 90 deg from the
                // tangent point); emit NaN there to match gwcs / wcslib
                // rather than the finite wrong-sign value cot would give.
                let (ilon, ilat) = (r(ins[0]), r(ins[1]));
                let (ox, oy) = (r(outs[0]), r(outs[1]));
                for k in 0..n {
                    let lat = regs[ilat + k];
                    if lat <= 0.0 {
                        regs[ox + k] = f64::NAN;
                        regs[oy + k] = f64::NAN;
                        continue;
                    }
                    let lon = regs[ilon + k].to_radians();
                    let rr = lat.to_radians().tan().recip().to_degrees();
                    let (slon, clon) = lon.sin_cos();
                    regs[ox + k] = rr * slon;
                    regs[oy + k] = -rr * clon;
                }
            }
            Op::TanDeproject => {
                // Plane -> native spherical: lon = atan2(x, -y),
                // lat = atan(180 / (pi * R)), R = hypot(x, y) in degrees.
                let (ix, iy) = (r(ins[0]), r(ins[1]));
                let (olon, olat) = (r(outs[0]), r(outs[1]));
                for k in 0..n {
                    let (x, y) = (regs[ix + k], regs[iy + k]);
                    let rr = x.hypot(y).to_radians();
                    regs[olon + k] = x.atan2(-y).to_degrees();
                    regs[olat + k] = rr.recip().atan().to_degrees();
                }
            }
            Op::GrismForward {
                axis,
                orders,
                models,
            } => {
                let (ix, iy, ix0, iy0, iord) =
                    (r(ins[0]), r(ins[1]), r(ins[2]), r(ins[3]), r(ins[4]));
                let o = r(outs[0]);
                let Some(m) = order_model(orders, models, regs, iord, n) else {
                    regs[o..o + n].fill(f64::NAN);
                    return;
                };
                for k in 0..n {
                    let (x0, y0) = (regs[ix0 + k], regs[iy0 + k]);
                    let dist = match axis {
                        GrismAxis::Row => regs[ix + k] - x0,
                        GrismAxis::Column => regs[iy + k] - y0,
                    };
                    let t = m.alongdisp.invert(dist, x0, y0);
                    regs[o + k] = m.lmodel.eval(t, x0, y0);
                }
            }
            Op::GrismBackward { orders, models } => {
                let (ix, iy, ilam, iord) = (r(ins[0]), r(ins[1]), r(ins[2]), r(ins[3]));
                let (ox, oy) = (r(outs[0]), r(outs[1]));
                let Some(m) = order_model(orders, models, regs, iord, n) else {
                    regs[ox..ox + n].fill(f64::NAN);
                    regs[oy..oy + n].fill(f64::NAN);
                    return;
                };
                for k in 0..n {
                    let (x0, y0) = (regs[ix + k], regs[iy + k]);
                    let t = m.lmodel.invert(regs[ilam + k], x0, y0);
                    regs[ox + k] = x0 + m.xmodel.eval(t, x0, y0);
                    regs[oy + k] = y0 + m.ymodel.eval(t, x0, y0);
                }
            }
            Op::Binary { kind } => {
                let (ia, ib, o) = (r(ins[0]), r(ins[1]), r(outs[0]));
                for k in 0..n {
                    let (a, b) = (regs[ia + k], regs[ib + k]);
                    regs[o + k] = match kind {
                        BinaryKind::Add => a + b,
                        BinaryKind::Sub => a - b,
                        BinaryKind::Mul => a * b,
                        BinaryKind::Div => a / b,
                    };
                }
            }
            Op::Unitless2DirCos => {
                let (ix, iy) = (r(ins[0]), r(ins[1]));
                let (oa, ob, oc) = (r(outs[0]), r(outs[1]), r(outs[2]));
                for k in 0..n {
                    let (x, y) = (regs[ix + k], regs[iy + k]);
                    let norm = (1.0 + x * x + y * y).sqrt();
                    regs[oa + k] = x / norm;
                    regs[ob + k] = y / norm;
                    regs[oc + k] = 1.0 / norm;
                }
            }
            Op::DirCos2Unitless => {
                let (ix, iy, iz) = (r(ins[0]), r(ins[1]), r(ins[2]));
                let (ox, oy) = (r(outs[0]), r(outs[1]));
                for k in 0..n {
                    let z = regs[iz + k];
                    regs[ox + k] = regs[ix + k] / z;
                    regs[oy + k] = regs[iy + k] / z;
                }
            }
            Op::GratingWavelength { factor } => {
                let (ia, ib, o) = (r(ins[0]), r(ins[1]), r(outs[0]));
                for k in 0..n {
                    regs[o + k] = (regs[ia + k] + regs[ib + k]) / factor;
                }
            }
            Op::GratingAngles3D { factor } => {
                let (ilam, ia, ib) = (r(ins[0]), r(ins[1]), r(ins[2]));
                let (oa, ob, oc) = (r(outs[0]), r(outs[1]), r(outs[2]));
                for k in 0..n {
                    let alpha = regs[ia + k] - factor * regs[ilam + k];
                    let beta = -regs[ib + k];
                    regs[oa + k] = alpha;
                    regs[ob + k] = beta;
                    regs[oc + k] = (1.0 - alpha * alpha - beta * beta).sqrt();
                }
            }
            Op::Tabular1d {
                points,
                values,
                fill,
            } => {
                let (i, o) = (r(ins[0]), r(outs[0]));
                let last = points.len() - 1;
                for k in 0..n {
                    let x = regs[i + k];
                    regs[o + k] = if !(points[0]..=points[last]).contains(&x) {
                        *fill
                    } else {
                        let hi = points.partition_point(|&p| p < x).max(1);
                        let (p0, p1) = (points[hi - 1], points[hi.min(last)]);
                        if hi > last || p1 == p0 {
                            values[last]
                        } else {
                            let frac = (x - p0) / (p1 - p0);
                            values[hi - 1] * (1.0 - frac) + values[hi] * frac
                        }
                    };
                }
            }
            Op::Logical {
                condition,
                compareto,
                value,
            } => {
                let (i, o) = (r(ins[0]), r(outs[0]));
                for k in 0..n {
                    let x = regs[i + k];
                    let hit = !x.is_nan()
                        && match condition {
                            LogicalCondition::Gt => x > *compareto,
                            LogicalCondition::Lt => x < *compareto,
                            LogicalCondition::Eq => x == *compareto,
                            LogicalCondition::Ne => x != *compareto,
                        };
                    regs[o + k] = if hit { *value } else { x };
                }
            }
            Op::Select { key, cases } => {
                // Copy the op inputs out of the register file (they must
                // outlive the mutable writes to the output registers).
                let input_slices: Vec<Vec<f64>> = ins
                    .iter()
                    .map(|&reg| regs[r(reg)..r(reg) + n].to_vec())
                    .collect();
                let input_refs: Vec<&[f64]> = input_slices.iter().map(Vec::as_slice).collect();

                for &out in outs {
                    let o = r(out);
                    regs[o..o + n].fill(f64::NAN);
                }

                // Labels come in long runs (region images are piecewise
                // constant along memory order), so group by run-length
                // ranges: gather/scatter then works on contiguous spans.
                let mut groups: Vec<(i64, Vec<(usize, usize)>)> = Vec::new();
                let mut k = 0;
                while k < n {
                    let label = key.label(&input_refs, k);
                    let start = k;
                    k += 1;
                    while k < n && key.label(&input_refs, k) == label {
                        k += 1;
                    }
                    if label == 0 {
                        continue;
                    }
                    match groups.iter_mut().find(|(l, _)| *l == label) {
                        Some((_, runs)) => runs.push((start, k - start)),
                        None => groups.push((label, vec![(start, k - start)])),
                    }
                }
                for (label, runs) in groups {
                    let Some((_, program)) = cases.iter().find(|(l, _)| *l == label) else {
                        continue; // stays NaN
                    };
                    let gathered: Vec<Vec<f64>> = input_refs
                        .iter()
                        .map(|column| {
                            let mut buffer = Vec::with_capacity(runs.iter().map(|(_, l)| l).sum());
                            for &(start, len) in &runs {
                                buffer.extend_from_slice(&column[start..start + len]);
                            }
                            buffer
                        })
                        .collect();
                    let gathered_refs: Vec<&[f64]> = gathered.iter().map(Vec::as_slice).collect();
                    // Arity and wiring were validated at construction, and
                    // gathered inputs share one length by construction.
                    let results = program
                        .eval(&gathered_refs)
                        .expect("validated select sub-program");
                    for (&out, column) in outs.iter().zip(&results) {
                        let o = r(out);
                        let mut cursor = 0;
                        for &(start, len) in &runs {
                            regs[o + start..o + start + len]
                                .copy_from_slice(&column[cursor..cursor + len]);
                            cursor += len;
                        }
                    }
                }
            }
        }
    }
}

/// Look up the per-order model for the (constant) order register.
#[inline]
fn order_model<'m, M>(
    orders: &[i32],
    models: &'m [M],
    regs: &[f64],
    iord: usize,
    n: usize,
) -> Option<&'m M> {
    if n == 0 {
        return None;
    }
    let order = regs[iord] as i32;
    orders.iter().position(|&o| o == order).map(|i| &models[i])
}

/// Horner evaluation of `sum_i coeffs[i] * x^i`.
#[inline]
fn poly1d(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Dense-matrix 2-D polynomial: `sum_{i,j} c[i][j] x^i y^j` with `c`
/// flattened row-major over `(degree+1) x (degree+1)` (row = x power).
/// Horner in y innermost, Horner in x outermost.
#[inline]
fn poly2d(degree: usize, coeffs: &[f64], x: f64, y: f64) -> f64 {
    let w = degree + 1;
    let mut acc = 0.0;
    for i in (0..w).rev() {
        let row = &coeffs[i * w..(i + 1) * w];
        let mut row_acc = 0.0;
        for &c in row.iter().rev() {
            row_acc = row_acc * y + c;
        }
        acc = acc * x + row_acc;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly2d_matches_direct_sum() {
        // p(x, y) = 1 + 2y + 3x + 4xy + 5x^2 over degree 2.
        let mut c = vec![0.0; 9];
        c[0] = 1.0; // x^0 y^0
        c[1] = 2.0; // x^0 y^1
        c[3] = 3.0; // x^1 y^0
        c[4] = 4.0; // x^1 y^1
        c[6] = 5.0; // x^2 y^0
        let (x, y) = (1.3, -0.7);
        let want = 1.0 + 2.0 * y + 3.0 * x + 4.0 * x * y + 5.0 * x * x;
        assert!((poly2d(2, &c, x, y) - want).abs() < 1e-14);
    }

    #[test]
    fn tpoly_quadratic_invert_round_trips() {
        // lambda(t) = 3.9 + 1.1 t + 0.02 t^2 (near-linear dispersion).
        let p = TPoly::TOnly {
            coeffs: vec![3.9, 1.1, 0.02],
        };
        for &t in &[0.05, 0.3, 0.5, 0.9] {
            let lam = p.eval(t, 0.0, 0.0);
            assert!((p.invert(lam, 0.0, 0.0) - t).abs() < 1e-12);
        }
    }

    #[test]
    fn tan_project_round_trips() {
        let mut regs = vec![0.0; 2 * 4];
        // lon = 30 deg, lat = 89.5 deg (near the projection pole).
        regs[0] = 30.0;
        regs[4] = 89.5;
        Op::TanProject.apply(&mut regs, 4, 1, &[0, 1], &[0, 1]);
        Op::TanDeproject.apply(&mut regs, 4, 1, &[0, 1], &[0, 1]);
        assert!((regs[0] - 30.0).abs() < 1e-10);
        assert!((regs[4] - 89.5).abs() < 1e-10);
    }
}

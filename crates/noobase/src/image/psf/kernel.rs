//! Shared separable bicubic Catmull-Rom interpolation kernel.
//!
//! This is the single source of the interpolation weights used by both
//! the forward operator ([`super::render`]) and its adjoint
//! ([`super::accumulate`]). Sharing one weight function is the structural
//! guarantee that `accumulate` is the *exact* transpose of `render`:
//! `render` gathers `value += epsf[ru, rv] * w_u * w_v` while
//! `accumulate` scatters `epsf_acc[ru, rv] += coeff * w_u * w_v` over the
//! very same per-tap weights, the same `floor`-derived tap base, and the
//! same zero-padding rule. Were the two to compute weights independently,
//! any drift between them would silently break the inner-product
//! transpose identity that the solver relies on.
//!
//! Keys cubic convolution with `a = -1/2` (the Catmull-Rom member),
//! four taps per axis. It is a pure-Rust, psf-module-internal helper that
//! never crosses the FFI boundary and is never visible to the caller.

use ndarray::ArrayView2;

/// Keys cubic-convolution (Catmull-Rom, `a = -1/2`) weights for the four
/// taps surrounding a fractional position `frac` in `[0, 1)`. The
/// returned array is ordered `[w(-1), w(0), w(+1), w(+2)]` relative to
/// `floor(coordinate)`.
///
/// The weights sum to one (partition of unity) and at `frac = 0` collapse
/// to `[0, 1, 0, 0]`, which is what makes an integer-aligned sample an
/// exact, interpolation-free grid pick (and, on the adjoint side, an
/// exact grid scatter).
#[inline]
pub(super) fn catmull_rom_weights(frac: f64) -> [f64; 4] {
    // Catmull-Rom basis on the unit interval, parameter t = frac:
    //   w(-1) = -0.5 t^3 +      t^2 - 0.5 t
    //   w( 0) =  1.5 t^3 -  2.5 t^2       + 1
    //   w(+1) = -1.5 t^3 +  2.0 t^2 + 0.5 t
    //   w(+2) =  0.5 t^3 -  0.5 t^2
    let t = frac;
    [
        ((-0.5 * t + 1.0) * t - 0.5) * t,
        ((1.5 * t - 2.5) * t) * t + 1.0,
        ((-1.5 * t + 2.0) * t + 0.5) * t,
        ((0.5 * t - 0.5) * t) * t,
    ]
}

/// Separable bicubic Catmull-Rom point sample of `epsf` at the continuous
/// model coordinate `(k_u, k_v)`. Taps that fall outside the model grid
/// are zero (zero padding).
///
/// The adjoint scatters the identical per-tap weights returned by
/// [`catmull_rom_weights`] at the identical tap base derived here, so this
/// function and the scatter stay exact transposes by construction.
pub(super) fn catmull_rom_sample(epsf: &ArrayView2<f64>, k_u: f64, k_v: f64) -> f64 {
    let side = epsf.shape()[0] as i64; // square: shape()[0] == shape()[1]

    let u_floor = k_u.floor();
    let v_floor = k_v.floor();
    let weights_u = catmull_rom_weights(k_u - u_floor);
    let weights_v = catmull_rom_weights(k_v - v_floor);

    // The four taps of a Catmull-Rom segment sit at integer offsets
    // -1, 0, +1, +2 from floor(k); `weights_*[t]` is the weight of tap
    // `base + t`. The adjoint must scatter these very same weights.
    let base_u = u_floor as i64 - 1;
    let base_v = v_floor as i64 - 1;

    let mut value = 0.0_f64;
    for tap_u in 0..4 {
        let row = base_u + tap_u as i64;
        if row < 0 || row >= side {
            continue; // zero-pad outside the model grid
        }
        let weight_u = weights_u[tap_u];
        for tap_v in 0..4 {
            let column = base_v + tap_v as i64;
            if column < 0 || column >= side {
                continue; // zero-pad outside the model grid
            }
            value += epsf[(row as usize, column as usize)] * weight_u * weights_v[tap_v];
        }
    }
    value
}

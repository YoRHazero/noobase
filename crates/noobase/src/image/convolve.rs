//! Image-domain convolution entry points: a fixed 2-D PSF (req2) and an
//! along-axis 1-D Gaussian matched filter for grism line search (req3).
//!
//! Both are thin domain wrappers over the `convolve` layer. The image
//! subsystem's missing-data convention — NaN (and ±inf) is missing data,
//! exactly as `image::reproject_exact` (ROADMAP D9) — applies to
//! [`convolve_psf`] and to [`convolve_gaussian_axis`] when
//! `renormalize = true`. With `renormalize = false`,
//! `convolve_gaussian_axis` intentionally uses the bare matched-filter
//! correlation path, so non-finite samples follow ordinary floating-point
//! propagation instead of being excluded.

use ndarray::{Array2, ArrayView2, Axis};

use crate::convolve::{
    Boundary, GaussianSampling, Normalization, conv_axis, conv_axis_renorm, conv2d_renorm,
    gaussian1d,
};
use crate::float::Float;

/// Gaussian truncation radius in units of σ (matches [`gaussian1d`]'s
/// default).
const TRUNCATE: f64 = 4.0;

/// Convolve `image` with a fixed, centered 2-D `psf` (true convolution).
///
/// The PSF is flipped (both axes reversed) and fed to the
/// normalized-convolution kernel, so this is genuine **convolution**,
/// not correlation (ROADMAP D3): an asymmetric ePSF keeps its
/// orientation. NaN / ±inf pixels are treated as missing and excluded
/// from both the weighted sum and its normalizer, and image edges are
/// likewise missing — this NaN-as-missing renormalization *is* the image
/// subsystem's convention (ROADMAP D9/D13), so there is deliberately
/// **no boundary parameter and no renormalize switch**: turning the
/// renorm off would only let a single NaN poison its whole neighborhood,
/// which has no valid use here (this is the intentional asymmetry with
/// [`convolve_gaussian_axis`], whose matched-filter use *must* be able
/// to disable renorm).
///
/// PSF normalization remains the caller's responsibility. Because the
/// renorm interior divides by `Σ psf`, a `Sum`-normalized PSF gives an
/// interior identity with renorm acting only at NaN/edges (flux
/// conserving); an unnormalized PSF yields a local weighted *mean*
/// rather than a flux-conserving convolution (not a bug — it is the
/// definition of the renorm).
///
/// Does **not** do: a noise model, a separable fast path, or an FFT
/// backend — all benchmark-driven and intentionally absent in v1
/// (ROADMAP D6).
///
/// Panics if `psf` does not have odd dimensions (a centered PSF needs a
/// well-defined middle tap).
pub fn convolve_psf<T: Float>(image: ArrayView2<T>, psf: ArrayView2<T>) -> Array2<T> {
    assert!(
        psf.nrows() % 2 == 1 && psf.ncols() % 2 == 1,
        "convolve_psf: psf must have odd dimensions (a centered PSF); got {}x{}",
        psf.nrows(),
        psf.ncols()
    );
    // Reverse both axes -> correlation with the flipped PSF is exactly
    // convolution with the original. The reversed view is zero-copy;
    // conv2d_renorm indexes it correctly through its strides.
    let flipped = psf.slice(ndarray::s![..;-1, ..;-1]);
    let (value, _weight) = conv2d_renorm(image, flipped);
    value
}

/// Correlate every lane along `axis` with a 1-D Gaussian of width
/// `sigma` — the grism emission-line matched filter (req3).
///
/// The kernel is built by [`gaussian1d`] with the requested
/// `normalization` (typically [`Normalization::L2`], which makes the
/// peak response the matched-filter S/N-optimal statistic). This is a
/// **correlation** (the kernel is not flipped); a Gaussian is symmetric
/// so it coincides with convolution anyway (ROADMAP D3).
///
/// `renormalize` selects the boundary model (ROADMAP D13):
/// - `false` — bare [`conv_axis`] using `boundary`. This is the matched
///   filter's natural mode: renormalization would distort the matched
///   response near the edges / missing data (ROADMAP D5 caveat), so it
///   must be switchable off (the reason this entry has the switch and
///   [`convolve_psf`] does not).
/// - `true` — NaN-as-missing [`conv_axis_renorm`]; out-of-bounds taps
///   are missing and `boundary` is **ignored** (renorm carries its own
///   mutually exclusive boundary model).
///
/// Does **not** do: the noise map needed for a true matched-filter S/N.
/// Out of scope here; a caller forms `S/N = corr / sqrt(conv(variance,
/// kernel²))` itself.
///
/// Panics if `sigma <= 0` (via [`gaussian1d`]) or if `axis` is not
/// `Axis(0)` or `Axis(1)`.
pub fn convolve_gaussian_axis<T: Float>(
    image: ArrayView2<T>,
    sigma: f64,
    axis: Axis,
    normalization: Normalization,
    boundary: Boundary,
    renormalize: bool,
) -> Array2<T> {
    let kernel = gaussian1d::<T>(
        sigma,
        TRUNCATE,
        GaussianSampling::ErfIntegrated,
        normalization,
    );
    if renormalize {
        let (value, _weight) = conv_axis_renorm(image, kernel.view(), axis);
        value
    } else {
        conv_axis(image, kernel.view(), axis, boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve::conv2d;
    use ndarray::{Array2, array};

    const TOL_F64: f64 = 1e-12;

    fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn true_convolution_keeps_psf_orientation_f64() {
        // Impulse image; a PSF with all its mass one row ABOVE center.
        // True convolution reproduces the PSF as-is at the impulse, so
        // the output spike lands one row above (2,2) -> (1,2). A
        // mistaken correlation (no flip) would put it one row below at
        // (3,2); assert that did NOT happen.
        let mut image = Array2::<f64>::zeros((5, 5));
        image[(2, 2)] = 1.0;
        let mut psf = Array2::<f64>::zeros((3, 3));
        psf[(0, 1)] = 1.0; // Σ = 1, already Sum-normalized
        let out = convolve_psf(image.view(), psf.view());
        assert!(approx_eq_f64(out[(1, 2)], 1.0, TOL_F64));
        assert!(approx_eq_f64(out[(3, 2)], 0.0, TOL_F64));
    }

    #[test]
    fn sum_normalized_psf_interior_equals_true_convolution_f64() {
        // NaN-free image, Sum-normalized PSF: at the interior D = Σpsf =
        // 1, so the renorm value equals plain convolution there.
        let mut image = Array2::<f64>::zeros((7, 7));
        for ((i, j), v) in image.indexed_iter_mut() {
            *v = (i * 5 + j * 3) as f64 * 0.25 - 2.0;
        }
        // Asymmetric, Sum-normalized 3x3 PSF.
        let mut psf = array![
            [0.05_f64, 0.10, 0.02],
            [0.15, 0.30, 0.08],
            [0.04, 0.18, 0.08]
        ];
        let sum: f64 = psf.iter().sum();
        psf.mapv_inplace(|v| v / sum);
        let out = convolve_psf(image.view(), psf.view());
        // Reference: true convolution = correlation with the flipped
        // PSF via the bare kernel.
        let flipped = psf.slice(ndarray::s![..;-1, ..;-1]);
        let reference = conv2d(image.view(), flipped, Boundary::Zero);
        for i in 1..6 {
            for j in 1..6 {
                assert!(
                    approx_eq_f64(out[(i, j)], reference[(i, j)], 1e-9),
                    "interior mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn renorm_does_not_darken_at_nan_or_edges_f64() {
        // Constant image with one NaN. The renorm reconstructs the NaN
        // pixel from valid neighbors and renormalizes the edges, so the
        // output is the same constant everywhere (no darkening).
        let constant = 4.2_f64;
        let mut image = Array2::<f64>::from_elem((9, 9), constant);
        image[(4, 4)] = f64::NAN;
        let mut psf = Array2::<f64>::from_elem((3, 3), 1.0);
        let sum: f64 = psf.iter().sum();
        psf.mapv_inplace(|v| v / sum); // Sum-normalized
        let out = convolve_psf(image.view(), psf.view());
        for v in out.iter() {
            assert!(v.is_finite());
            assert!(approx_eq_f64(*v, constant, 1e-9));
        }
    }

    #[test]
    #[should_panic(expected = "odd dimensions")]
    fn even_dim_psf_panics() {
        let image = Array2::<f64>::zeros((4, 4));
        let psf = Array2::<f64>::from_elem((2, 2), 0.25);
        let _ = convolve_psf(image.view(), psf.view());
    }

    #[test]
    fn matched_filter_peak_maximized_at_injected_sigma_f64() {
        // A Gaussian emission line along Axis(0) (replicated across
        // columns). With an L2-normalized template the matched-filter
        // peak response at the line center is maximal when the probe σ
        // equals the injected σ (Cauchy-Schwarz).
        let rows = 121usize;
        let cols = 4usize;
        let line_row = 60.0_f64;
        let injected_sigma = 5.0_f64;
        let mut image = Array2::<f64>::zeros((rows, cols));
        for i in 0..rows {
            let z = (i as f64 - line_row) / injected_sigma;
            let amplitude = (-0.5 * z * z).exp();
            for j in 0..cols {
                image[(i, j)] = amplitude;
            }
        }
        let peak_at = |probe_sigma: f64| {
            let response = convolve_gaussian_axis(
                image.view(),
                probe_sigma,
                Axis(0),
                Normalization::L2,
                Boundary::Zero,
                false,
            );
            response[(60, 0)]
        };
        let matched = peak_at(injected_sigma);
        assert!(matched > peak_at(injected_sigma * 0.5));
        assert!(matched > peak_at(injected_sigma * 2.0));
    }

    #[test]
    fn gaussian_axis_selects_the_requested_axis_f64() {
        // Each row is constant (the image varies only along Axis(0)).
        // Smoothing along Axis(1) (the constant direction) with a
        // Sum-normalized renormalized kernel is an identity; smoothing
        // along Axis(0) (the varying direction) changes the image.
        let rows = 9usize;
        let cols = 9usize;
        let mut image = Array2::<f64>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                image[(i, j)] = (i as f64 - 4.0).powi(2);
            }
        }
        let along_constant = convolve_gaussian_axis(
            image.view(),
            1.5,
            Axis(1),
            Normalization::Sum,
            Boundary::Zero,
            true,
        );
        for i in 0..rows {
            for j in 0..cols {
                assert!(approx_eq_f64(along_constant[(i, j)], image[(i, j)], 1e-9));
            }
        }
        let along_varying = convolve_gaussian_axis(
            image.view(),
            1.5,
            Axis(0),
            Normalization::Sum,
            Boundary::Zero,
            true,
        );
        let mut differs = false;
        for i in 0..rows {
            for j in 0..cols {
                if !approx_eq_f64(along_varying[(i, j)], image[(i, j)], 1e-6) {
                    differs = true;
                }
            }
        }
        assert!(differs, "smoothing the varying axis must change the image");
    }

    #[test]
    fn renormalize_switch_controls_edge_behavior_f64() {
        // Constant image. renormalize=true conserves the constant at the
        // edges; renormalize=false with Boundary::Zero darkens them.
        let image = Array2::<f64>::from_elem((20, 3), 5.0);
        let renormed = convolve_gaussian_axis(
            image.view(),
            2.0,
            Axis(0),
            Normalization::Sum,
            Boundary::Zero,
            true,
        );
        for v in renormed.iter() {
            assert!(approx_eq_f64(*v, 5.0, 1e-9));
        }
        let zero_pad = convolve_gaussian_axis(
            image.view(),
            2.0,
            Axis(0),
            Normalization::Sum,
            Boundary::Zero,
            false,
        );
        assert!(zero_pad[(0, 0)] < 5.0 - 1e-3, "edge should be darkened");
        assert!(
            approx_eq_f64(zero_pad[(10, 0)], 5.0, 1e-6),
            "interior should be unaffected"
        );
    }

    #[test]
    fn works_with_f32() {
        let mut image = Array2::<f32>::from_elem((6, 6), 3.0);
        image[(2, 3)] = f32::NAN;
        let psf = Array2::<f32>::from_elem((3, 3), 1.0 / 9.0);
        let out = convolve_psf(image.view(), psf.view());
        for v in out.iter() {
            assert!(v.is_finite());
            assert!((*v - 3.0).abs() <= 1e-4);
        }
        let line = convolve_gaussian_axis(
            image.view(),
            1.0,
            Axis(1),
            Normalization::L2,
            Boundary::Reflect,
            false,
        );
        assert_eq!(line.dim(), (6, 6));
    }
}

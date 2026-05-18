use std::f64::consts::SQRT_2;

use ndarray::Array1;

use crate::convolve::Normalization;
use crate::float::Float;

/// How the Gaussian profile is discretized onto integer taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianSampling {
    /// Per-bin integral of the unit Gaussian:
    /// `kernel[k] = Φ((k + 0.5)/σ) − Φ((k − 0.5)/σ)`.
    ///
    /// This is the default. For narrow σ (≲ 1.5 px) point sampling
    /// noticeably biases the kernel; the per-bin erf integral stays
    /// accurate (matching astropy's `Gaussian1DKernel` default).
    ErfIntegrated,
    /// Point sample of the unnormalized Gaussian at integer offsets:
    /// `kernel[k] = exp(−k² / (2σ²))`. Biased for small σ.
    PointSampled,
}

/// Build a 1-D Gaussian correlation kernel of odd length
/// `2 * ceil(truncate * sigma) + 1`, centered at index `len / 2` (so the
/// peak tap is the exact middle, consistent with [`super::conv1d`]'s
/// `center = kernel.len() / 2` convention).
///
/// `sigma` and `truncate` are in pixel units. The kernel is computed in
/// `f64` (libm `erf`) and cast to `T` at the end: the kernel is tiny, so
/// this keeps f32 and f64 kernels equally accurate and follows the
/// crate's "compute in f64, cast at the boundary" idiom (cf.
/// [`crate::image::reproject_exact`]).
///
/// `normalization` is the *kernel* normalization, decided here at
/// construction time. It is orthogonal to the boundary renormalization
/// the NaN wrappers apply (ROADMAP D4):
/// - [`Normalization::Sum`]: `Σ kernel = 1` — flux-conserving (req1 LSF,
///   req2 PSF).
/// - [`Normalization::L2`]: `Σ kernel² = 1` — matched-filter S/N optimal
///   (req3 grism line search).
/// - [`Normalization::None`]: raw samples, unscaled.
///
/// Panics if `sigma <= 0` or `truncate <= 0` — a kernel needs a positive
/// width, and downstream domain entries (e.g. `Spectrum::convolve_lsf`)
/// translate an invalid scientific resolution into a typed error before
/// reaching here.
pub fn gaussian1d<T: Float>(
    sigma: f64,
    truncate: f64,
    sampling: GaussianSampling,
    normalization: Normalization,
) -> Array1<T> {
    assert!(
        sigma > 0.0,
        "gaussian1d sigma must be positive, got {sigma}"
    );
    assert!(
        truncate > 0.0,
        "gaussian1d truncate must be positive, got {truncate}"
    );
    let radius = (truncate * sigma).ceil() as usize;
    let length = 2 * radius + 1;
    let mut kernel = Array1::<f64>::zeros(length);
    for tap in 0..length {
        let offset = tap as f64 - radius as f64;
        kernel[tap] = match sampling {
            GaussianSampling::ErfIntegrated => {
                normal_cdf((offset + 0.5) / sigma) - normal_cdf((offset - 0.5) / sigma)
            }
            GaussianSampling::PointSampled => (-(offset * offset) / (2.0 * sigma * sigma)).exp(),
        };
    }
    match normalization {
        Normalization::Sum => {
            let sum: f64 = kernel.iter().sum();
            kernel.mapv_inplace(|value| value / sum);
        }
        Normalization::L2 => {
            let l2_norm = kernel.iter().map(|value| value * value).sum::<f64>().sqrt();
            kernel.mapv_inplace(|value| value / l2_norm);
        }
        Normalization::None => {}
    }
    kernel.mapv(|value| T::from_f64(value).expect("gaussian kernel value fits in T"))
}

/// Standard normal CDF `Φ(x) = 0.5 * (1 + erf(x / √2))`.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_F64: f64 = 1e-12;
    const TOL_F32: f32 = 1e-6;

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL_F64 * a.abs().max(b.abs()).max(1.0)
    }

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL_F32 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn length_is_odd_and_matches_formula_f64() {
        for (sigma, truncate, expected_len) in [
            (1.0, 4.0, 9),  // radius ceil(4) = 4
            (2.0, 4.0, 17), // radius ceil(8) = 8
            (1.5, 4.0, 13), // radius ceil(6) = 6
            (0.3, 4.0, 5),  // radius ceil(1.2) = 2
            (1.0, 3.0, 7),  // radius ceil(3) = 3
        ] {
            let kernel = gaussian1d::<f64>(
                sigma,
                truncate,
                GaussianSampling::ErfIntegrated,
                Normalization::None,
            );
            assert_eq!(kernel.len(), expected_len);
            assert_eq!(kernel.len() % 2, 1, "kernel length must be odd");
        }
    }

    #[test]
    fn erf_integrated_known_values_sigma_one_f64() {
        // sigma = 1, truncate = 4 -> radius 4, length 9, center index 4.
        // kernel[c+k] = Φ((k+0.5)) − Φ((k−0.5)).
        let kernel = gaussian1d::<f64>(
            1.0,
            4.0,
            GaussianSampling::ErfIntegrated,
            Normalization::None,
        );
        let center = kernel.len() / 2;
        assert!(approx_eq_f64(kernel[center], 0.382_924_922_548_03));
        assert!(approx_eq_f64(kernel[center + 1], 0.241_730_337_457_13));
        assert!(approx_eq_f64(kernel[center + 2], 0.060_597_535_943_08));
    }

    #[test]
    fn kernel_is_symmetric_about_center_f64() {
        for sampling in [
            GaussianSampling::ErfIntegrated,
            GaussianSampling::PointSampled,
        ] {
            for normalization in [Normalization::Sum, Normalization::L2, Normalization::None] {
                let kernel = gaussian1d::<f64>(1.3, 4.0, sampling, normalization);
                let center = kernel.len() / 2;
                for k in 1..=center {
                    assert!(
                        approx_eq_f64(kernel[center + k], kernel[center - k]),
                        "asymmetric at k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn sum_normalization_sums_to_one_f64() {
        for sampling in [
            GaussianSampling::ErfIntegrated,
            GaussianSampling::PointSampled,
        ] {
            for sigma in [0.5, 1.0, 2.7] {
                let kernel = gaussian1d::<f64>(sigma, 4.0, sampling, Normalization::Sum);
                let sum: f64 = kernel.iter().sum();
                assert!(approx_eq_f64(sum, 1.0), "sigma={sigma} sum={sum}");
            }
        }
    }

    #[test]
    fn l2_normalization_sum_of_squares_is_one_f64() {
        for sampling in [
            GaussianSampling::ErfIntegrated,
            GaussianSampling::PointSampled,
        ] {
            for sigma in [0.5, 1.0, 2.7] {
                let kernel = gaussian1d::<f64>(sigma, 4.0, sampling, Normalization::L2);
                let sum_sq: f64 = kernel.iter().map(|v| v * v).sum();
                assert!(approx_eq_f64(sum_sq, 1.0), "sigma={sigma} ss={sum_sq}");
            }
        }
    }

    #[test]
    fn point_sampled_known_values_sigma_one_f64() {
        let kernel = gaussian1d::<f64>(
            1.0,
            4.0,
            GaussianSampling::PointSampled,
            Normalization::None,
        );
        let center = kernel.len() / 2;
        assert!(approx_eq_f64(kernel[center], 1.0)); // exp(0)
        assert!(approx_eq_f64(kernel[center + 1], (-0.5_f64).exp()));
        assert!(approx_eq_f64(kernel[center + 2], (-2.0_f64).exp()));
    }

    #[test]
    fn erf_and_point_sampling_differ_for_narrow_sigma_f64() {
        // ROADMAP D10: the two sampling modes are genuinely different,
        // most visibly for narrow sigma. Compare Sum-normalized kernels.
        let erf = gaussian1d::<f64>(
            0.5,
            4.0,
            GaussianSampling::ErfIntegrated,
            Normalization::Sum,
        );
        let point = gaussian1d::<f64>(0.5, 4.0, GaussianSampling::PointSampled, Normalization::Sum);
        assert_eq!(erf.len(), point.len());
        let max_abs_diff = erf
            .iter()
            .zip(point.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff > 1e-3,
            "sampling modes too close: max diff {max_abs_diff}"
        );
    }

    #[test]
    fn raw_erf_kernel_sum_approaches_one_with_wider_truncate_f64() {
        // Unnormalized erf kernel sums to Φ(R+0.5) − Φ(−R−0.5); the
        // missing tail shrinks as truncate grows.
        let sum4: f64 = gaussian1d::<f64>(
            1.0,
            4.0,
            GaussianSampling::ErfIntegrated,
            Normalization::None,
        )
        .iter()
        .sum();
        let sum6: f64 = gaussian1d::<f64>(
            1.0,
            6.0,
            GaussianSampling::ErfIntegrated,
            Normalization::None,
        )
        .iter()
        .sum();
        assert!(sum4 < 1.0 && sum4 > 0.999);
        assert!(sum6 < 1.0 && sum6 > sum4);
        // The missing mass is the two tails beyond ±(R + 0.5)σ. At
        // truncate = 6 (σ = 1) that is 2·Φ(−6.5) ≈ 8e-11, so 1e-9 is the
        // physically correct bound here, not floating-point epsilon.
        assert!((1.0 - sum6).abs() < 1e-9, "sum6={sum6}");
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn zero_sigma_panics() {
        let _ = gaussian1d::<f64>(
            0.0,
            4.0,
            GaussianSampling::ErfIntegrated,
            Normalization::Sum,
        );
    }

    #[test]
    #[should_panic(expected = "truncate must be positive")]
    fn zero_truncate_panics() {
        let _ = gaussian1d::<f64>(
            1.0,
            0.0,
            GaussianSampling::ErfIntegrated,
            Normalization::Sum,
        );
    }

    #[test]
    fn works_with_f32() {
        let kernel = gaussian1d::<f32>(
            1.0,
            4.0,
            GaussianSampling::ErfIntegrated,
            Normalization::Sum,
        );
        assert_eq!(kernel.len(), 9);
        let sum: f32 = kernel.iter().sum();
        assert!(approx_eq_f32(sum, 1.0));
        let center = kernel.len() / 2;
        for k in 1..=center {
            assert!(approx_eq_f32(kernel[center + k], kernel[center - k]));
        }
    }
}

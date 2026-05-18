//! Line-spread-function broadening of a noise-free template
//! ([`Spectrum::convolve_lsf`]).
//!
//! Both supported specs reduce to a single dimensionless quantity, the
//! Gaussian sigma **in `ln λ`**:
//!
//! - `ConstantR(R)`: a constant resolving power `R = λ / Δλ_FWHM`, so
//!   `σ_λ = λ / (R · 2√(2 ln 2))` and `σ_lnλ = σ_λ / λ = 1 / (R · 2√(2 ln 2))`.
//! - `ConstantVelocitySigma { sigma, speed_of_light }`: a constant
//!   velocity dispersion, so `σ_λ / λ = sigma / c` and
//!   `σ_lnλ = sigma / c`.
//!
//! In both cases `σ_λ(λ) = σ_lnλ · λ` — the kernel widens linearly with
//! wavelength. This is the single fact the whole module is built on.
//!
//! Two evaluation paths (ROADMAP D7), selected automatically; **any**
//! wavelength grid is accepted:
//!
//! - **Fast path** — the grid is `Log`-spaced *and* actually uniform in
//!   `ln λ` (`Grid::is_uniform`). Then `σ` is a constant number of
//!   pixels and the broadening collapses to a single fixed
//!   [`gaussian1d`] kernel applied once with edge renormalization.
//! - **General path** — anything else (linear, irregular, non-uniform
//!   `Log`). Each output pixel integrates an erf Gaussian of its own
//!   `σ_λ(λ_i)` over the actual (possibly non-uniform) bins and divides
//!   by the in-window weight. Exact and flux-conserving; it deliberately
//!   does *not* go through a `linear → log → linear` rebin round trip
//!   (which would irreversibly smear the red end before convolving).
//!
//! `is_uniform` is only the fast-path *discriminant*, never a rejection
//! gate: `Spacing::Log` is a midpoint convention and does not by itself
//! guarantee `ln λ` uniformity, so dispatching on `spacing == Log` alone
//! would silently miscompute a non-uniform log grid.
//!
//! The input is treated as a noise-free template (ROADMAP D8): only
//! `flux` is broadened. A spectrum carrying `error` or `mask` is
//! *rejected* ([`LsfError::NotATemplate`]), not silently stripped — an
//! observed spectrum already contains its instrumental LSF, so
//! re-convolving one is a scientific misuse, and silently dropping its
//! error/mask would be the crate's only silent data loss.

use ndarray::Array1;
use thiserror::Error;

use crate::bins::{GridKind, Spacing};
use crate::convolve::{Boundary, GaussianSampling, Normalization, conv1d, gaussian1d, normal_cdf};
use crate::float::Float;
use crate::spectroscopy::Spectrum;

/// Gaussian kernel truncation radius, in units of σ (matches
/// [`gaussian1d`]'s default and the general path's window).
const TRUNCATE: f64 = 4.0;

/// Relative per-step tolerance for accepting a `Log` grid as uniform in
/// `ln λ` (the fast-path discriminant). Tight enough that the
/// single-kernel approximation error is negligible, loose enough to
/// admit a genuinely log-uniform grid carrying float noise.
const UNIFORM_REL_TOL: f64 = 1e-6;

/// FWHM-to-sigma factor `2√(2 ln 2) ≈ 2.354820045` for a Gaussian.
#[inline]
fn fwhm_per_sigma() -> f64 {
    2.0 * (2.0 * std::f64::consts::LN_2).sqrt()
}

/// How the line-spread function widens with wavelength. Both variants
/// describe a Gaussian whose `σ / λ` ratio is constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LsfSpec {
    /// Constant resolving power `R = λ / Δλ_FWHM`.
    ConstantR(f64),
    /// Constant velocity dispersion. `sigma` and `speed_of_light` must
    /// share units (noobase does not track units; consistency is the
    /// caller's responsibility).
    ConstantVelocitySigma { sigma: f64, speed_of_light: f64 },
}

/// Errors from [`Spectrum::convolve_lsf`].
///
/// Algorithmic edge cases (grid spacing, non-uniformity, edge
/// truncation) are *not* errors — they are handled by automatic path
/// selection and per-pixel renormalization.
#[derive(Debug, Error, PartialEq)]
pub enum LsfError {
    #[error(
        "convolve_lsf input carries error/mask; the LSF applies only to noise-free templates — strip error and mask first"
    )]
    NotATemplate,
    #[error("LSF resolution must be positive (R > 0, or sigma > 0 with positive speed of light)")]
    InvalidResolution,
}

impl<T: Float> Spectrum<T> {
    /// Broaden the template with a Gaussian line-spread function.
    ///
    /// The input is treated as a noise-free model: only `flux` is
    /// convolved and the output carries no `error`/`mask`. The
    /// wavelength grid is returned unchanged (same values, spacing, and
    /// kind). Flux is conserved, including at the spectrum edges (each
    /// output pixel is renormalized by the in-window kernel weight).
    ///
    /// Works for *any* wavelength grid (see the module docs for the
    /// fast/general path selection). Precondition: wavelengths are
    /// physical (strictly positive); `Grid` already guarantees strict
    /// monotonicity.
    ///
    /// # Errors
    ///
    /// - [`LsfError::NotATemplate`] if the spectrum carries `error` or
    ///   `mask` (ROADMAP D8).
    /// - [`LsfError::InvalidResolution`] if `R <= 0`, or `sigma <= 0`,
    ///   or `speed_of_light <= 0`.
    pub fn convolve_lsf(&self, spec: LsfSpec) -> Result<Spectrum<T>, LsfError> {
        // D8: an LSF target is noise-free by construction. Reject (do
        // not silently drop) error/mask.
        if self.error().is_some() || self.mask().is_some() {
            return Err(LsfError::NotATemplate);
        }

        let sigma_ln_lambda = match spec {
            LsfSpec::ConstantR(resolving_power) => {
                if resolving_power <= 0.0 {
                    return Err(LsfError::InvalidResolution);
                }
                1.0 / (resolving_power * fwhm_per_sigma())
            }
            LsfSpec::ConstantVelocitySigma {
                sigma,
                speed_of_light,
            } => {
                if sigma <= 0.0 || speed_of_light <= 0.0 {
                    return Err(LsfError::InvalidResolution);
                }
                sigma / speed_of_light
            }
        };

        let bin_count = self.n_bins();
        let wavelength = self.wavelength();

        // Bin-center wavelengths (the convolution axis) and bin edges
        // (the integration limits for the general path).
        let centers_grid = match wavelength.kind() {
            GridKind::Centers => None,
            GridKind::Edges => Some(wavelength.to_centers()),
        };
        let centers = match &centers_grid {
            Some(grid) => grid.values(),
            None => wavelength.values(),
        };

        let use_fast_path = wavelength.spacing() == Spacing::Log
            && wavelength
                .is_uniform(T::from_f64(UNIFORM_REL_TOL).expect("uniform tolerance fits in T"))
            && bin_count >= 2;

        let new_flux = if use_fast_path {
            self.lsf_fast_path(sigma_ln_lambda, centers)
        } else {
            self.lsf_general_path(sigma_ln_lambda, centers)
        };

        // Lengths are guaranteed to match (built with `bin_count`
        // entries); `Spectrum::new` would only re-validate.
        Ok(Spectrum::new(wavelength.clone(), new_flux, None, None)
            .expect("convolved flux length equals wavelength bin count"))
    }

    /// `Log`-uniform fast path: one fixed Gaussian kernel (σ constant in
    /// pixels because the grid is uniform in `ln λ`), applied with edge
    /// renormalization so flux is conserved at the borders too.
    fn lsf_fast_path(&self, sigma_ln_lambda: f64, centers: ndarray::ArrayView1<T>) -> Array1<T> {
        let bin_count = centers.len();
        // Uniform step in ln λ between adjacent bins.
        let ln_first = centers[0].to_f64().expect("wavelength fits in f64").ln();
        let ln_last = centers[bin_count - 1]
            .to_f64()
            .expect("wavelength fits in f64")
            .ln();
        let ln_step = (ln_last - ln_first) / (bin_count as f64 - 1.0);
        let sigma_pixels = sigma_ln_lambda / ln_step;

        let kernel = gaussian1d::<T>(
            sigma_pixels,
            TRUNCATE,
            GaussianSampling::ErfIntegrated,
            Normalization::Sum,
        );
        // out = correlate(flux) / correlate(ones): interior pixels see a
        // unit-sum kernel (== plain correlation); edge pixels are
        // renormalized by the in-bounds weight, conserving flux. No NaN
        // is possible (template), so this never divides by zero — the
        // center tap alone keeps the denominator positive everywhere.
        let flux = self.flux();
        let numerator = conv1d(flux, kernel.view(), Boundary::Zero);
        let ones = Array1::<T>::from_elem(bin_count, T::one());
        let denominator = conv1d(ones.view(), kernel.view(), Boundary::Zero);
        let mut output = Array1::<T>::zeros(bin_count);
        for index in 0..bin_count {
            output[index] = numerator[index] / denominator[index];
        }
        output
    }

    /// General path: per output pixel, integrate an erf Gaussian of that
    /// pixel's own `σ_λ(λ_i) = σ_lnλ · λ_i` over the actual bins and
    /// normalize by the in-window weight. Exact for any (non-uniform)
    /// grid and flux-conserving including at the edges.
    fn lsf_general_path(&self, sigma_ln_lambda: f64, centers: ndarray::ArrayView1<T>) -> Array1<T> {
        let bin_count = centers.len();
        let wavelength = self.wavelength();
        let edges_grid = wavelength.to_edges();
        let edges = edges_grid.values();
        let flux = self.flux();

        let mut output = Array1::<T>::zeros(bin_count);
        for output_index in 0..bin_count {
            let lambda = centers[output_index]
                .to_f64()
                .expect("wavelength fits in f64");
            let sigma = sigma_ln_lambda * lambda;
            let half_window = TRUNCATE * sigma;
            let low_wave = lambda - half_window;
            let high_wave = lambda + half_window;

            // Window = bins whose center is within ±truncate·σ of λ_i
            // (same truncation convention as gaussian1d's tap radius).
            let mut numerator = 0.0_f64;
            let mut denominator = 0.0_f64;
            let mut bin = lower_bound(centers, low_wave);
            while bin < bin_count {
                let center = centers[bin].to_f64().expect("wavelength fits in f64");
                if center > high_wave {
                    break;
                }
                let edge_lo = edges[bin].to_f64().expect("edge fits in f64");
                let edge_hi = edges[bin + 1].to_f64().expect("edge fits in f64");
                // Integral of the unit Gaussian (mean λ_i, sigma σ) over
                // this bin: the same erf source as gaussian1d.
                let weight =
                    normal_cdf((edge_hi - lambda) / sigma) - normal_cdf((edge_lo - lambda) / sigma);
                numerator += weight * flux[bin].to_f64().expect("flux fits in f64");
                denominator += weight;
                bin += 1;
            }

            // The bin containing λ_i always carries positive Gaussian
            // mass, so the denominator is positive in practice; the
            // guard only defends a pathologically degenerate window.
            let value = if denominator > 0.0 {
                numerator / denominator
            } else {
                flux[output_index].to_f64().expect("flux fits in f64")
            };
            output[output_index] = T::from_f64(value).expect("convolved flux fits in T");
        }
        output
    }
}

/// Smallest index `i` with `centers[i].to_f64() >= target`
/// (`centers` is strictly increasing).
#[inline]
fn lower_bound<T: Float>(centers: ndarray::ArrayView1<T>, target: f64) -> usize {
    let mut low = 0usize;
    let mut high = centers.len();
    while low < high {
        let mid = (low + high) / 2;
        if centers[mid].to_f64().expect("wavelength fits in f64") < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{Grid, GridKind, Spacing};
    use ndarray::Array1;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn centers_log(start: f64, end: f64, n: usize) -> Grid<f64> {
        Grid::<f64>::logspace(start, end, n, GridKind::Centers)
    }

    fn centers_linear(start: f64, end: f64, n: usize) -> Grid<f64> {
        Grid::<f64>::linspace(start, end, n, GridKind::Centers)
    }

    #[test]
    fn rejects_input_with_error() {
        let wavelength = centers_log(4000.0, 7000.0, 50);
        let flux = Array1::<f64>::from_elem(50, 1.0);
        let error = Array1::<f64>::from_elem(50, 0.1);
        let spectrum = Spectrum::new(wavelength, flux, Some(error), None).unwrap();
        let err = spectrum
            .convolve_lsf(LsfSpec::ConstantR(2000.0))
            .unwrap_err();
        assert_eq!(err, LsfError::NotATemplate);
    }

    #[test]
    fn rejects_input_with_mask() {
        let wavelength = centers_log(4000.0, 7000.0, 50);
        let flux = Array1::<f64>::from_elem(50, 1.0);
        let mask = Array1::<bool>::from_elem(50, false);
        let spectrum = Spectrum::new(wavelength, flux, None, Some(mask)).unwrap();
        let err = spectrum
            .convolve_lsf(LsfSpec::ConstantR(2000.0))
            .unwrap_err();
        assert_eq!(err, LsfError::NotATemplate);
    }

    #[test]
    fn rejects_invalid_resolution() {
        let wavelength = centers_log(4000.0, 7000.0, 20);
        let flux = Array1::<f64>::from_elem(20, 1.0);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        for spec in [
            LsfSpec::ConstantR(0.0),
            LsfSpec::ConstantR(-100.0),
            LsfSpec::ConstantVelocitySigma {
                sigma: 0.0,
                speed_of_light: 299792.458,
            },
            LsfSpec::ConstantVelocitySigma {
                sigma: 50.0,
                speed_of_light: 0.0,
            },
            LsfSpec::ConstantVelocitySigma {
                sigma: 50.0,
                speed_of_light: -1.0,
            },
        ] {
            assert_eq!(
                spectrum.convolve_lsf(spec).unwrap_err(),
                LsfError::InvalidResolution
            );
        }
    }

    #[test]
    fn flux_only_input_is_accepted() {
        let wavelength = centers_log(4000.0, 7000.0, 30);
        let flux = Array1::<f64>::from_elem(30, 2.0);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        assert!(spectrum.convolve_lsf(LsfSpec::ConstantR(3000.0)).is_ok());
    }

    #[test]
    fn constant_template_stays_constant_fast_path_f64() {
        // Log-uniform grid -> fast path. Constant template must stay
        // constant everywhere, edges included (flux conservation).
        let wavelength = centers_log(4000.0, 7000.0, 256);
        let flux = Array1::<f64>::from_elem(256, 3.7);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum.convolve_lsf(LsfSpec::ConstantR(2000.0)).unwrap();
        for value in out.flux().iter() {
            assert!(approx_eq(*value, 3.7, 1e-9));
        }
    }

    #[test]
    fn constant_template_stays_constant_general_path_f64() {
        // Linear grid -> general path. Same conservation requirement.
        let wavelength = centers_linear(4000.0, 7000.0, 200);
        let flux = Array1::<f64>::from_elem(200, -1.5);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum
            .convolve_lsf(LsfSpec::ConstantVelocitySigma {
                sigma: 60.0,
                speed_of_light: 299792.458,
            })
            .unwrap();
        for value in out.flux().iter() {
            assert!(approx_eq(*value, -1.5, 1e-9));
        }
    }

    #[test]
    fn output_preserves_grid_and_drops_error_mask() {
        let wavelength = centers_linear(5000.0, 6000.0, 40);
        let flux: Array1<f64> = (0..40).map(|i| (i as f64).sin() + 2.0).collect();
        let spectrum = Spectrum::new(wavelength.clone(), flux, None, None).unwrap();
        let out = spectrum.convolve_lsf(LsfSpec::ConstantR(5000.0)).unwrap();
        assert_eq!(out.wavelength().kind(), wavelength.kind());
        assert_eq!(out.wavelength().spacing(), wavelength.spacing());
        assert_eq!(out.wavelength().len(), wavelength.len());
        for i in 0..wavelength.len() {
            assert_eq!(out.wavelength().values()[i], wavelength.values()[i]);
        }
        assert!(out.error().is_none());
        assert!(out.mask().is_none());
    }

    #[test]
    fn line_flux_is_conserved_for_interior_line_general_path() {
        // A Gaussian emission line well inside a uniform linear grid:
        // the LSF must conserve the integrated flux (constant bin width,
        // so Σ flux is invariant). Tight bound because the line and all
        // its broadened mass stay far from the edges.
        let n = 1200;
        let wavelength = centers_linear(5000.0, 6200.0, n);
        let step = (6200.0 - 5000.0) / (n as f64 - 1.0);
        let line_center = 5600.0;
        let line_sigma = 4.0;
        let flux: Array1<f64> = (0..n)
            .map(|i| {
                let lambda = 5000.0 + step * i as f64;
                let z = (lambda - line_center) / line_sigma;
                (-0.5 * z * z).exp()
            })
            .collect();
        let total_in: f64 = flux.iter().sum();
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum.convolve_lsf(LsfSpec::ConstantR(2000.0)).unwrap();
        let total_out: f64 = out.flux().iter().sum();
        assert!(
            approx_eq(total_out, total_in, 2e-3),
            "flux not conserved: in={total_in} out={total_out}"
        );
    }

    #[test]
    fn broadening_adds_in_quadrature_general_path() {
        // Gaussian line convolved by a Gaussian LSF -> Gaussian with
        // σ_out² = σ_line² + σ_lsf², σ_lsf = λ0 / (R · 2√(2ln2)).
        // Measured via the flux-weighted second moment.
        let n = 2000;
        let (lo, hi) = (5200.0_f64, 5800.0_f64);
        let wavelength = centers_linear(lo, hi, n);
        let step = (hi - lo) / (n as f64 - 1.0);
        let line_center = 5500.0;
        let line_sigma = 5.0;
        let flux: Array1<f64> = (0..n)
            .map(|i| {
                let lambda = lo + step * i as f64;
                let z = (lambda - line_center) / line_sigma;
                (-0.5 * z * z).exp()
            })
            .collect();
        let resolving_power = 1500.0;
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum
            .convolve_lsf(LsfSpec::ConstantR(resolving_power))
            .unwrap();

        let out_flux = out.flux();
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        for i in 0..n {
            let lambda = lo + step * i as f64;
            sum_w += out_flux[i];
            sum_wx += out_flux[i] * lambda;
        }
        let mean = sum_wx / sum_w;
        let mut sum_wxx = 0.0;
        for i in 0..n {
            let lambda = lo + step * i as f64;
            sum_wxx += out_flux[i] * (lambda - mean).powi(2);
        }
        let measured_sigma = (sum_wxx / sum_w).sqrt();
        let sigma_lsf = line_center / (resolving_power * fwhm_per_sigma());
        let expected_sigma = (line_sigma * line_sigma + sigma_lsf * sigma_lsf).sqrt();
        assert!(
            approx_eq(measured_sigma, expected_sigma, 1e-2),
            "measured σ {measured_sigma} vs expected {expected_sigma}"
        );
    }

    #[test]
    fn fast_path_and_general_path_agree_on_same_sigma() {
        // ROADMAP §7: identical wavelength values, one tagged Log
        // (fast path) and one tagged Linear (general path). Under the
        // same physical σ_λ ∝ λ the two paths must agree closely (they
        // differ only by ln-space vs λ-space discretization).
        let n = 400;
        let values: Array1<f64> = {
            let log_lo = 4500.0_f64.ln();
            let log_hi = 6500.0_f64.ln();
            (0..n)
                .map(|i| (log_lo + (log_hi - log_lo) * i as f64 / (n as f64 - 1.0)).exp())
                .collect()
        };
        let template: Array1<f64> = values
            .iter()
            .map(|&lambda| 1.0 + (-0.5 * ((lambda - 5500.0) / 80.0).powi(2)).exp())
            .collect();

        let grid_log = Grid::new(values.clone(), Spacing::Log, GridKind::Centers).unwrap();
        let grid_lin = Grid::new(values.clone(), Spacing::Linear, GridKind::Centers).unwrap();
        let spec = LsfSpec::ConstantR(3000.0);
        let out_fast = Spectrum::new(grid_log, template.clone(), None, None)
            .unwrap()
            .convolve_lsf(spec)
            .unwrap();
        let out_general = Spectrum::new(grid_lin, template.clone(), None, None)
            .unwrap()
            .convolve_lsf(spec)
            .unwrap();

        let fast = out_fast.flux();
        let general = out_general.flux();
        // Compare the interior (away from edge truncation differences).
        for i in 20..(n - 20) {
            assert!(
                approx_eq(fast[i], general[i], 1e-3),
                "path mismatch at {i}: fast {} general {}",
                fast[i],
                general[i]
            );
        }
    }

    #[test]
    fn single_bin_spectrum_is_identity() {
        // n_bins = 1 (Edges grid of length 2): the only bin is its own
        // window, so flux is returned unchanged.
        let wavelength = Grid::new(
            Array1::from(vec![5000.0_f64, 5010.0]),
            Spacing::Linear,
            GridKind::Edges,
        )
        .unwrap();
        let flux = Array1::from(vec![42.0_f64]);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum.convolve_lsf(LsfSpec::ConstantR(2000.0)).unwrap();
        assert_eq!(out.n_bins(), 1);
        assert!(approx_eq(out.flux()[0], 42.0, 1e-12));
    }

    #[test]
    fn works_with_f32() {
        let wavelength = Grid::<f32>::logspace(4000.0, 7000.0, 128, GridKind::Centers);
        let flux = Array1::<f32>::from_elem(128, 5.0);
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let out = spectrum
            .convolve_lsf(LsfSpec::ConstantVelocitySigma {
                sigma: 70.0,
                speed_of_light: 299792.458,
            })
            .unwrap();
        for value in out.flux().iter() {
            assert!((*value - 5.0).abs() <= 1e-3);
        }
    }
}

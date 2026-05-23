use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::axis::Grid;
use crate::float::Float;

/// Photometric integration convention.
///
/// `PhotonCounting` weights the integrand with an additional `lambda` factor
/// (appropriate for photon-counting detectors such as CCDs and HgCdTe arrays).
/// `EnergyWeighted` omits that factor (appropriate for bolometric /
/// energy-integrating detectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotometryConvention {
    PhotonCounting,
    EnergyWeighted,
}

/// Result of a synthetic photometry evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotometryResult<T: Float> {
    pub band_flux: T,
    pub band_error: Option<T>,
    pub coverage: T,
}

#[derive(Debug, Error, PartialEq)]
pub enum PhotometryError {
    #[error("spectrum_flux length {flux} does not match spectrum bin count {expected}")]
    SpectrumFluxLengthMismatch { flux: usize, expected: usize },
    #[error("spectrum_error length {error} does not match spectrum bin count {expected}")]
    SpectrumErrorLengthMismatch { error: usize, expected: usize },
    #[error("transmission_values length {values} does not match transmission bin count {expected}")]
    TransmissionLengthMismatch { values: usize, expected: usize },
    #[error("spectrum and transmission curves do not overlap in wavelength")]
    NoOverlap,
}

/// Compute synthetic photometry of a spectrum through a transmission curve.
///
/// Returns the band-averaged flux density, the propagated 1-sigma uncertainty
/// (only when `spectrum_error` is `Some`), and the geometric coverage of the
/// filter by the spectrum.
///
/// # Conventions
///
/// `PhotonCounting` is the appropriate choice for photon-counting detectors
/// (CCDs, HgCdTe arrays including JWST NIRCam). The integrand carries an
/// additional `lambda` factor that accounts for the conversion from energy
/// density to photon rate.
///
/// `EnergyWeighted` is appropriate for bolometric / energy-integrating
/// detectors, where the integrand has no `lambda` factor.
///
/// # Coverage
///
/// `coverage` is the fraction of the filter's transmission integral that is
/// probed by the spectrum:
/// `coverage = (integral over spectrum ∩ filter range) / (integral over full filter range)`,
/// using the same weighting convention as the flux integral. The returned
/// `band_flux` is normalized over the covered overlap only; when coverage is
/// below 1.0 it is therefore a partial-band estimate whose bias depends on the
/// unobserved SED/filter region. SED-fitting callers should threshold on this
/// (for example, require coverage > 0.999).
///
/// # Error propagation
///
/// The 1-sigma error is propagated assuming spectrum bins are statistically
/// independent. The transmission curve's bin density does not affect this
/// assumption — it only enters the deterministic weights. The assumption is
/// violated if the spectrum was previously upsampled (e.g. via `Spectrum::rebin`
/// onto a finer grid), in which case adjacent bins share noise and the
/// returned `band_error` underestimates the true uncertainty. See
/// `Spectrum::rebin` for the same caveat.
///
/// # Errors
///
/// Returns `PhotometryError::SpectrumFluxLengthMismatch` /
/// `SpectrumErrorLengthMismatch` / `TransmissionLengthMismatch` if input array
/// lengths do not match the corresponding grids' bin counts. Returns
/// `PhotometryError::NoOverlap` if the two grids do not overlap in wavelength.
pub fn synthetic<T: Float>(
    spectrum_grid: &Grid<T>,
    spectrum_flux: ArrayView1<T>,
    spectrum_error: Option<ArrayView1<T>>,
    transmission_grid: &Grid<T>,
    transmission_values: ArrayView1<T>,
    convention: PhotometryConvention,
) -> Result<PhotometryResult<T>, PhotometryError> {
    let spectrum_edges_grid = spectrum_grid.to_edges();
    let transmission_edges_grid = transmission_grid.to_edges();
    let spectrum_edges = spectrum_edges_grid.values();
    let transmission_edges = transmission_edges_grid.values();
    let spectrum_bin_count = spectrum_edges.len() - 1;
    let transmission_bin_count = transmission_edges.len() - 1;

    if spectrum_flux.len() != spectrum_bin_count {
        return Err(PhotometryError::SpectrumFluxLengthMismatch {
            flux: spectrum_flux.len(),
            expected: spectrum_bin_count,
        });
    }
    if let Some(error_view) = spectrum_error.as_ref()
        && error_view.len() != spectrum_bin_count
    {
        return Err(PhotometryError::SpectrumErrorLengthMismatch {
            error: error_view.len(),
            expected: spectrum_bin_count,
        });
    }
    if transmission_values.len() != transmission_bin_count {
        return Err(PhotometryError::TransmissionLengthMismatch {
            values: transmission_values.len(),
            expected: transmission_bin_count,
        });
    }

    let two = T::from_usize(2).expect("2 fits in T");

    // Full-filter denominator: walk the transmission edges alone, treating
    // each transmission bin as standalone with its own center and width.
    let mut denominator_full = T::zero();
    for transmission_index in 0..transmission_bin_count {
        let lo = transmission_edges[transmission_index];
        let hi = transmission_edges[transmission_index + 1];
        let lambda_mid = (lo + hi) / two;
        let width = hi - lo;
        let transmission_value = transmission_values[transmission_index];
        let contribution = match convention {
            PhotometryConvention::PhotonCounting => transmission_value * lambda_mid * width,
            PhotometryConvention::EnergyWeighted => transmission_value * width,
        };
        denominator_full = denominator_full + contribution;
    }

    // Two-pointer walk over spectrum and transmission edges.
    let mut numerator = T::zero();
    let mut denominator = T::zero();
    let mut per_spectrum_weight: Option<Array1<T>> = spectrum_error
        .as_ref()
        .map(|_| Array1::<T>::zeros(spectrum_bin_count));

    if spectrum_bin_count > 0 && transmission_bin_count > 0 {
        let mut spectrum_index = 0usize;
        let mut transmission_index = 0usize;
        while spectrum_index < spectrum_bin_count && transmission_index < transmission_bin_count {
            let spectrum_lo = spectrum_edges[spectrum_index];
            let spectrum_hi = spectrum_edges[spectrum_index + 1];
            let transmission_lo = transmission_edges[transmission_index];
            let transmission_hi = transmission_edges[transmission_index + 1];
            let lo = if spectrum_lo > transmission_lo {
                spectrum_lo
            } else {
                transmission_lo
            };
            let hi = if spectrum_hi < transmission_hi {
                spectrum_hi
            } else {
                transmission_hi
            };
            if hi > lo {
                let lambda_mid = (lo + hi) / two;
                let width = hi - lo;
                let transmission_value = transmission_values[transmission_index];
                let weight = match convention {
                    PhotometryConvention::PhotonCounting => transmission_value * lambda_mid * width,
                    PhotometryConvention::EnergyWeighted => transmission_value * width,
                };
                numerator = numerator + spectrum_flux[spectrum_index] * weight;
                denominator = denominator + weight;
                if let Some(weights) = per_spectrum_weight.as_mut() {
                    weights[spectrum_index] = weights[spectrum_index] + weight;
                }
            }
            // Advance whichever bin ends first; if both end at the same edge,
            // advance both to avoid a zero-width pair on the next iteration.
            if spectrum_hi < transmission_hi {
                spectrum_index += 1;
            } else if transmission_hi < spectrum_hi {
                transmission_index += 1;
            } else {
                spectrum_index += 1;
                transmission_index += 1;
            }
        }
    }

    if denominator == T::zero() {
        return Err(PhotometryError::NoOverlap);
    }

    let band_flux = numerator / denominator;
    let coverage = if denominator_full > T::zero() {
        denominator / denominator_full
    } else {
        T::zero()
    };

    let band_error = per_spectrum_weight.as_ref().map(|weights| {
        let error_view = spectrum_error.expect("error view is present when weights are tracked");
        let mut band_variance = T::zero();
        for spectrum_index in 0..spectrum_bin_count {
            let weight = weights[spectrum_index];
            if weight != T::zero() {
                let normalized = weight / denominator;
                let sigma = error_view[spectrum_index];
                band_variance = band_variance + normalized * normalized * sigma * sigma;
            }
        }
        band_variance.sqrt()
    });

    Ok(PhotometryResult {
        band_flux,
        band_error,
        coverage,
    })
}

/// Result of [`SyntheticOperator::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApplyResult<T: Float> {
    pub band_flux: T,
    pub band_error: Option<T>,
}

/// Internal per-spectrum-bin record used by [`SyntheticOperator`]. Stores the
/// already-normalized weight and its square so `apply` reduces to one
/// multiply-add per record without any further division.
#[derive(Debug)]
struct Contribution<T: Float> {
    spec_index: usize,
    normalized_weight: T,
    squared_weight: T,
}

/// Pre-baked photometry operator for hot-path use.
///
/// Construction performs the two-pointer walk over both grids once, caching
/// per-spectrum-bin normalized weights (and their squares for error
/// propagation). Subsequent `apply` calls reduce to a sparse dot product over
/// only the spectrum bins that contribute to the band, with zero allocation
/// and no grid traversal.
///
/// Use when the spectrum and transmission grids are fixed across many
/// evaluations (e.g. SED fitting at fixed redshift, where only the model flux
/// changes). For varying grids (e.g. redshift fitting that shifts the filter
/// in the rest frame), use the free function `synthetic` which recomputes
/// per call.
///
/// The mathematical result is identical to `synthetic` for matching inputs;
/// the operator only changes when work happens, not what it computes.
#[derive(Debug)]
pub struct SyntheticOperator<T: Float> {
    contributions: Vec<Contribution<T>>,
    coverage: T,
    spectrum_bin_count: usize,
}

impl<T: Float> SyntheticOperator<T> {
    /// Build the operator by walking both grids once and caching the
    /// normalized per-spectrum-bin weights.
    pub fn new(
        spectrum_grid: &Grid<T>,
        transmission_grid: &Grid<T>,
        transmission_values: ArrayView1<T>,
        convention: PhotometryConvention,
    ) -> Result<Self, PhotometryError> {
        let spectrum_edges_grid = spectrum_grid.to_edges();
        let transmission_edges_grid = transmission_grid.to_edges();
        let spectrum_edges = spectrum_edges_grid.values();
        let transmission_edges = transmission_edges_grid.values();
        let spectrum_bin_count = spectrum_edges.len() - 1;
        let transmission_bin_count = transmission_edges.len() - 1;

        if transmission_values.len() != transmission_bin_count {
            return Err(PhotometryError::TransmissionLengthMismatch {
                values: transmission_values.len(),
                expected: transmission_bin_count,
            });
        }

        let two = T::from_usize(2).expect("2 fits in T");

        let mut denominator_full = T::zero();
        for transmission_index in 0..transmission_bin_count {
            let lo = transmission_edges[transmission_index];
            let hi = transmission_edges[transmission_index + 1];
            let lambda_mid = (lo + hi) / two;
            let width = hi - lo;
            let transmission_value = transmission_values[transmission_index];
            let contribution = match convention {
                PhotometryConvention::PhotonCounting => transmission_value * lambda_mid * width,
                PhotometryConvention::EnergyWeighted => transmission_value * width,
            };
            denominator_full = denominator_full + contribution;
        }

        let mut per_spectrum_weight = Array1::<T>::zeros(spectrum_bin_count);
        let mut denominator = T::zero();

        if spectrum_bin_count > 0 && transmission_bin_count > 0 {
            let mut spectrum_index = 0usize;
            let mut transmission_index = 0usize;
            while spectrum_index < spectrum_bin_count && transmission_index < transmission_bin_count
            {
                let spectrum_lo = spectrum_edges[spectrum_index];
                let spectrum_hi = spectrum_edges[spectrum_index + 1];
                let transmission_lo = transmission_edges[transmission_index];
                let transmission_hi = transmission_edges[transmission_index + 1];
                let lo = if spectrum_lo > transmission_lo {
                    spectrum_lo
                } else {
                    transmission_lo
                };
                let hi = if spectrum_hi < transmission_hi {
                    spectrum_hi
                } else {
                    transmission_hi
                };
                if hi > lo {
                    let lambda_mid = (lo + hi) / two;
                    let width = hi - lo;
                    let transmission_value = transmission_values[transmission_index];
                    let weight = match convention {
                        PhotometryConvention::PhotonCounting => {
                            transmission_value * lambda_mid * width
                        }
                        PhotometryConvention::EnergyWeighted => transmission_value * width,
                    };
                    denominator = denominator + weight;
                    per_spectrum_weight[spectrum_index] =
                        per_spectrum_weight[spectrum_index] + weight;
                }
                if spectrum_hi < transmission_hi {
                    spectrum_index += 1;
                } else if transmission_hi < spectrum_hi {
                    transmission_index += 1;
                } else {
                    spectrum_index += 1;
                    transmission_index += 1;
                }
            }
        }

        if denominator == T::zero() {
            return Err(PhotometryError::NoOverlap);
        }

        let coverage = if denominator_full > T::zero() {
            denominator / denominator_full
        } else {
            T::zero()
        };

        let mut contributions = Vec::new();
        for spectrum_index in 0..spectrum_bin_count {
            let weight = per_spectrum_weight[spectrum_index];
            if weight != T::zero() {
                let normalized = weight / denominator;
                let squared = normalized * normalized;
                contributions.push(Contribution {
                    spec_index: spectrum_index,
                    normalized_weight: normalized,
                    squared_weight: squared,
                });
            }
        }

        Ok(Self {
            contributions,
            coverage,
            spectrum_bin_count,
        })
    }

    /// Apply the operator to a spectrum flux array, optionally with errors.
    ///
    /// Returns `band_flux` and (if `spectrum_error` is `Some`) the propagated
    /// 1-sigma `band_error`. Coverage is a property of the grids and is available
    /// via `coverage()`.
    ///
    /// The error propagation assumes spectrum bins are statistically independent.
    /// See `synthetic` for the full caveat.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum_flux.len() != self.spectrum_bin_count()`, or if
    /// `spectrum_error` is `Some` and its length does not match.
    pub fn apply(
        &self,
        spectrum_flux: ArrayView1<T>,
        spectrum_error: Option<ArrayView1<T>>,
    ) -> ApplyResult<T> {
        assert_eq!(
            spectrum_flux.len(),
            self.spectrum_bin_count,
            "spectrum_flux length {} does not match operator's spectrum bin count {}",
            spectrum_flux.len(),
            self.spectrum_bin_count
        );
        if let Some(error_view) = spectrum_error.as_ref() {
            assert_eq!(
                error_view.len(),
                self.spectrum_bin_count,
                "spectrum_error length {} does not match operator's spectrum bin count {}",
                error_view.len(),
                self.spectrum_bin_count
            );
        }

        let mut band_flux = T::zero();
        let mut band_variance = T::zero();
        for contribution in &self.contributions {
            band_flux =
                band_flux + spectrum_flux[contribution.spec_index] * contribution.normalized_weight;
            if let Some(error_view) = spectrum_error.as_ref() {
                let sigma = error_view[contribution.spec_index];
                band_variance = band_variance + contribution.squared_weight * sigma * sigma;
            }
        }

        let band_error = spectrum_error.map(|_| band_variance.sqrt());
        ApplyResult {
            band_flux,
            band_error,
        }
    }

    /// Coverage fraction (see [`synthetic`] for definition). Computed once at
    /// construction.
    pub fn coverage(&self) -> T {
        self.coverage
    }

    /// Number of spectrum bins the operator was constructed against. `apply`
    /// requires a flux array of exactly this length.
    pub fn spectrum_bin_count(&self) -> usize {
        self.spectrum_bin_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axis::{GridKind, Spacing};
    use ndarray::array;

    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-5;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn approx_eq_f32(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn linear_edges_f64(values: &[f64]) -> Grid<f64> {
        Grid::new(
            values.iter().copied().collect(),
            Spacing::Linear,
            GridKind::Edges,
        )
        .unwrap()
    }

    fn linear_edges_f32(values: &[f32]) -> Grid<f32> {
        Grid::new(
            values.iter().copied().collect(),
            Spacing::Linear,
            GridKind::Edges,
        )
        .unwrap()
    }

    #[test]
    fn identity_flat_filter_flat_flux_photon_counting_f64() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 1.0);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let result = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        assert!(approx_eq(result.band_flux, 1.0, TOL));
        assert!(approx_eq(result.coverage, 1.0, TOL));
        assert!(result.band_error.is_none());
    }

    #[test]
    fn identity_flat_filter_flat_flux_energy_weighted_f64() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 1.0);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let result = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::EnergyWeighted,
        )
        .unwrap();
        assert!(approx_eq(result.band_flux, 1.0, TOL));
        assert!(approx_eq(result.coverage, 1.0, TOL));
        assert!(result.band_error.is_none());
    }

    #[test]
    fn identity_flat_filter_flat_flux_photon_counting_f32() {
        let grid = linear_edges_f32(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f32>::from_elem(4, 1.0);
        let transmission = Array1::<f32>::from_elem(4, 1.0);
        let result = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        assert!(approx_eq_f32(result.band_flux, 1.0, TOL_F32));
        assert!(approx_eq_f32(result.coverage, 1.0, TOL_F32));
        assert!(result.band_error.is_none());
    }

    #[test]
    fn photon_counting_vs_energy_weighted_differ_on_sloped_flux() {
        // Bin centers at 0.5, 1.5, 2.5, 3.5; flux = center; T = 1.
        // Energy-weighted: numerator = Σ flux_i * width_i = 0.5+1.5+2.5+3.5 = 8;
        //   denominator = 4; band_flux = 2.0.
        // Photon-counting: numerator = Σ flux_i * lambda_mid_i * width_i
        //   with overlap = full bin (since grids match), lambda_mid_i = center_i:
        //   = 0.25 + 2.25 + 6.25 + 12.25 = 21.0;
        //   denominator = Σ lambda_mid_i * width_i = 0.5+1.5+2.5+3.5 = 8;
        //   band_flux = 21.0 / 8.0 = 2.625.
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = array![0.5_f64, 1.5, 2.5, 3.5];
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let energy = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::EnergyWeighted,
        )
        .unwrap();
        assert!(approx_eq(energy.band_flux, 2.0, TOL));

        let photon = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        assert!(approx_eq(photon.band_flux, 21.0 / 8.0, TOL));
        assert!((photon.band_flux - energy.band_flux).abs() > 0.1);
    }

    #[test]
    fn rectangular_filter_on_flat_flux_energy_weighted() {
        // Spectrum: flat flux F=3.0 over [0, 10] (10 unit-width bins).
        // Filter: rect T=1 over [3, 7] (4 unit-width bins).
        // band_flux should equal F. coverage = full = 1.0 here because the
        // filter range is fully covered by the spectrum.
        let constant_flux = 3.0_f64;
        let spectrum_grid =
            linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let flux = Array1::<f64>::from_elem(10, constant_flux);
        let transmission_grid = linear_edges_f64(&[3.0, 4.0, 5.0, 6.0, 7.0]);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let result = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::EnergyWeighted,
        )
        .unwrap();
        assert!(approx_eq(result.band_flux, constant_flux, TOL));
        assert!(approx_eq(result.coverage, 1.0, TOL));
    }

    #[test]
    fn partial_coverage_photon_counting_lambda_weighted() {
        // Spectrum: [0, 10], 10 unit-width bins, flat flux F=2.0.
        // Filter: rect T=1 over [5, 15], 10 unit-width bins.
        // Overlap is [5, 10].
        // Photon-counting:
        //   D_full = Σ_{j=5..15} 1 * (j-0.5) * 1, lambda_mid_j across [5,15]:
        //   centers 5.5, 6.5, ..., 14.5 -> sum = (5.5+14.5)/2 * 10 = 100.
        //   D = Σ over overlap [5, 10]: centers 5.5, 6.5, 7.5, 8.5, 9.5
        //     -> sum = (5.5+9.5)/2 * 5 = 37.5.
        //   coverage = 37.5 / 100 = 0.375.
        // band_flux = (2.0 * 37.5) / 37.5 = 2.0 (flat flux invariance).
        let flux_value = 2.0_f64;
        let spectrum_grid =
            linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let flux = Array1::<f64>::from_elem(10, flux_value);
        let transmission_grid =
            linear_edges_f64(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let transmission = Array1::<f64>::from_elem(10, 1.0);
        let result = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        assert!(approx_eq(result.band_flux, flux_value, TOL));
        assert!(approx_eq(result.coverage, 0.375, TOL));
    }

    #[test]
    fn no_overlap_returns_error() {
        let spectrum_grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let flux = Array1::<f64>::from_elem(5, 1.0);
        let transmission_grid = linear_edges_f64(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let transmission = Array1::<f64>::from_elem(5, 1.0);
        let err = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(err, PhotometryError::NoOverlap);
    }

    #[test]
    fn flux_length_mismatch_errors() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let bad_flux = array![1.0_f64, 2.0];
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let err = synthetic(
            &grid,
            bad_flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PhotometryError::SpectrumFluxLengthMismatch {
                flux: 2,
                expected: 4
            }
        );
    }

    #[test]
    fn error_length_mismatch_errors() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 1.0);
        let bad_error = array![0.1_f64, 0.2];
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let err = synthetic(
            &grid,
            flux.view(),
            Some(bad_error.view()),
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PhotometryError::SpectrumErrorLengthMismatch {
                error: 2,
                expected: 4
            }
        );
    }

    #[test]
    fn transmission_length_mismatch_errors() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 1.0);
        let bad_transmission = array![1.0_f64, 1.0];
        let err = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            bad_transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PhotometryError::TransmissionLengthMismatch {
                values: 2,
                expected: 4
            }
        );
    }

    #[test]
    fn constant_variance_rectangular_filter_energy_weighted() {
        // 4 unit-width spectrum bins fully inside a rect filter of equal extent.
        // Energy-weighted weights: weight_i = T * width = 1 for each i.
        // denominator = 4. normalized_weight_i = 1/4.
        // band_variance = Σ (1/4)^2 * σ^2 = 4 * σ^2 / 16 = σ^2 / 4.
        // band_error = σ / 2 = σ / sqrt(N) with N=4.
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 0.0);
        let sigma = 0.5_f64;
        let error = Array1::<f64>::from_elem(4, sigma);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let result = synthetic(
            &grid,
            flux.view(),
            Some(error.view()),
            &grid,
            transmission.view(),
            PhotometryConvention::EnergyWeighted,
        )
        .unwrap();
        let expected = sigma / 2.0_f64;
        let band_error = result.band_error.expect("error is present");
        assert!(approx_eq(band_error, expected, TOL));
    }

    #[test]
    fn error_none_survives() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = Array1::<f64>::from_elem(4, 1.0);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let result = synthetic(
            &grid,
            flux.view(),
            None,
            &grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        assert!(result.band_error.is_none());
    }

    #[test]
    fn enum_is_copy_and_eq() {
        let convention = PhotometryConvention::PhotonCounting;
        let copied = convention;
        assert_eq!(convention, copied);
        assert_ne!(convention, PhotometryConvention::EnergyWeighted);
    }

    // ------------------------------------------------------------------
    // SyntheticOperator
    // ------------------------------------------------------------------

    fn operator_test_inputs_f64() -> (Grid<f64>, Array1<f64>, Grid<f64>, Array1<f64>) {
        // Spectrum: 6 unit-width bins over [0, 6]; filter: 4 unit-width bins
        // over [1, 5]. Mostly overlapping but not identical.
        let spectrum_grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flux = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transmission_grid = linear_edges_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let transmission = array![0.4_f64, 0.7, 0.9, 0.5];
        (spectrum_grid, flux, transmission_grid, transmission)
    }

    fn operator_test_inputs_f32() -> (Grid<f32>, Array1<f32>, Grid<f32>, Array1<f32>) {
        let spectrum_grid = linear_edges_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flux = array![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transmission_grid = linear_edges_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let transmission = array![0.4_f32, 0.7, 0.9, 0.5];
        (spectrum_grid, flux, transmission_grid, transmission)
    }

    #[test]
    fn operator_apply_matches_synthetic_no_error_photon_counting_f64() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let convention = PhotometryConvention::PhotonCounting;
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let direct = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let applied = operator.apply(flux.view(), None);
        assert!(approx_eq(applied.band_flux, direct.band_flux, TOL));
        assert!(applied.band_error.is_none());
    }

    #[test]
    fn operator_apply_matches_synthetic_no_error_energy_weighted_f64() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let convention = PhotometryConvention::EnergyWeighted;
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let direct = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let applied = operator.apply(flux.view(), None);
        assert!(approx_eq(applied.band_flux, direct.band_flux, TOL));
    }

    #[test]
    fn operator_apply_matches_synthetic_with_error_f64() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let error = array![0.1_f64, 0.15, 0.2, 0.25, 0.3, 0.35];
        let convention = PhotometryConvention::PhotonCounting;
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let direct = synthetic(
            &spectrum_grid,
            flux.view(),
            Some(error.view()),
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let applied = operator.apply(flux.view(), Some(error.view()));
        assert!(approx_eq(applied.band_flux, direct.band_flux, TOL));
        let applied_error = applied.band_error.expect("error present");
        let direct_error = direct.band_error.expect("error present");
        assert!(approx_eq(applied_error, direct_error, TOL));
    }

    #[test]
    fn operator_apply_matches_synthetic_with_error_f32() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f32();
        let error = array![0.1_f32, 0.15, 0.2, 0.25, 0.3, 0.35];
        let convention = PhotometryConvention::EnergyWeighted;
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let direct = synthetic(
            &spectrum_grid,
            flux.view(),
            Some(error.view()),
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let applied = operator.apply(flux.view(), Some(error.view()));
        assert!(approx_eq_f32(applied.band_flux, direct.band_flux, TOL_F32));
        let applied_error = applied.band_error.expect("error present");
        let direct_error = direct.band_error.expect("error present");
        assert!(approx_eq_f32(applied_error, direct_error, TOL_F32));
    }

    #[test]
    fn operator_coverage_matches_synthetic() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let convention = PhotometryConvention::PhotonCounting;
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        let direct = synthetic(
            &spectrum_grid,
            flux.view(),
            None,
            &transmission_grid,
            transmission.view(),
            convention,
        )
        .unwrap();
        assert!(approx_eq(operator.coverage(), direct.coverage, TOL));
    }

    #[test]
    fn operator_new_rejects_transmission_length_mismatch() {
        let grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let bad_transmission = array![1.0_f64, 1.0];
        let err = SyntheticOperator::new(
            &grid,
            &grid,
            bad_transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(
            err,
            PhotometryError::TransmissionLengthMismatch {
                values: 2,
                expected: 4
            }
        );
    }

    #[test]
    fn operator_new_rejects_no_overlap() {
        let spectrum_grid = linear_edges_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let transmission_grid = linear_edges_f64(&[10.0, 11.0, 12.0, 13.0, 14.0]);
        let transmission = Array1::<f64>::from_elem(4, 1.0);
        let err = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap_err();
        assert_eq!(err, PhotometryError::NoOverlap);
    }

    #[test]
    #[should_panic(expected = "spectrum_flux length")]
    fn operator_apply_panics_on_flux_length_mismatch() {
        let (spectrum_grid, _flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        let bad_flux = array![1.0_f64, 2.0];
        let _ = operator.apply(bad_flux.view(), None);
    }

    #[test]
    #[should_panic(expected = "spectrum_error length")]
    fn operator_apply_panics_on_error_length_mismatch() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        let bad_error = array![0.1_f64, 0.2];
        let _ = operator.apply(flux.view(), Some(bad_error.view()));
    }

    #[test]
    fn operator_repeated_apply_is_pure() {
        let (spectrum_grid, flux, transmission_grid, transmission) = operator_test_inputs_f64();
        let operator = SyntheticOperator::new(
            &spectrum_grid,
            &transmission_grid,
            transmission.view(),
            PhotometryConvention::PhotonCounting,
        )
        .unwrap();
        let first = operator.apply(flux.view(), None);
        let second = operator.apply(flux.view(), None);
        let third = operator.apply(flux.view(), None);
        assert!(approx_eq(first.band_flux, second.band_flux, TOL));
        assert!(approx_eq(second.band_flux, third.band_flux, TOL));
    }
}

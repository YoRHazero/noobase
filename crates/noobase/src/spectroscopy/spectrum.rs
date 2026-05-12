use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::bins::{Grid, GridKind, overlap};
use crate::float::Float;

#[derive(Debug, Error, PartialEq)]
pub enum SpectrumError {
    #[error("flux length {flux} does not match wavelength bin count {expected}")]
    FluxLengthMismatch { flux: usize, expected: usize },
    #[error("error length {error} does not match wavelength bin count {expected}")]
    ErrorLengthMismatch { error: usize, expected: usize },
    #[error("mask length {mask} does not match wavelength bin count {expected}")]
    MaskLengthMismatch { mask: usize, expected: usize },
}

/// A 1-D spectrum: a wavelength grid plus per-bin flux, optional 1-sigma error,
/// and optional validity mask. The mask convention is `true = valid`, which is
/// the inverse of astropy's masked-array convention.
#[derive(Debug, Clone)]
pub struct Spectrum<T: Float> {
    wavelength: Grid<T>,
    flux: Array1<T>,
    error: Option<Array1<T>>,
    mask: Option<Array1<bool>>,
}

impl<T: Float> Spectrum<T> {
    /// Construct a spectrum, validating that every per-bin array has the same
    /// length as the wavelength grid's bin count (`len()` for `Centers`,
    /// `len() - 1` for `Edges`).
    pub fn new(
        wavelength: Grid<T>,
        flux: Array1<T>,
        error: Option<Array1<T>>,
        mask: Option<Array1<bool>>,
    ) -> Result<Self, SpectrumError> {
        let expected = bin_count(&wavelength);
        if flux.len() != expected {
            return Err(SpectrumError::FluxLengthMismatch {
                flux: flux.len(),
                expected,
            });
        }
        if let Some(ref err) = error
            && err.len() != expected
        {
            return Err(SpectrumError::ErrorLengthMismatch {
                error: err.len(),
                expected,
            });
        }
        if let Some(ref m) = mask
            && m.len() != expected
        {
            return Err(SpectrumError::MaskLengthMismatch {
                mask: m.len(),
                expected,
            });
        }
        Ok(Self {
            wavelength,
            flux,
            error,
            mask,
        })
    }

    pub fn wavelength(&self) -> &Grid<T> {
        &self.wavelength
    }

    pub fn flux(&self) -> ArrayView1<'_, T> {
        self.flux.view()
    }

    pub fn error(&self) -> Option<ArrayView1<'_, T>> {
        self.error.as_ref().map(|array| array.view())
    }

    pub fn mask(&self) -> Option<ArrayView1<'_, bool>> {
        self.mask.as_ref().map(|array| array.view())
    }

    pub fn n_bins(&self) -> usize {
        self.flux.len()
    }

    /// Resamples the spectrum onto `target` using the bins::overlap primitives.
    ///
    /// The error is propagated by squaring to variance, applying
    /// `bins::overlap::rebin_variance` (which assumes source bins are
    /// statistically independent), and taking the square root. The output
    /// represents the marginal 1-sigma uncertainty per target bin; it does not
    /// capture covariance between target bins.
    ///
    /// When the target is finer than the source (upsampling), neighboring
    /// target bins that draw from the same source bin are strongly correlated.
    /// The per-bin sigma values returned here are still individually correct as
    /// marginals, but downstream operations that assume independent bins (e.g.
    /// summing under quadrature) will underestimate the true uncertainty. A
    /// future `Spectrum` may carry a full covariance representation; for now
    /// this is the caller's responsibility.
    ///
    /// The mask convention is `true = valid`. A target bin is marked valid iff
    /// every source bin with non-zero overlap into it is valid; otherwise the
    /// target bin is marked invalid.
    ///
    /// The output wavelength reuses `target.kind()` unchanged.
    pub fn rebin(&self, target: &Grid<T>) -> Spectrum<T> {
        let output_flux = overlap::rebin(&self.wavelength, self.flux.view(), target);
        let target_bin_count = output_flux.len();
        let output_error = self.error.as_ref().map(|error_array| {
            let variance: Array1<T> = error_array.mapv(|value| value * value);
            let output_variance =
                overlap::rebin_variance(&self.wavelength, variance.view(), target);
            output_variance.mapv(|value| value.sqrt())
        });
        let output_mask = self.mask.as_ref().map(|source_mask| {
            let mut mask = Array1::<bool>::from_elem(target_bin_count, true);
            overlap::for_each(
                &self.wavelength,
                target,
                |target_index, source_index, _overlap_width| {
                    if !source_mask[source_index] {
                        mask[target_index] = false;
                    }
                },
            );
            mask
        });
        Spectrum {
            wavelength: target.clone(),
            flux: output_flux,
            error: output_error,
            mask: output_mask,
        }
    }
}

fn bin_count<T: Float>(wavelength: &Grid<T>) -> usize {
    match wavelength.kind() {
        GridKind::Centers => wavelength.len(),
        GridKind::Edges => wavelength.len() - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{GridKind, Spacing};
    use ndarray::array;

    const TOL: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn centers_grid_f64(values: &[f64]) -> Grid<f64> {
        Grid::new(
            values.iter().copied().collect(),
            Spacing::Linear,
            GridKind::Centers,
        )
        .unwrap()
    }

    fn edges_grid_f64(values: &[f64]) -> Grid<f64> {
        Grid::new(
            values.iter().copied().collect(),
            Spacing::Linear,
            GridKind::Edges,
        )
        .unwrap()
    }

    #[test]
    fn new_centers_flux_only() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0, 4.0]);
        let flux = array![10.0_f64, 20.0, 30.0, 40.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        assert_eq!(spectrum.n_bins(), 4);
        assert_eq!(spectrum.wavelength().kind(), GridKind::Centers);
        assert!(spectrum.error().is_none());
        assert!(spectrum.mask().is_none());
    }

    #[test]
    fn new_edges_bin_count_is_len_minus_one() {
        let wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = array![1.0_f64, 2.0, 3.0, 4.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        assert_eq!(spectrum.n_bins(), 4);
        assert_eq!(spectrum.wavelength().len(), 5);
        assert_eq!(spectrum.wavelength().kind(), GridKind::Edges);
    }

    #[test]
    fn new_with_flux_error_mask_all_some() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![10.0_f64, 20.0, 30.0];
        let error = array![0.1_f64, 0.2, 0.3];
        let mask = array![true, false, true];
        let spectrum = Spectrum::new(wavelength, flux, Some(error), Some(mask)).unwrap();
        assert_eq!(spectrum.n_bins(), 3);
        let flux_view = spectrum.flux();
        assert!(approx_eq(flux_view[0], 10.0, TOL));
        assert!(approx_eq(flux_view[2], 30.0, TOL));
        let error_view = spectrum.error().unwrap();
        assert_eq!(error_view.len(), 3);
        assert!(approx_eq(error_view[1], 0.2, TOL));
        let mask_view = spectrum.mask().unwrap();
        assert_eq!(mask_view.len(), 3);
        assert!(mask_view[0]);
        assert!(!mask_view[1]);
        assert!(mask_view[2]);
    }

    #[test]
    fn new_rejects_flux_length_mismatch_centers() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![10.0_f64, 20.0];
        let err = Spectrum::new(wavelength, flux, None, None).unwrap_err();
        assert_eq!(
            err,
            SpectrumError::FluxLengthMismatch {
                flux: 2,
                expected: 3,
            }
        );
    }

    #[test]
    fn new_rejects_flux_length_mismatch_edges() {
        let wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0]);
        // Edges -> 3 bins expected; pass 4-element flux to trigger mismatch.
        let flux = array![1.0_f64, 2.0, 3.0, 4.0];
        let err = Spectrum::new(wavelength, flux, None, None).unwrap_err();
        assert_eq!(
            err,
            SpectrumError::FluxLengthMismatch {
                flux: 4,
                expected: 3,
            }
        );
    }

    #[test]
    fn new_rejects_error_length_mismatch() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![10.0_f64, 20.0, 30.0];
        let error = array![0.1_f64, 0.2];
        let err = Spectrum::new(wavelength, flux, Some(error), None).unwrap_err();
        assert_eq!(
            err,
            SpectrumError::ErrorLengthMismatch {
                error: 2,
                expected: 3,
            }
        );
    }

    #[test]
    fn new_rejects_mask_length_mismatch() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![10.0_f64, 20.0, 30.0];
        let mask = array![true, false];
        let err = Spectrum::new(wavelength, flux, None, Some(mask)).unwrap_err();
        assert_eq!(
            err,
            SpectrumError::MaskLengthMismatch {
                mask: 2,
                expected: 3,
            }
        );
    }

    #[test]
    fn accessors_return_expected_lengths_and_values() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0, 4.0]);
        let flux = array![10.0_f64, 20.0, 30.0, 40.0];
        let error = array![1.0_f64, 1.0, 1.0, 1.0];
        let mask = array![true, true, false, true];
        let spectrum = Spectrum::new(wavelength, flux, Some(error), Some(mask)).unwrap();
        assert_eq!(spectrum.n_bins(), 4);
        assert_eq!(spectrum.flux().len(), 4);
        assert_eq!(spectrum.error().unwrap().len(), 4);
        assert_eq!(spectrum.mask().unwrap().len(), 4);
        assert_eq!(spectrum.wavelength().len(), 4);
        let flux_view = spectrum.flux();
        assert!(approx_eq(flux_view[3], 40.0, TOL));
    }

    #[test]
    fn works_with_f64_smoke() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![1.0_f64, 2.0, 3.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        assert_eq!(spectrum.n_bins(), 3);
    }

    #[test]
    fn works_with_f32_smoke() {
        let wavelength = Grid::<f32>::linspace(1.0, 3.0, 3, GridKind::Centers);
        let flux = array![1.0_f32, 2.0, 3.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        assert_eq!(spectrum.n_bins(), 3);
    }

    #[test]
    fn rebin_identity_preserves_flux_error_mask() {
        let wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux = array![1.0_f64, 2.0, 3.0, 4.0];
        let error = array![0.5_f64, 0.4, 0.3, 0.2];
        let mask = array![true, true, false, true];
        let spectrum = Spectrum::new(
            wavelength.clone(),
            flux.clone(),
            Some(error.clone()),
            Some(mask.clone()),
        )
        .unwrap();
        let output = spectrum.rebin(&wavelength);
        assert_eq!(output.n_bins(), 4);
        let flux_view = output.flux();
        for index in 0..4 {
            assert!(approx_eq(flux_view[index], flux[index], TOL));
        }
        let error_view = output.error().unwrap();
        for index in 0..4 {
            assert!(approx_eq(error_view[index], error[index], TOL));
        }
        let mask_view = output.mask().unwrap();
        for index in 0..4 {
            assert_eq!(mask_view[index], mask[index]);
        }
    }

    #[test]
    fn rebin_downsample_two_to_one_flux_only() {
        // Source: 4 width-1 bins -> target: 2 width-2 bins.
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 3.0, 5.0, 9.0];
        let spectrum = Spectrum::new(source_wavelength, flux, None, None).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert_eq!(output.n_bins(), 2);
        let flux_view = output.flux();
        assert!(approx_eq(flux_view[0], (1.0 + 3.0) / 2.0, TOL));
        assert!(approx_eq(flux_view[1], (5.0 + 9.0) / 2.0, TOL));
        assert!(output.error().is_none());
        assert!(output.mask().is_none());
    }

    #[test]
    fn rebin_downsample_constant_error_yields_sqrt_v_over_2() {
        // Source variance v per width-1 bin; target width-2 bin variance = v/2;
        // target sigma = sqrt(v/2).
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let variance = 0.8_f64;
        let sigma = variance.sqrt();
        let flux = Array1::<f64>::from_elem(4, 0.0);
        let error = Array1::<f64>::from_elem(4, sigma);
        let spectrum = Spectrum::new(source_wavelength, flux, Some(error), None).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        let error_view = output.error().unwrap();
        assert_eq!(error_view.len(), 2);
        let expected = (variance / 2.0).sqrt();
        assert!(approx_eq(error_view[0], expected, TOL));
        assert!(approx_eq(error_view[1], expected, TOL));
    }

    #[test]
    fn rebin_mask_propagation_two_to_one() {
        // Source mask [true, false, true, true] over width-1 bins;
        // target width-2 bins -> [false, true].
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 1.0, 1.0, 1.0];
        let mask = array![true, false, true, true];
        let spectrum = Spectrum::new(source_wavelength, flux, None, Some(mask)).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        let mask_view = output.mask().unwrap();
        assert_eq!(mask_view.len(), 2);
        assert!(!mask_view[0]);
        assert!(mask_view[1]);
    }

    #[test]
    fn rebin_combined_flux_error_mask_propagate_together() {
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 3.0, 5.0, 9.0];
        let variance_value = 0.5_f64;
        let sigma = variance_value.sqrt();
        let error = Array1::<f64>::from_elem(4, sigma);
        let mask = array![true, true, true, false];
        let spectrum = Spectrum::new(source_wavelength, flux, Some(error), Some(mask)).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert_eq!(output.n_bins(), 2);
        let flux_view = output.flux();
        assert!(approx_eq(flux_view[0], 2.0, TOL));
        assert!(approx_eq(flux_view[1], 7.0, TOL));
        let error_view = output.error().unwrap();
        let expected_sigma = (variance_value / 2.0).sqrt();
        assert!(approx_eq(error_view[0], expected_sigma, TOL));
        assert!(approx_eq(error_view[1], expected_sigma, TOL));
        let mask_view = output.mask().unwrap();
        assert!(mask_view[0]);
        assert!(!mask_view[1]);
    }

    #[test]
    fn rebin_upsample_single_source_bin_marginal_values() {
        // One width-4 source bin with constant flux f and sigma s.
        // Target: 4 width-1 bins fully inside the source.
        // Marginal output flux per target bin = f. Variance per target bin:
        //   (1^2 * s^2) / 1^2 = s^2 -> sigma = s.
        let source_wavelength = edges_grid_f64(&[0.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux_value = 7.0_f64;
        let sigma = 0.6_f64;
        let flux = array![flux_value];
        let error = array![sigma];
        let spectrum = Spectrum::new(source_wavelength, flux, Some(error), None).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert_eq!(output.n_bins(), 4);
        let flux_view = output.flux();
        let error_view = output.error().unwrap();
        for index in 0..4 {
            assert!(approx_eq(flux_view[index], flux_value, TOL));
            assert!(approx_eq(error_view[index], sigma, TOL));
        }
    }

    #[test]
    fn rebin_mask_present_error_absent() {
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 1.0, 1.0, 1.0];
        let mask = array![false, true, true, true];
        let spectrum = Spectrum::new(source_wavelength, flux, None, Some(mask)).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert!(output.error().is_none());
        let mask_view = output.mask().unwrap();
        assert!(!mask_view[0]);
        assert!(mask_view[1]);
    }

    #[test]
    fn rebin_error_present_mask_absent() {
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![0.0_f64, 0.0, 0.0, 0.0];
        let variance_value = 0.25_f64;
        let sigma = variance_value.sqrt();
        let error = Array1::<f64>::from_elem(4, sigma);
        let spectrum = Spectrum::new(source_wavelength, flux, Some(error), None).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert!(output.mask().is_none());
        let error_view = output.error().unwrap();
        let expected_sigma = (variance_value / 2.0).sqrt();
        assert!(approx_eq(error_view[0], expected_sigma, TOL));
        assert!(approx_eq(error_view[1], expected_sigma, TOL));
    }

    #[test]
    fn rebin_output_wavelength_kind_matches_target() {
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 2.0, 3.0, 4.0];
        let spectrum = Spectrum::new(source_wavelength, flux, None, None).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert_eq!(output.wavelength().kind(), GridKind::Edges);
        assert_eq!(output.wavelength().len(), 3);
    }
}

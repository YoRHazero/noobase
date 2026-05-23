use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::axis::{Grid, GridKind, overlap};
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
/// and optional mask. The mask convention is `true = invalid` (a `true` entry
/// marks the bin as masked / excluded), matching astropy's masked-array
/// convention.
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
    /// The mask convention is `true = invalid`. A target bin is marked invalid
    /// iff any source bin with non-zero overlap into it is invalid; otherwise
    /// the target bin is marked valid.
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
            // `true = invalid`. Fold is a logical OR over "invalid": a target
            // bin is invalid iff any source bin overlapping it is invalid. The
            // identity element is therefore `false` (nothing masked).
            let mut mask = Array1::<bool>::from_elem(target_bin_count, false);
            overlap::for_each(
                &self.wavelength,
                target,
                |target_index, source_index, _overlap_width| {
                    if source_mask[source_index] {
                        mask[target_index] = true;
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

    /// Convert flux density from f_lambda to f_nu via `f_nu = f_lambda * lambda^2 / c`.
    ///
    /// `speed_of_light` must match the wavelength axis units. For example, if the
    /// wavelength axis is in angstroms, pass `c` in angstroms per second
    /// (approximately `2.998e18`). noobase does not track units; consistency is the
    /// caller's responsibility.
    ///
    /// The wavelength grid is preserved unchanged. The error array, if present,
    /// scales element-wise by the same factor (`sigma_new = sigma_old * lambda^2 / c`).
    /// The mask is preserved.
    ///
    /// This method does not check whether the input is in fact f_lambda; it
    /// unconditionally applies the conversion factor. The caller is responsible
    /// for tracking which density the spectrum currently represents.
    pub fn to_f_nu(&self, speed_of_light: T) -> Spectrum<T> {
        self.convert_with_factor(|wavelength| (wavelength * wavelength) / speed_of_light)
    }

    /// Convert flux density from f_nu to f_lambda via `f_lambda = f_nu * c / lambda^2`.
    ///
    /// `speed_of_light` must match the wavelength axis units. For example, if the
    /// wavelength axis is in angstroms, pass `c` in angstroms per second
    /// (approximately `2.998e18`). noobase does not track units; consistency is the
    /// caller's responsibility.
    ///
    /// The wavelength grid is preserved unchanged. The error array, if present,
    /// scales element-wise by the same factor (`sigma_new = sigma_old * c / lambda^2`).
    /// The mask is preserved.
    ///
    /// This method does not check whether the input is in fact f_nu; it
    /// unconditionally applies the conversion factor. The caller is responsible
    /// for tracking which density the spectrum currently represents.
    pub fn to_f_lambda(&self, speed_of_light: T) -> Spectrum<T> {
        self.convert_with_factor(|wavelength| speed_of_light / (wavelength * wavelength))
    }

    /// Apply a per-bin scalar factor (computed from each bin's center wavelength)
    /// to flux and error, leaving wavelength and mask untouched. Used to share
    /// the conversion plumbing between `to_f_nu` and `to_f_lambda`.
    fn convert_with_factor<F>(&self, factor_for: F) -> Spectrum<T>
    where
        F: Fn(T) -> T,
    {
        let centers_grid = match self.wavelength.kind() {
            GridKind::Centers => None,
            GridKind::Edges => Some(self.wavelength.to_centers()),
        };
        let wavelength_centers = match &centers_grid {
            Some(grid) => grid.values(),
            None => self.wavelength.values(),
        };
        let bin_count = self.flux.len();
        let mut new_flux = Array1::<T>::zeros(bin_count);
        for index in 0..bin_count {
            let factor = factor_for(wavelength_centers[index]);
            new_flux[index] = self.flux[index] * factor;
        }
        let new_error = self.error.as_ref().map(|error_array| {
            let mut output = Array1::<T>::zeros(bin_count);
            for index in 0..bin_count {
                let factor = factor_for(wavelength_centers[index]);
                output[index] = error_array[index] * factor;
            }
            output
        });
        // Direct struct construction: lengths are guaranteed to match self,
        // so going through `Spectrum::new` would only repeat validation.
        Spectrum {
            wavelength: self.wavelength.clone(),
            flux: new_flux,
            error: new_error,
            mask: self.mask.clone(),
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
    use crate::axis::{GridKind, Spacing};
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
        // `true = invalid`. Source mask [true, false, false, false] over
        // width-1 bins; target width-2 bins. Target bin 0 (sources 0,1) has
        // an invalid source -> invalid (true). Target bin 1 (sources 2,3) has
        // no invalid source -> valid (false). Expect [true, false].
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 1.0, 1.0, 1.0];
        let mask = array![true, false, false, false];
        let spectrum = Spectrum::new(source_wavelength, flux, None, Some(mask)).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        let mask_view = output.mask().unwrap();
        assert_eq!(mask_view.len(), 2);
        assert!(mask_view[0]);
        assert!(!mask_view[1]);
    }

    #[test]
    fn rebin_combined_flux_error_mask_propagate_together() {
        let source_wavelength = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let target_wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 3.0, 5.0, 9.0];
        let variance_value = 0.5_f64;
        let sigma = variance_value.sqrt();
        let error = Array1::<f64>::from_elem(4, sigma);
        // `true = invalid`. Target bin 0 (sources 0,1) has no invalid source
        // -> valid (false). Target bin 1 (sources 2,3) has an invalid source
        // -> invalid (true).
        let mask = array![false, false, false, true];
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
        assert!(!mask_view[0]);
        assert!(mask_view[1]);
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
        // `true = invalid`. Target bin 0 (sources 0,1) has an invalid source
        // -> invalid (true). Target bin 1 (sources 2,3) has none -> valid.
        let mask = array![true, false, false, false];
        let spectrum = Spectrum::new(source_wavelength, flux, None, Some(mask)).unwrap();
        let output = spectrum.rebin(&target_wavelength);
        assert!(output.error().is_none());
        let mask_view = output.mask().unwrap();
        assert!(mask_view[0]);
        assert!(!mask_view[1]);
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

    fn approx_eq_array_f64(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for index in 0..actual.len() {
            assert!(
                approx_eq(actual[index], expected[index], tol),
                "index {index}: actual={}, expected={}",
                actual[index],
                expected[index]
            );
        }
    }

    #[test]
    fn flux_conversion_round_trip_identity_f64() {
        let wavelength = centers_grid_f64(&[1000.0, 1500.0, 2000.0, 2500.0]);
        let flux = array![3.5_f64, 4.1, 5.7, 2.3];
        let spectrum = Spectrum::new(wavelength, flux.clone(), None, None).unwrap();
        let speed_of_light = 2.998e18_f64;
        let recovered = spectrum.to_f_nu(speed_of_light).to_f_lambda(speed_of_light);
        let recovered_flux = recovered.flux();
        approx_eq_array_f64(
            recovered_flux.as_slice().unwrap(),
            flux.as_slice().unwrap(),
            TOL,
        );
    }

    #[test]
    fn flux_conversion_round_trip_identity_f32() {
        let wavelength = Grid::<f32>::linspace(1000.0, 2500.0, 4, GridKind::Centers);
        let flux = array![3.5_f32, 4.1, 5.7, 2.3];
        let spectrum = Spectrum::new(wavelength, flux.clone(), None, None).unwrap();
        let speed_of_light = 2.998e18_f32;
        let recovered = spectrum.to_f_nu(speed_of_light).to_f_lambda(speed_of_light);
        let recovered_flux = recovered.flux();
        let tol_f32 = 1e-4_f32;
        for index in 0..flux.len() {
            let actual = recovered_flux[index];
            let expected = flux[index];
            let diff = (actual - expected).abs();
            let bound = tol_f32 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                diff <= bound,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn flux_conversion_analytical_to_f_nu_centers() {
        // Two centers, flux=1 each, c=1: output flux = lambda^2.
        let wavelength = centers_grid_f64(&[2.0, 5.0]);
        let flux = array![1.0_f64, 1.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let output = spectrum.to_f_nu(1.0);
        let output_flux = output.flux();
        assert!(approx_eq(output_flux[0], 4.0, TOL));
        assert!(approx_eq(output_flux[1], 25.0, TOL));
    }

    #[test]
    fn flux_conversion_error_scaling_to_f_nu() {
        let wavelength = centers_grid_f64(&[2.0, 5.0]);
        let flux = array![1.0_f64, 1.0];
        let error = array![0.1_f64, 0.2];
        let spectrum = Spectrum::new(wavelength, flux, Some(error.clone()), None).unwrap();
        let speed_of_light = 1.0_f64;
        let output = spectrum.to_f_nu(speed_of_light);
        let output_error = output.error().unwrap();
        // factor[0] = 4 / 1 = 4; factor[1] = 25 / 1 = 25.
        assert!(approx_eq(output_error[0], 0.1 * 4.0, TOL));
        assert!(approx_eq(output_error[1], 0.2 * 25.0, TOL));
    }

    #[test]
    fn flux_conversion_error_scaling_to_f_lambda() {
        let wavelength = centers_grid_f64(&[2.0, 5.0]);
        let flux = array![1.0_f64, 1.0];
        let error = array![0.4_f64, 0.5];
        let spectrum = Spectrum::new(wavelength, flux, Some(error), None).unwrap();
        let speed_of_light = 1.0_f64;
        let output = spectrum.to_f_lambda(speed_of_light);
        let output_error = output.error().unwrap();
        // factor[0] = 1 / 4 = 0.25; factor[1] = 1 / 25 = 0.04.
        assert!(approx_eq(output_error[0], 0.4 * 0.25, TOL));
        assert!(approx_eq(output_error[1], 0.5 * 0.04, TOL));
    }

    #[test]
    fn flux_conversion_mask_preservation() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0, 4.0]);
        let flux = array![1.0_f64, 1.0, 1.0, 1.0];
        let mask = array![true, false, true, false];
        let spectrum = Spectrum::new(wavelength, flux, None, Some(mask.clone())).unwrap();
        let output = spectrum.to_f_nu(1.0);
        let output_mask = output.mask().unwrap();
        assert_eq!(output_mask.len(), mask.len());
        for index in 0..mask.len() {
            assert_eq!(output_mask[index], mask[index]);
        }
        let output_back = spectrum.to_f_lambda(1.0);
        let output_back_mask = output_back.mask().unwrap();
        for index in 0..mask.len() {
            assert_eq!(output_back_mask[index], mask[index]);
        }
    }

    #[test]
    fn flux_conversion_error_none_survives() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![1.0_f64, 2.0, 3.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let output = spectrum.to_f_nu(1.0);
        assert!(output.error().is_none());
        let output_back = spectrum.to_f_lambda(1.0);
        assert!(output_back.error().is_none());
    }

    #[test]
    fn flux_conversion_mask_none_survives() {
        let wavelength = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux = array![1.0_f64, 2.0, 3.0];
        let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
        let output = spectrum.to_f_nu(1.0);
        assert!(output.mask().is_none());
        let output_back = spectrum.to_f_lambda(1.0);
        assert!(output_back.mask().is_none());
    }

    #[test]
    fn flux_conversion_centers_path_preserves_wavelength() {
        let wavelength = centers_grid_f64(&[1.5, 2.5, 3.5]);
        let flux = array![1.0_f64, 2.0, 3.0];
        let spectrum = Spectrum::new(wavelength.clone(), flux, None, None).unwrap();
        let output = spectrum.to_f_nu(1.0);
        assert_eq!(output.wavelength().kind(), wavelength.kind());
        assert_eq!(output.wavelength().spacing(), wavelength.spacing());
        let input_values = wavelength.values();
        let output_values = output.wavelength().values();
        assert_eq!(output_values.len(), input_values.len());
        for index in 0..input_values.len() {
            assert_eq!(output_values[index], input_values[index]);
        }
    }

    #[test]
    fn flux_conversion_edges_path_uses_centers_and_preserves_wavelength() {
        // Edges [0, 2, 4] -> centers [1, 3]. flux=[1, 1], c=1.
        // factor = [1^2, 3^2] = [1, 9].
        let wavelength = edges_grid_f64(&[0.0, 2.0, 4.0]);
        let flux = array![1.0_f64, 1.0];
        let spectrum = Spectrum::new(wavelength.clone(), flux, None, None).unwrap();
        let output = spectrum.to_f_nu(1.0);
        let output_flux = output.flux();
        assert!(approx_eq(output_flux[0], 1.0, TOL));
        assert!(approx_eq(output_flux[1], 9.0, TOL));
        // Wavelength grid is preserved as Edges (not collapsed to centers).
        assert_eq!(output.wavelength().kind(), GridKind::Edges);
        let input_values = wavelength.values();
        let output_values = output.wavelength().values();
        assert_eq!(output_values.len(), input_values.len());
        for index in 0..input_values.len() {
            assert_eq!(output_values[index], input_values[index]);
        }
    }

    #[test]
    fn flux_conversion_n_bins_preserved_centers_and_edges() {
        let centers = centers_grid_f64(&[1.0, 2.0, 3.0]);
        let flux_centers = array![1.0_f64, 2.0, 3.0];
        let spectrum_centers = Spectrum::new(centers, flux_centers, None, None).unwrap();
        assert_eq!(
            spectrum_centers.to_f_nu(1.0).n_bins(),
            spectrum_centers.n_bins()
        );
        assert_eq!(
            spectrum_centers.to_f_lambda(1.0).n_bins(),
            spectrum_centers.n_bins()
        );

        let edges = edges_grid_f64(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let flux_edges = array![1.0_f64, 2.0, 3.0, 4.0];
        let spectrum_edges = Spectrum::new(edges, flux_edges, None, None).unwrap();
        assert_eq!(
            spectrum_edges.to_f_nu(1.0).n_bins(),
            spectrum_edges.n_bins()
        );
        assert_eq!(
            spectrum_edges.to_f_lambda(1.0).n_bins(),
            spectrum_edges.n_bins()
        );
    }
}

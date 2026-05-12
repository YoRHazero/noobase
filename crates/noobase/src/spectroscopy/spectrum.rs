use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::bins::{Grid, GridKind};
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
}

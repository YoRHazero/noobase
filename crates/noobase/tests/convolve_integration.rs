//! Cross-module integration tests for the convolve surface.
//!
//! These exercise the *public* API only (`noobase::convolve::*`,
//! `noobase::image::convolve_*`, `noobase::spectroscopy::Spectrum::
//! convolve_lsf`) to pin that the layers compose across module
//! boundaries as documented: the bare kernels feed the NaN wrappers,
//! which feed the image domain entries; and the LSF entry composes the
//! `bins` Grid with the `convolve` Gaussian. Per-module behavior is
//! already covered by the in-crate `mod tests`; this file is the
//! outside-the-crate contract.

use ndarray::{Array1, Array2, Axis};
use noobase::convolve::{Boundary, GaussianSampling, Normalization, conv2d, gaussian1d};
use noobase::image::{convolve_gaussian_axis, convolve_psf};
use noobase::spectroscopy::{LsfSpec, Spectrum};
use noobase::{Grid, GridKind};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn convolve_lsf_conserves_constant_via_public_api_f64() {
    // Spectrum (spectroscopy) + Grid (bins) + Gaussian (convolve) all
    // composed through the public surface; constant template stays
    // constant including the edges.
    let wavelength = Grid::<f64>::logspace(4000.0, 7000.0, 200, GridKind::Centers);
    let flux = Array1::<f64>::from_elem(200, 2.5);
    let spectrum = Spectrum::new(wavelength, flux, None, None).unwrap();
    let broadened = spectrum.convolve_lsf(LsfSpec::ConstantR(2500.0)).unwrap();
    for value in broadened.flux().iter() {
        assert!(approx_eq(*value, 2.5, 1e-9));
    }
    assert!(broadened.error().is_none());
    assert!(broadened.mask().is_none());
}

#[test]
fn convolve_lsf_rejects_non_template_via_public_api() {
    let wavelength = Grid::<f64>::logspace(4000.0, 7000.0, 32, GridKind::Centers);
    let flux = Array1::<f64>::from_elem(32, 1.0);
    let error = Array1::<f64>::from_elem(32, 0.1);
    let observed = Spectrum::new(wavelength, flux, Some(error), None).unwrap();
    assert!(observed.convolve_lsf(LsfSpec::ConstantR(2000.0)).is_err());
}

#[test]
fn convolve_psf_interior_composes_bare_conv2d_f64() {
    // image::convolve_psf flips the PSF and renormalizes. With a
    // Sum-normalized PSF and a NaN-free image, the interior weight is 1,
    // so the interior must equal the bare conv2d of the flipped PSF —
    // pinning the documented cross-module composition.
    let mut image = Array2::<f64>::zeros((8, 9));
    for ((i, j), v) in image.indexed_iter_mut() {
        *v = (i * 4 + j * 3) as f64 * 0.1 - 1.5;
    }
    let mut psf = ndarray::array![
        [0.05_f64, 0.10, 0.02],
        [0.15, 0.30, 0.08],
        [0.04, 0.18, 0.08]
    ];
    let sum: f64 = psf.iter().sum();
    psf.mapv_inplace(|v| v / sum);

    let via_entry = convolve_psf(image.view(), psf.view());
    let flipped = psf.slice(ndarray::s![..;-1, ..;-1]).to_owned();
    let via_bare = conv2d(image.view(), flipped.view(), Boundary::Zero);
    for i in 1..7 {
        for j in 1..8 {
            assert!(
                approx_eq(via_entry[(i, j)], via_bare[(i, j)], 1e-9),
                "interior mismatch at ({i},{j})"
            );
        }
    }
}

#[test]
fn convolve_gaussian_axis_matched_filter_peaks_at_injected_sigma_f64() {
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
    let peak = |probe: f64| {
        convolve_gaussian_axis(
            image.view(),
            probe,
            Axis(0),
            Normalization::L2,
            Boundary::Zero,
            false,
        )[(60, 0)]
    };
    let matched = peak(injected_sigma);
    assert!(matched > peak(injected_sigma * 0.5));
    assert!(matched > peak(injected_sigma * 2.0));
}

#[test]
fn gaussian1d_public_normalizations() {
    // Confirms the convolve re-exports are reachable at the crate path
    // and the two kernel normalizations behave.
    let sum_kernel = gaussian1d::<f64>(
        1.7,
        4.0,
        GaussianSampling::ErfIntegrated,
        Normalization::Sum,
    );
    let l2_kernel = gaussian1d::<f64>(1.7, 4.0, GaussianSampling::ErfIntegrated, Normalization::L2);
    let sum: f64 = sum_kernel.iter().sum();
    let sum_sq: f64 = l2_kernel.iter().map(|v| v * v).sum();
    assert!(approx_eq(sum, 1.0, 1e-12));
    assert!(approx_eq(sum_sq, 1.0, 1e-12));
}

#[test]
fn f32_and_f64_agree_within_f32_precision() {
    // The convolve entries are generic over the float channel; the same
    // data as f32 or f64 must agree to within f32 precision.
    let mut image_f64 = Array2::<f64>::zeros((10, 11));
    for ((i, j), v) in image_f64.indexed_iter_mut() {
        *v = (i as f64) * 0.3 + (j as f64) * 0.2 + 1.0;
    }
    let image_f32 = image_f64.mapv(|v| v as f32);
    let psf_f64 = Array2::<f64>::from_elem((3, 3), 1.0 / 9.0);
    let psf_f32 = psf_f64.mapv(|v| v as f32);

    let out_f64 = convolve_psf(image_f64.view(), psf_f64.view());
    let out_f32 = convolve_psf(image_f32.view(), psf_f32.view());
    let tolerance = 1e-5;
    for i in 0..10 {
        for j in 0..11 {
            let a = out_f64[(i, j)];
            let b = out_f32[(i, j)] as f64;
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tolerance * scale,
                "f32/f64 mismatch at ({i},{j}): {a} vs {b}"
            );
        }
    }

    // Same for the LSF entry.
    let wavelength_f64 = Grid::<f64>::logspace(4000.0, 7000.0, 128, GridKind::Centers);
    let wavelength_f32 = Grid::<f32>::logspace(4000.0, 7000.0, 128, GridKind::Centers);
    let flux_f64: Array1<f64> = (0..128)
        .map(|i| 1.0 + 0.3 * (i as f64 / 7.0).sin())
        .collect();
    let flux_f32 = flux_f64.mapv(|v| v as f32);
    let lsf_f64 = Spectrum::new(wavelength_f64, flux_f64, None, None)
        .unwrap()
        .convolve_lsf(LsfSpec::ConstantR(3000.0))
        .unwrap();
    let lsf_f32 = Spectrum::new(wavelength_f32, flux_f32, None, None)
        .unwrap()
        .convolve_lsf(LsfSpec::ConstantR(3000.0))
        .unwrap();
    for (a, b) in lsf_f64.flux().iter().zip(lsf_f32.flux().iter()) {
        let b = *b as f64;
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= 1e-4 * scale,
            "lsf f32/f64 mismatch: {a} vs {b}"
        );
    }
}

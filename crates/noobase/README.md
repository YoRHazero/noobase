# noobase (Rust core)

Foundational pure-function utilities for astronomy analysis.

This is the Rust core of the [noobase project](https://github.com/YoRHazero/noobase).
See the [project README](https://github.com/YoRHazero/noobase#readme) for overview, quick start, and Python bindings.

## Use

```toml
[dependencies]
noobase = "0.0.2"
```

## Public surface

- `bins::Grid` — 1-D monotonic axis
- `bins::overlap` — overlap-weighted rebin primitives
- `convolve` — pure 1-D / axis / 2-D correlation kernels, `gaussian1d`, and NaN-aware normalized convolution wrappers
- `spectroscopy::Spectrum` — spectrum container with optional error / mask
- `spectroscopy::LsfSpec` + `Spectrum::convolve_lsf` — Gaussian LSF broadening for noise-free spectral templates
- `photometry::{synthetic, SyntheticOperator}` — synthetic photometry
- `image::reproject_exact` — surface-brightness-conserving image reprojection via planar polygon clipping (rayon-parallel)
- `image::{convolve_psf, convolve_gaussian_axis}` — true 2-D PSF convolution and 1-D Gaussian axis correlation / matched filtering
- `image::stamp::build_stamp` — point-source recenter + fixed-window stamp extraction (sub-pixel centroid recorded, not applied)
- `image::psf` — oversampled-ePSF / extended-PSF construction stack: `build_epsf`, `build_extended_psf`, plus the `robust_combine` / `solve_flux_background` / `stitch_psf` leaves and the `render` / `accumulate` adjoint pair

## Examples

```rust
use ndarray::{Array1, Array2};
use noobase::bins::{Grid, GridKind};
use noobase::image;
use noobase::spectroscopy::{LsfSpec, Spectrum};

let wavelength = Grid::<f64>::linspace(1.0, 5.0, 200, GridKind::Centers);
let flux = Array1::from_elem(200, 1.0);
let template = Spectrum::new(wavelength, flux, None, None).unwrap();
let broadened = template.convolve_lsf(LsfSpec::ConstantR(3000.0)).unwrap();

let image_in = Array2::<f64>::ones((32, 32));
let psf = Array2::<f64>::from_elem((3, 3), 1.0 / 9.0);
let convolved = image::convolve_psf(image_in.view(), psf.view());
assert_eq!(broadened.n_bins(), 200);
assert_eq!(convolved.dim(), (32, 32));
```

## Status

Pre-1.0, API unstable. Breaking changes expected between minor versions.

## License

MIT.

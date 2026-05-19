# noobase

Foundational pure-function utilities for astronomy analysis. Rust core with Python bindings via PyO3.

> Status: pre-1.0, API unstable. Breaking changes expected between minor versions.

## Install

```bash
pip install noobase
```

Requires Python 3.12 or newer. Wheels are published for linux-x86_64, macos-arm64, and windows-x86_64; a source distribution is also available.

## Quick start

```python
import numpy as np
import noobase

wavelength = np.linspace(1.0, 5.0, 200)
flux = np.exp(-((wavelength - 3.0) ** 2) / 0.5)
error = 0.01 * np.ones_like(flux)

spectrum = noobase.spectroscopy.Spectrum(
    wavelength=wavelength,
    flux=flux,
    error=error,
    spacing="linear",
    kind="centers",
)

transmission_grid = np.linspace(2.5, 3.5, 50)
transmission_values = np.exp(-((transmission_grid - 3.0) ** 2) / 0.05)

band_flux, band_error, coverage = spectrum.synthetic_photometry(
    transmission_grid=transmission_grid,
    transmission_values=transmission_values,
    convention="photon_counting",
)
```

## What's in the box

- `Grid` — 1-D monotonic axis (linear / log, centers / edges)
- `Spectrum` — wavelength + flux + optional error + optional mask, with rebinning, flux density convention conversion, and noise-free-template LSF broadening via `Spectrum.convolve_lsf`
- `photometry.SyntheticOperator` — cached synthetic photometry suited for MCMC hot loops
- `image.reproject_exact` — surface-brightness-conserving image reprojection via planar polygon clipping (rayon-parallel; WCS handling stays in the caller's astropy / gwcs)
- `image.make_pixel_corners` — turn a pair of `pixel_to_world_values` / `world_to_pixel_values` callables (astropy.wcs or gwcs) into the corner array consumed by `reproject_exact`, with optional `coarse_step` for expensive WCS chains
- `image.convolve_psf` — true 2-D PSF convolution with NaN-as-missing edge / mask renormalization
- `image.convolve_gaussian_axis` — 1-D Gaussian axis correlation for grism-style matched filtering
- `image.build_stamp` — recenter a point-source cutout into a fixed stamp (sub-pixel centroid recorded, not applied)
- `image.psf.build_epsf` — oversampled ePSF from a stack of under-sampled stamps via projected Landweber / Irani–Peleg super-resolution
- `image.psf.build_extended_psf` — bright-star wing stacking + core↔wing stitch into an encircled-energy-normalised extended PSF, with `robust_combine` / `solve_flux_background` / `stitch_psf` exposed as leaves

## Convolution examples

```python
template = noobase.spectroscopy.Spectrum(
    wavelength=wavelength,
    flux=flux,
    spacing="linear",
    kind="centers",
)
broadened = template.convolve_lsf(spec="constant_r", resolving_power=3000.0)

psf = psf / psf.sum()
model_image = noobase.image.convolve_psf(image, psf)

line_response = noobase.image.convolve_gaussian_axis(
    image,
    sigma=2.5,
    axis=0,
    normalization="l2",
)
```

See the full [project README](https://github.com/YoRHazero/noobase#readme) on GitHub for the complete feature list, the workspace layout, and the development workflow.

## License

MIT.

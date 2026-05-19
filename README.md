# noobase

Foundational pure-function utilities for astronomy analysis. Rust core with Python bindings.

> Status: pre-1.0, API unstable. Breaking changes expected between minor versions.

## Features

- f32/f64 native dispatch (no implicit casts in hot paths)
- `Grid` — 1-D monotonic axis (centers / edges, linear / log)
- `Spectrum` — wavelength + flux + optional error + optional mask
- `bins::overlap` — overlap-weighted rebin, variance propagation, coverage
- `convolve` — pure 1-D / axis / 2-D correlation kernels, Gaussian kernel construction, and NaN-aware normalized-convolution wrappers
- `Spectrum::rebin` — flux + error + mask through the same operator
- `Spectrum::to_f_nu` / `to_f_lambda` — flux density convention conversion
- `Spectrum::convolve_lsf` — Gaussian line-spread-function broadening for noise-free templates (constant resolving power or constant velocity dispersion)
- `photometry::synthetic` + `SyntheticOperator` — synthetic photometry through transmission curves (e.g. JWST NIRCam filters)
- `image::reproject_exact` — surface-brightness-conserving image reprojection via planar polygon clipping (rayon-parallel; WCS handling stays in the caller's astropy / gwcs)
- `image.make_pixel_corners` — Python-side helper that turns a pair of `pixel_to_world_values` / `world_to_pixel_values` callables (e.g. `astropy.wcs` or `gwcs`) into the corner array consumed by `reproject_exact`
- `image::convolve_psf` — true 2-D PSF convolution with NaN-as-missing edge / mask renormalization
- `image::convolve_gaussian_axis` — Gaussian axis correlation for grism-style line matched filtering
- `image::build_stamp` — recenter a point-source cutout and extract a fixed-size stamp; the sub-pixel centroid is recorded as natural dither phase, not applied, so the noise stays uncorrelated
- `image::psf::build_epsf` — oversampled ePSF from a stack of under-sampled stamps, solved as a forward-model super-resolution problem (projected Landweber / Irani–Peleg; flux, background and centroid nuisance solved jointly)
- `image::psf::build_extended_psf` — bright-star wing stacking plus a core↔wing raised-cosine feather, encircled-energy normalised, into a full diffraction-spike/wing extended PSF
- `image::psf::{robust_combine, solve_flux_background, stitch_psf}` — lower-level leaves: sign-agnostic σ-clip / median stack reducer, exact 2×2 weighted LLSQ flux+background solver, core↔wing stitch

## Install

Python (via PyPI):

```bash
uv add noobase
```

Rust: the crate is not yet on crates.io. Pull it from git:

```toml
[dependencies]
noobase = { git = "https://github.com/YoRHazero/noobase.git", tag = "v0.0.2" }
```

## Quick start (Python)

Build a `Spectrum` with a numpy wavelength array plus spacing/kind kwargs, optional error and mask:

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
```

Rebin onto a coarser target grid; error and mask flow through the same operator:

```python
target = np.linspace(1.0, 5.0, 40)
rebinned = spectrum.rebin(target=target, spacing="linear", kind="centers")
print(rebinned.flux.shape, rebinned.error.shape)
```

Broaden a noise-free spectral template with a Gaussian line-spread function. `convolve_lsf` rejects spectra that carry `error` or `mask`, so observed spectra do not get silently re-convolved as if they were templates:

```python
template = noobase.spectroscopy.Spectrum(
    wavelength=wavelength,
    flux=flux,
    spacing="linear",
    kind="centers",
)
broadened = template.convolve_lsf(spec="constant_r", resolving_power=3000.0)
print(broadened.flux.shape)
```

Compute synthetic photometry through a NIRCam-like transmission curve. The return is `(band_flux, band_error, coverage)` — use `coverage` to gate bands that are not fully covered by the spectrum:

```python
transmission_grid = np.linspace(2.5, 3.5, 50)
transmission_values = np.exp(-((transmission_grid - 3.0) ** 2) / 0.05)

band_flux, band_error, coverage = spectrum.synthetic_photometry(
    transmission_grid=transmission_grid,
    transmission_values=transmission_values,
    convention="photon_counting",
)

coverage_threshold = 0.99
if coverage >= coverage_threshold:
    print(f"band flux = {band_flux}, error = {band_error}")
```

For MCMC hot loops, build a `SyntheticOperator` once and reuse it for every model evaluation:

```python
operator = noobase.photometry.SyntheticOperator(
    spectrum_grid=spectrum.wavelength,
    transmission_grid=transmission_grid,
    transmission_values=transmission_values,
    convention="photon_counting",
)
for theta in samples:
    model_flux = forward_model(theta)
    model_error = forward_error(theta)
    flux, error = operator.apply(model_flux, spectrum_error=model_error)
```

Reproject an image onto another image's pixel grid. `noobase` does not depend on astropy or gwcs — `make_pixel_corners` takes the WCSs' `pixel_to_world_values` / `world_to_pixel_values` methods as plain callables and handles the half-pixel corner offset internally. The reprojection itself is then a pure planar polygon clip in input-pixel space, parallelised over output rows:

```python
import noobase

# image_target + wcs_target: the frame you want to align onto.
# image_source + wcs_source: the image to reproject.

pixel_corners = noobase.image.make_pixel_corners(
    image_target.shape,
    target_pixel_to_world=wcs_target.pixel_to_world_values,
    source_world_to_pixel=wcs_source.world_to_pixel_values,
)

image, footprint, weight = noobase.image.reproject_exact(image_source, pixel_corners)
# `image` matches `image_target.shape`; surface brightness is conserved.
# `footprint` is the pure geometric overlap fraction with the source image bounds.
# `weight` is the same restricted to non-NaN inputs; invariant: weight <= footprint.
# Use `weight / footprint` to recover the non-NaN fraction inside the covered region.
```

`make_pixel_corners` works with anything that exposes the two methods, including `astropy.wcs.WCS` and `gwcs`. The callables receive 2-D ndarrays of corner-node pixel coordinates (already shifted by -0.5) and must return world / pixel ndarrays in the same shape — exactly the contract that both libraries already implement.

Convolve an image with a centered PSF, or run a 1-D Gaussian matched filter along one image axis:

```python
psf = psf / psf.sum()
model_image = noobase.image.convolve_psf(image_source, psf)

line_response = noobase.image.convolve_gaussian_axis(
    image_source,
    sigma=2.5,
    axis=0,
    normalization="l2",
    boundary="zero",
    renormalize=False,
)
```

Build a PSF from a batch of point sources. `build_stamp` recenters each cutout into a fixed window (recording, not applying, the sub-pixel centroid); the stamp stack then drives `build_epsf`, which solves an oversampled ePSF as a forward-model super-resolution problem:

```python
import numpy as np
import noobase

# `cutouts`: rough square cutouts, one per point source.
stamp_size = 25
stamps = [noobase.image.build_stamp(c, stamp_size) for c in cutouts]
stamps = [s for s in stamps if s is not None]  # None = centroid / window failed

data = np.stack([s.stamp for s in stamps])        # (N, stamp_size, stamp_size)
delta_init = np.stack([s.delta for s in stamps])  # (N, 2) sub-pixel phase

oversample = 4
epsf = noobase.image.psf.build_epsf(data, delta_init, oversample)
print(epsf.epsf.shape, epsf.converged, epsf.iterations)
```

For the full diffraction spikes and wings, stack larger cutouts of bright stars and stitch them onto the ePSF core:

```python
# `wing_cutouts`: larger cutouts of bright (possibly saturated-core) stars.
wings = [noobase.image.build_stamp(c, 64) for c in wing_cutouts]
wings = [w for w in wings if w is not None]

extended = noobase.image.psf.build_extended_psf(
    wing_data=np.stack([w.stamp for w in wings]),
    wing_delta=np.stack([w.delta for w in wings]),
    core=epsf.epsf,
    oversample=oversample,
)
model = extended.extended  # ExtendedPsf: .core (oversampled) + .wing (native)
```

## Workspace layout

```
crates/noobase     - Rust core (crates.io as `noobase`)
crates/noobase-py  - Python bindings via PyO3 (PyPI as `noobase`)
```

## Development

```bash
cargo test                                    # Rust tests
cargo clippy --workspace --all-targets        # lints
uv run pytest crates/noobase-py/tests/        # Python tests (auto-rebuilds via uv cache-keys)
```

## License

MIT. See [LICENSE](LICENSE).

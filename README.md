# noobase

Foundational pure-function utilities for astronomy analysis. Rust core with Python bindings.

> Status: pre-1.0, API unstable. Breaking changes expected between minor versions.

## Features

- f32/f64 native dispatch (no implicit casts in hot paths)
- `Grid` — 1-D monotonic axis (centers / edges, linear / log)
- `Spectrum` — wavelength + flux + optional error + optional mask
- `bins::overlap` — overlap-weighted rebin, variance propagation, coverage
- `Spectrum::rebin` — flux + error + mask through the same operator
- `Spectrum::to_f_nu` / `to_f_lambda` — flux density convention conversion
- `photometry::synthetic` + `SyntheticOperator` — synthetic photometry through transmission curves (e.g. JWST NIRCam filters)
- `image::reproject_exact` — surface-brightness-conserving image reprojection via planar polygon clipping (rayon-parallel; WCS handling stays in the caller's astropy / gwcs)
- `image.make_pixel_corners` — Python-side helper that turns a pair of `pixel_to_world_values` / `world_to_pixel_values` callables (e.g. `astropy.wcs` or `gwcs`) into the corner array consumed by `reproject_exact`

## Install

After `v0.0.1` publishes, the canonical install is `pip install noobase` for Python and `cargo add noobase` for Rust. In the meantime, install from this repo:

Python (via git):

```bash
pip install git+https://github.com/YoRHazero/noobase.git@v0.0.1
```

Rust:

```toml
[dependencies]
noobase = { git = "https://github.com/YoRHazero/noobase.git", tag = "v0.0.1" }
```

## Quick start (Python)

Build a `Spectrum` with a numpy wavelength array plus spacing/kind kwargs, optional error and mask:

```python
import numpy as np
import noobase

wavelength = np.linspace(1.0, 5.0, 200)
flux = np.exp(-((wavelength - 3.0) ** 2) / 0.5)
error = 0.01 * np.ones_like(flux)

spectrum = noobase.Spectrum(
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

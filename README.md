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

Reproject an image onto another image's pixel grid. `noobase` does not depend on astropy — the caller is expected to pre-compute the corner field using whichever WCS toolkit they prefer (`astropy.wcs`, `gwcs`, …) and pass it in as a plain ndarray. The reprojection itself is then a pure planar polygon clip in input-pixel space, parallelised over output rows:

```python
import numpy as np
from astropy.wcs import WCS
import noobase

# image_reference + wcs_reference: the frame you want to align onto.
# image_other + wcs_other: the image to reproject.

height_out, width_out = image_reference.shape
y_node, x_node = np.indices((height_out + 1, width_out + 1))
x_node = x_node - 0.5  # astropy convention: integer == pixel center, corners at half-integer
y_node = y_node - 0.5

# Map output pixel corners through (WCS_reference -> sky -> WCS_other).
sky = wcs_reference.pixel_to_world_values(x_node, y_node)
x_in, y_in = wcs_other.world_to_pixel_values(*sky)
pixel_corners = np.stack([x_in, y_in], axis=-1)  # (height_out + 1, width_out + 1, 2)

image, footprint, weight = noobase.image.reproject_exact(image_other, pixel_corners)
# `image` matches `image_reference.shape`; surface brightness is conserved.
# `footprint` is the pure geometric overlap fraction with the input image bounds.
# `weight` is the same restricted to non-NaN inputs; invariant: weight <= footprint.
# Use `weight / footprint` to recover the non-NaN fraction inside the covered region.
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

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

spectrum = noobase.Spectrum(
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
- `Spectrum` — wavelength + flux + optional error + optional mask, with rebinning and flux density convention conversion
- `photometry.SyntheticOperator` — cached synthetic photometry suited for MCMC hot loops

See the full [project README](https://github.com/YoRHazero/noobase#readme) on GitHub for the complete feature list, the workspace layout, and the development workflow.

## License

MIT.

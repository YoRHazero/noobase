# noobase

Foundational pure-function utilities for astronomy analysis. The package has a
Rust core and Python bindings, with a focus on deterministic numerical kernels
for spectra, photometry, image reprojection, convolution, and PSF construction.

!!! warning "Pre-1.0"

    noobase is pre-1.0. Public APIs may change between minor versions.

## Install

Python:

```bash
uv add noobase
```

Rust:

```toml
[dependencies]
noobase = { git = "https://github.com/YoRHazero/noobase.git", tag = "v0.0.2" }
```

## Python Quick Start

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

target = np.linspace(1.0, 5.0, 40)
rebinned = spectrum.rebin(target=target, spacing="linear", kind="centers")
```

## API Reference

The Python API reference is generated from runtime docstrings exposed by the
PyO3 bindings.

- [Core](api/core.md): `Grid`
- [Spectroscopy](api/spectroscopy.md): `Spectrum`
- [Overlap](api/overlap.md): bin-overlap rebinning primitives
- [Photometry](api/photometry.md): synthetic photometry
- [Image](api/image.md): reprojection, convolution, and stamp extraction
- [PSF](api/psf.md): ePSF and extended-PSF construction

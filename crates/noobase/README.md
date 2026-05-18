# noobase (Rust core)

Foundational pure-function utilities for astronomy analysis.

This is the Rust core of the [noobase project](https://github.com/YoRHazero/noobase).
See the [project README](https://github.com/YoRHazero/noobase#readme) for overview, quick start, and Python bindings.

## Use

```toml
[dependencies]
noobase = "0.0.1"
```

## Public surface

- `bins::Grid` — 1-D monotonic axis
- `bins::overlap` — overlap-weighted rebin primitives
- `spectroscopy::Spectrum` — spectrum container with optional error / mask
- `photometry::{synthetic, SyntheticOperator}` — synthetic photometry
- `image::reproject_exact` — surface-brightness-conserving image reprojection via planar polygon clipping (rayon-parallel)
- `image::stamp::build_stamp` — point-source recenter + fixed-window stamp extraction (sub-pixel centroid recorded, not applied)
- `image::psf` — oversampled-ePSF / extended-PSF construction stack: `build_epsf`, `build_extended_psf`, plus the `robust_combine` / `solve_flux_background` / `stitch_psf` leaves and the `render` / `accumulate` adjoint pair

## Status

Pre-1.0, API unstable. Breaking changes expected between minor versions.

## License

MIT.

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

## Status

Pre-1.0, API unstable. Breaking changes expected between minor versions.

## License

MIT.

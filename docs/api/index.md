# API Reference

This section is rendered from the Python API exposed by the package. Most
public objects are implemented in Rust through PyO3, so mkdocstrings uses
runtime inspection to read signatures and docstrings from the local `noobase`
package.

Rust API documentation should continue to use `cargo doc` or docs.rs;
this site focuses on the Python-facing package.

## Sections

- [Axis](axis.md)
- [Convolve](convolve.md)
- [Spectroscopy](spectroscopy.md) — `Spectrum` and `synthetic_photometry`
- [Image](image.md)
    - [PSF](psf.md) — `noobase.image.psf`
    - [Aperture](aperture.md) — `noobase.aperture`

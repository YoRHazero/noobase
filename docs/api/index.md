# API Reference

This section is rendered from the Python API exposed by the package. Most
public objects are implemented in Rust through PyO3, so mkdocstrings uses
runtime inspection to read signatures and docstrings from the local `noobase`
package.

Rust API documentation should continue to use `cargo doc` or docs.rs;
this site focuses on the Python-facing package.

## Sections

- [Core](core.md)
- [Spectroscopy](spectroscopy.md)
- [Overlap](overlap.md)
- [Photometry](photometry.md)
- [Image](image.md)
- [PSF](psf.md)

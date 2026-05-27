//! Aperture mask construction for source photometry.
//!
//! This module is the *mask-generation* stage of the photometry pipeline:
//! given a science image (and optional 1-sigma error, segmentation labels,
//! and seed pixels), it produces a boolean mask delimiting the pixels that
//! belong to a given source. It does **not** perform cutout extraction,
//! resampling, coarse segmentation, flux measurement, PSF matching, or
//! forward modeling — those live in upstream/downstream layers.
//!
//! The module is organised by algorithm. The first algorithm shipped is
//! [`region_growing`] (greedy heap-driven growth with a dual geometric
//! annulus stop criterion); Kron / isophotal aperture families are
//! reserved for future sub-modules at the same level.

pub mod region_growing;

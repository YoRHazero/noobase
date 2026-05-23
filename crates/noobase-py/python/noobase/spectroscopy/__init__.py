"""Spectroscopy-domain containers and operations."""

from noobase._core.spectroscopy import Spectrum

from . import synthetic_photometry

__all__ = ["Spectrum", "synthetic_photometry"]

"""Sanity test that every public PyO3 binding carries a useful docstring."""

import noobase


def test_all_bindings_have_docstrings():
    """Every public PyO3 binding must have a non-empty docstring."""
    targets = [
        noobase.Grid,
        noobase.Grid.__init__,
        noobase.Grid.linspace,
        noobase.Grid.logspace,
        noobase.Grid.from_array,
        noobase.Grid.to_edges,
        noobase.Grid.to_centers,
        noobase.Grid.is_uniform,
        noobase.Spectrum,
        noobase.Spectrum.__init__,
        noobase.Spectrum.rebin,
        noobase.Spectrum.to_f_nu,
        noobase.Spectrum.to_f_lambda,
        noobase.Spectrum.synthetic_photometry,
        noobase.overlap.rebin,
        noobase.overlap.rebin_variance,
        noobase.overlap.coverage,
        noobase.photometry.synthetic,
        noobase.photometry.SyntheticOperator,
        noobase.photometry.SyntheticOperator.__init__,
        noobase.photometry.SyntheticOperator.apply,
        noobase.image.reproject_exact,
        noobase.image.build_stamp,
        noobase.image.StampResult,
        noobase.image.psf.robust_combine,
        noobase.image.psf.solve_flux_background,
        noobase.image.psf.build_epsf,
        noobase.image.psf.stitch_psf,
        noobase.image.psf.build_extended_psf,
        noobase.image.psf.RobustCombined,
        noobase.image.psf.FluxBackground,
        noobase.image.psf.BuildEpsf,
        noobase.image.psf.ExtendedPsf,
        noobase.image.psf.ExtendedPsfBuilt,
    ]
    for target in targets:
        doc = target.__doc__
        assert doc is not None and len(doc.strip()) > 50, (
            f"{target!r} has no useful docstring (got: {doc!r})"
        )

    # Properties go through a different access pattern. PyO3 getters expose
    # their docstring on the descriptor; try multiple access paths to be
    # robust across PyO3 versions.
    property_targets = [
        (noobase.Grid, "values"),
        (noobase.Grid, "spacing"),
        (noobase.Grid, "kind"),
        (noobase.Grid, "dtype"),
        (noobase.Spectrum, "wavelength"),
        (noobase.Spectrum, "flux"),
        (noobase.Spectrum, "error"),
        (noobase.Spectrum, "mask"),
        (noobase.Spectrum, "n_bins"),
        (noobase.Spectrum, "dtype"),
        (noobase.photometry.SyntheticOperator, "coverage"),
        (noobase.photometry.SyntheticOperator, "dtype"),
        (noobase.image.StampResult, "stamp"),
        (noobase.image.StampResult, "error"),
        (noobase.image.StampResult, "valid"),
        (noobase.image.StampResult, "delta"),
        (noobase.image.StampResult, "origin"),
        (noobase.image.psf.RobustCombined, "combined"),
        (noobase.image.psf.RobustCombined, "weight"),
        (noobase.image.psf.RobustCombined, "count"),
        (noobase.image.psf.FluxBackground, "flux"),
        (noobase.image.psf.FluxBackground, "background"),
        (noobase.image.psf.FluxBackground, "ok"),
        (noobase.image.psf.BuildEpsf, "epsf"),
        (noobase.image.psf.BuildEpsf, "iterations"),
        (noobase.image.psf.BuildEpsf, "converged"),
        (noobase.image.psf.ExtendedPsf, "core"),
        (noobase.image.psf.ExtendedPsf, "oversample"),
        (noobase.image.psf.ExtendedPsf, "wing"),
        (noobase.image.psf.ExtendedPsfBuilt, "extended"),
        (noobase.image.psf.ExtendedPsfBuilt, "star_ok"),
        (noobase.image.psf.ExtendedPsfBuilt, "star_scale_from_core"),
    ]
    for cls, name in property_targets:
        descriptor = cls.__dict__.get(name) or getattr(cls, name)
        doc = getattr(descriptor, "__doc__", None)
        if doc is None and isinstance(descriptor, property):
            doc = descriptor.fget.__doc__ if descriptor.fget else None
        assert doc is not None and len(doc.strip()) > 20, (
            f"{cls.__name__}.{name} has no docstring (got: {doc!r})"
        )

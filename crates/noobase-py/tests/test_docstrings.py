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
    ]
    for cls, name in property_targets:
        descriptor = cls.__dict__.get(name) or getattr(cls, name)
        doc = getattr(descriptor, "__doc__", None)
        if doc is None and isinstance(descriptor, property):
            doc = descriptor.fget.__doc__ if descriptor.fget else None
        assert doc is not None and len(doc.strip()) > 20, (
            f"{cls.__name__}.{name} has no docstring (got: {doc!r})"
        )

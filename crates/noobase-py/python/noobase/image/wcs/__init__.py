"""Compiled WCS transform programs.

``WcsProgram`` evaluates a coordinate-transform chain that was *compiled*
elsewhere into a flat op-list spec -- typically by a caller that walks a
gwcs / astropy compound-model tree once and extracts plain coefficients.
Evaluation is pure ``float64`` array math with the GIL released, plus a
numpy-free scalar fast path; there is no model tree and no per-call
dispatch overhead.

Spec schema
-----------
A spec is a JSON-able dict describing a register machine::

    {
        "n_regs": int,          # number of f64 virtual registers
        "inputs": [int, ...],   # registers filled from the call arguments
        "outputs": [int, ...],  # registers returned by the call
        "ops": [op, ...],       # executed in order, element-wise
    }

Each op is a dict with ``"op"`` (the op-code), ``"in"`` / ``"out"``
(register lists; arities are fixed per op-code) and op-specific parameters:

===============  =======  ============================================================
op-code          arity    parameters
===============  =======  ============================================================
shift            1 -> 1   ``offset``
scale            1 -> 1   ``factor``
const            0 -> 1   ``value``
poly1d           1 -> 1   ``coeffs`` -- ``[c0, c1, ...]``, :math:`\\sum_i c_i x^i`
poly2d           2 -> 1   ``degree``, ``coeffs`` -- dense ``(degree+1)**2`` list,
                          row-major over ``c[i][j] x^i y^j`` (row = x power)
affine2          2 -> 2   ``matrix`` (2x2 nested), ``translation`` (len 2)
sph2cart         2 -> 3   -- (lon, lat in degrees -> unit x, y, z)
cart2sph         3 -> 2   ``wrap_lon_at`` -- 360 or 180
rot3             3 -> 3   ``matrix`` (3x3 nested)
tan_project      2 -> 2   -- (native spherical deg -> tangent plane deg)
tan_deproject    2 -> 2   -- (tangent plane deg -> native spherical deg)
grism_forward    5 -> 1   ``axis`` ('row'/'column'), ``orders``, ``alongdisp``,
                          ``lmodels`` -- per-order t-polynomial lists;
                          inputs (x, y, x0, y0, order) -> wavelength
grism_backward   4 -> 2   ``orders``, ``lmodels``, ``xmodels``, ``ymodels``;
                          inputs (x0, y0, wavelength, order) -> (x, y) dispersed
binary           2 -> 1   ``kind`` -- 'add' / 'sub' / 'mul' / 'div'
unitless2dircos  2 -> 3   -- ((x, y) -> (x, y, 1) / |(x, y, 1)|)
dircos2unitless  3 -> 2   -- ((x, y, z) -> (x/z, y/z))
grating_wavelen  2 -> 1   ``grating_wavelength``: ``factor`` (= groove density x
                          spectral order); (a_in, a_out) -> (a_in + a_out) / factor
grating_angles3d 3 -> 3   ``factor``; (lam, a_in, b_in) ->
                          (a_in - factor*lam, -b_in, sqrt(1 - a^2 - b^2))
tabular1d        1 -> 1   ``points`` (strictly ascending), ``values``, ``fill``;
                          linear interpolation, ``fill`` outside the domain
logical          1 -> 1   ``condition`` ('GT'/'LT'/'EQ'/'NE'), ``compareto``,
                          ``value``; substitute where true, NaN passes through
select           k -> m   ``label``, ``cases`` -- piecewise transform, see below
===============  =======  ============================================================

The ``select`` op (gwcs ``RegionsSelector``, used by NIRSpec IFU and MIRI
MRS) routes each element by a region label to one of its sub-programs::

    {"op": "select", "in": [...], "out": [...],
     "label": {"kind": "array", "data": <2-D int64 numpy array>}      # per-pixel
            | {"kind": "dict", "keys": [...], "labels": [...],       # quantized
               "key_input": 0, "atol": 1e-4},
     "cases": [{"label": 1, "program": <nested spec>}, ...]}

Array labels index ``data[floor(y + 0.5), floor(x + 0.5)]`` from the op's
first two inputs; dict labels match ``inputs[key_input]`` against ``keys``
within ``atol``. Label 0 (or out-of-bounds / no matching case) yields NaN
on every output. Sub-program arities must all match the op wiring. Note
the label array makes such specs numpy-carrying rather than purely
JSON-able; serialise it separately if caching.

A *t-polynomial* (grism trace model, stdatamodels "guess form") is either
``{"kind": "t", "coeffs": [...]}`` (depends only on t) or ``{"kind":
"spatial", "degree": d, "coeffs": [poly2d-list, ...]}`` where entry ``i`` is
the dense ``poly2d`` coefficient list of the ``t**i`` coefficient's spatial
dependence.

Semantics notes
---------------
- Plumbing models (``Mapping``, ``Identity``) must be compiled away into
  register wiring; there are deliberately no op-codes for them.
- The grism ``order`` input is read from the first element only (one order
  per call, matching the JWST transforms); it must be constant.
- Where jwst inverts grism trace polynomials on a sampled grid, these
  programs solve the (at most quadratic) polynomial exactly: round-trips
  are machine-precision, agreement with jwst is bounded by *jwst's*
  sampling error.
- The op-code ``select`` is reserved for future piecewise transforms
  (NIRSpec IFU slices, MIRI MRS regions) and is not implemented.
"""

from noobase._core.image import WcsProgram

__all__ = ["WcsProgram"]

"""Image-domain primitives (planar, no WCS)."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from noobase._core.image import reproject_exact

__all__ = ["make_pixel_corners", "reproject_exact"]


def make_pixel_corners(
    output_shape: tuple[int, int],
    *,
    output_pixel_to_world: Callable,
    input_world_to_pixel: Callable,
) -> NDArray[np.float64]:
    """Build the ``pixel_corners`` array consumed by ``reproject_exact``.

    Maps every node of the output pixel grid through
    ``output_pixel_to_world`` and then ``input_world_to_pixel`` to obtain
    the corresponding location in the input image's pixel coordinate
    system. The half-pixel corner offset required by the astropy / gwcs
    convention (integer ``(x, y)`` is the *center* of a pixel) is applied
    internally so the caller does not have to remember it.

    This helper does NOT depend on astropy or gwcs. Any pair of
    callables matching the documented signatures will work, including
    composed transforms, mock WCSs in tests, and FITS-WCS / gwcs
    objects' ``pixel_to_world_values`` / ``world_to_pixel_values``
    methods.

    Parameters
    ----------
    output_shape : tuple of (int, int)
        ``(H_out, W_out)`` — the shape of the reprojected image you want.
        Usually this is the shape of the reference image you are aligning
        onto.
    output_pixel_to_world : callable
        Forward transform on the *output* WCS. Signature::

            output_pixel_to_world(x, y) -> tuple of ndarray

        where ``x`` and ``y`` are 2-D ndarrays of output pixel
        coordinates (integer = pixel center) and the return is a tuple
        of world-coordinate arrays (for example ``(ra, dec)``).
        ``astropy.wcs.WCS.pixel_to_world_values`` and the equivalent
        gwcs API match this signature.
    input_world_to_pixel : callable
        Inverse transform on the *input* WCS. Signature::

            input_world_to_pixel(*world_arrays) -> (x_in, y_in)

        where the inputs are the world arrays produced by
        ``output_pixel_to_world`` and the return is the pair of 2-D
        ndarrays giving input-image pixel coordinates.
        ``astropy.wcs.WCS.world_to_pixel_values`` matches this
        signature.

    Returns
    -------
    ndarray
        ``pixel_corners`` of shape ``(H_out + 1, W_out + 1, 2)`` and
        dtype ``float64``. The last axis is ``(x_in, y_in)``. Pass it
        directly to ``reproject_exact``.

    Notes
    -----
    The callables are evaluated once on the full ``(H_out + 1, W_out + 1)``
    corner grid. For expensive transforms (notably JWST gwcs pipelines
    that solve a numerical inverse), this means one large vectorised
    call rather than per-pixel calls.
    """
    height_out, width_out = output_shape
    y_node, x_node = np.indices((height_out + 1, width_out + 1))
    x_pixel = x_node.astype(np.float64) - 0.5
    y_pixel = y_node.astype(np.float64) - 0.5
    world = output_pixel_to_world(x_pixel, y_pixel)
    if not isinstance(world, tuple):
        world = (world,)
    x_in, y_in = input_world_to_pixel(*world)
    pixel_corners = np.empty((height_out + 1, width_out + 1, 2), dtype=np.float64)
    pixel_corners[..., 0] = x_in
    pixel_corners[..., 1] = y_in
    return pixel_corners

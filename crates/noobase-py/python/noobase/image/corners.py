"""Build the ``pixel_corners`` array consumed by ``reproject_exact``.

Pure Python — no astropy, gwcs, or PyO3 dependency. Any pair of
callables matching the documented signatures will work, including
composed transforms, mock WCSs in tests, and FITS-WCS / gwcs objects'
``pixel_to_world_values`` / ``world_to_pixel_values`` methods.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def make_pixel_corners(
    target_shape: tuple[int, int],
    *,
    target_pixel_to_world: Callable,
    source_world_to_pixel: Callable,
) -> NDArray[np.float64]:
    """Build the ``pixel_corners`` array consumed by ``reproject_exact``.

    Maps every node of the *target* pixel grid (the frame you are
    aligning onto) through ``target_pixel_to_world`` and then
    ``source_world_to_pixel`` to obtain the corresponding location in
    the *source* image's pixel coordinate system (the image you are
    reprojecting). The half-pixel corner offset required by the astropy
    / gwcs convention (integer ``(x, y)`` is the *center* of a pixel)
    is applied internally so the caller does not have to remember it.

    The terminology follows image registration: ``source`` is the image
    being reprojected (the data you have), ``target`` is the frame you
    are aligning onto (the goal). In a later call to
    ``reproject_exact``, ``source`` will be passed as ``image_in`` and
    the corner array produced here projects each target pixel into the
    source's pixel coordinates.

    Parameters
    ----------
    target_shape : tuple of (int, int)
        ``(H_target, W_target)`` — the shape of the reprojected image
        you want. Usually this is the shape of the reference image you
        are aligning onto.
    target_pixel_to_world : callable
        Forward transform on the *target* WCS. Signature::

            target_pixel_to_world(x, y) -> tuple of ndarray

        where ``x`` and ``y`` are 2-D ndarrays of target pixel
        coordinates (integer = pixel center) and the return is a tuple
        of world-coordinate arrays (for example ``(ra, dec)``).
        ``astropy.wcs.WCS.pixel_to_world_values`` and the equivalent
        gwcs API match this signature.
    source_world_to_pixel : callable
        Inverse transform on the *source* WCS. Signature::

            source_world_to_pixel(*world_arrays) -> (x_source, y_source)

        where the inputs are the world arrays produced by
        ``target_pixel_to_world`` and the return is the pair of 2-D
        ndarrays giving source-image pixel coordinates.
        ``astropy.wcs.WCS.world_to_pixel_values`` matches this
        signature.

    Returns
    -------
    ndarray
        ``pixel_corners`` of shape ``(H_target + 1, W_target + 1, 2)``
        and dtype ``float64``. The last axis is
        ``(x_source, y_source)``. Pass it directly to
        ``reproject_exact``.

    Notes
    -----
    The callables are evaluated once on the full
    ``(H_target + 1, W_target + 1)`` corner grid. For expensive
    transforms (notably JWST gwcs pipelines that solve a numerical
    inverse), this means one large vectorised call rather than
    per-pixel calls.
    """
    height_target, width_target = target_shape
    y_node, x_node = np.indices((height_target + 1, width_target + 1))
    x_target = x_node.astype(np.float64) - 0.5
    y_target = y_node.astype(np.float64) - 0.5
    world = target_pixel_to_world(x_target, y_target)
    if not isinstance(world, tuple):
        world = (world,)
    x_source, y_source = source_world_to_pixel(*world)
    pixel_corners = np.empty(
        (height_target + 1, width_target + 1, 2), dtype=np.float64
    )
    pixel_corners[..., 0] = x_source
    pixel_corners[..., 1] = y_source
    return pixel_corners

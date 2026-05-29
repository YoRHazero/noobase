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
    coarse_step: tuple[int, int] | None = None,
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
    ``reproject_exact``, ``source`` will be passed as ``image`` and
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
    coarse_step : sequence of (int, int), optional
        Any 2-element sequence of positive ints — tuple, list, or
        ndarray are all accepted. If given as ``(step_h, step_w)``,
        evaluate the WCS callables on a coarse subgrid stepping every
        ``step_h`` rows and ``step_w`` columns of the target pixel
        grid, then bicubic-interpolate the result to the full
        ``(H_target + 1, W_target + 1)`` corner grid. Useful when the
        callables are expensive (notably JWST gwcs pipelines that solve
        a numerical inverse per evaluation): a 2048x2048 target with
        ``coarse_step=(64, 64)`` reduces the WCS call count from ~4M
        to ~1k. Both components must be positive and must divide their
        corresponding ``target_shape`` axis exactly; otherwise
        ``ValueError`` is raised. Default ``None`` evaluates the
        callables on the full corner grid (no interpolation).

    Returns
    -------
    ndarray
        ``pixel_corners`` of shape ``(H_target + 1, W_target + 1, 2)``
        and dtype ``float64``. The last axis is
        ``(x_source, y_source)``. Pass it directly to
        ``reproject_exact``.

    Notes
    -----
    Without ``coarse_step``, the callables are evaluated once on the
    full ``(H_target + 1, W_target + 1)`` corner grid in a single
    vectorised call.

    With ``coarse_step``, the bicubic upsampling uses Catmull-Rom
    cubic weights with linear extrapolation at the boundary (exact
    reconstruction for linear corner fields, residual ~h^2 * curvature
    in the boundary cells for smooth fields). For typical astronomical
    WCS pairs with ``coarse_step`` in the 16-128 range, the residual on
    the corner positions is well below 0.01 source-pixel, which is
    negligible compared to reprojection's polygon-clip accuracy.
    """
    height_target, width_target = target_shape

    if coarse_step is None:
        return _evaluate_dense_corners(
            target_shape,
            target_pixel_to_world,
            source_world_to_pixel,
        )

    # Accept any 2-element sequence (tuple, list, ndarray, ...).
    try:
        items = tuple(coarse_step)
    except TypeError:
        raise ValueError(
            f"coarse_step must be a length-2 sequence of positive ints; "
            f"got {coarse_step!r}"
        )
    if len(items) != 2:
        raise ValueError(
            f"coarse_step must be a length-2 sequence of positive ints; "
            f"got length {len(items)} from {coarse_step!r}"
        )
    try:
        step_h = int(items[0])
        step_w = int(items[1])
    except (TypeError, ValueError):
        raise ValueError(
            f"coarse_step components must be integer-valued; got {coarse_step!r}"
        )
    # Reject silent truncation (e.g. 1.5 -> 1).
    if step_h != items[0] or step_w != items[1]:
        raise ValueError(
            f"coarse_step components must be exact integers; got {coarse_step!r}"
        )
    if step_h <= 0 or step_w <= 0:
        raise ValueError(
            f"coarse_step components must be positive; got {coarse_step!r}"
        )
    if height_target % step_h != 0 or width_target % step_w != 0:
        raise ValueError(
            f"coarse_step {(step_h, step_w)} must divide target_shape "
            f"{target_shape} on each axis"
        )

    coarse_corners = _evaluate_coarse_corners(
        target_shape,
        (step_h, step_w),
        target_pixel_to_world,
        source_world_to_pixel,
    )
    return _bicubic_oversample(
        coarse_corners,
        target_corner_shape=(height_target + 1, width_target + 1),
        step=(step_h, step_w),
    )


def _evaluate_dense_corners(
    target_shape: tuple[int, int],
    target_pixel_to_world: Callable,
    source_world_to_pixel: Callable,
) -> NDArray[np.float64]:
    """Evaluate the WCS chain on the full target corner grid."""
    height_target, width_target = target_shape
    y_node, x_node = np.indices((height_target + 1, width_target + 1))
    x_target = x_node.astype(np.float64) - 0.5
    y_target = y_node.astype(np.float64) - 0.5
    return _project_through_chain(
        x_target, y_target, target_pixel_to_world, source_world_to_pixel
    )


def _evaluate_coarse_corners(
    target_shape: tuple[int, int],
    coarse_step: tuple[int, int],
    target_pixel_to_world: Callable,
    source_world_to_pixel: Callable,
) -> NDArray[np.float64]:
    """Evaluate the WCS chain on a coarse subgrid of target corners.

    Coarse corner ``(i, j)`` sits at target pixel position
    ``(i * step_h - 0.5, j * step_w - 0.5)`` — that is, on the same set
    of half-integer corner lines as the dense path, just thinned out.
    """
    height_target, width_target = target_shape
    step_h, step_w = coarse_step
    height_coarse = height_target // step_h
    width_coarse = width_target // step_w
    j_node, i_node = np.meshgrid(
        np.arange(width_coarse + 1, dtype=np.float64),
        np.arange(height_coarse + 1, dtype=np.float64),
    )
    x_target = j_node * step_w - 0.5
    y_target = i_node * step_h - 0.5
    return _project_through_chain(
        x_target, y_target, target_pixel_to_world, source_world_to_pixel
    )


def _project_through_chain(
    x_target: NDArray[np.float64],
    y_target: NDArray[np.float64],
    target_pixel_to_world: Callable,
    source_world_to_pixel: Callable,
) -> NDArray[np.float64]:
    world = target_pixel_to_world(x_target, y_target)
    if not isinstance(world, tuple):
        world = (world,)
    x_source, y_source = source_world_to_pixel(*world)
    corners = np.empty(x_target.shape + (2,), dtype=np.float64)
    corners[..., 0] = x_source
    corners[..., 1] = y_source
    return corners


def _catmull_rom_weights(fraction: NDArray[np.float64]) -> NDArray[np.float64]:
    """Catmull-Rom cubic Hermite weights for tap offsets (-1, 0, +1, +2).

    Returns array of shape ``(..., 4)``. The kernel is exact for cubic
    polynomials when all four taps are real samples.
    """
    t = fraction
    t_squared = t * t
    t_cubed = t_squared * t
    weight_minus_one = 0.5 * (-t_cubed + 2.0 * t_squared - t)
    weight_zero = 0.5 * (3.0 * t_cubed - 5.0 * t_squared + 2.0)
    weight_plus_one = 0.5 * (-3.0 * t_cubed + 4.0 * t_squared + t)
    weight_plus_two = 0.5 * (t_cubed - t_squared)
    return np.stack(
        [weight_minus_one, weight_zero, weight_plus_one, weight_plus_two],
        axis=-1,
    )


def _bicubic_oversample(
    coarse: NDArray[np.float64],
    *,
    target_corner_shape: tuple[int, int],
    step: tuple[int, int],
) -> NDArray[np.float64]:
    """Bicubic Catmull-Rom oversample of a coarse corner field.

    Boundary handling is linear extrapolation: the virtual samples one
    step beyond each edge are reconstructed from the two innermost real
    samples on that axis. This is exact for linear corner fields (so
    identity and affine WCS chains round-trip bit-for-bit) and leaves
    residual ``O(h^2 * curvature)`` for smooth fields in the boundary
    cells only — interior cells reach the full ``O(h^4)`` cubic
    accuracy.
    """
    height_target_corners, width_target_corners = target_corner_shape
    step_h, step_w = step

    # Extend the coarse corner field with linearly-extrapolated rows /
    # columns: one virtual sample above the top, two below the bottom,
    # one to the left, two to the right. The asymmetric padding matches
    # the 4-tap kernel's reach (offsets -1, 0, +1, +2 relative to the
    # floor of the sample position).
    extra_top = 2.0 * coarse[0:1] - coarse[1:2]
    extra_bottom_one = 2.0 * coarse[-1:] - coarse[-2:-1]
    extra_bottom_two = 3.0 * coarse[-1:] - 2.0 * coarse[-2:-1]
    extended_rows = np.concatenate(
        [extra_top, coarse, extra_bottom_one, extra_bottom_two], axis=0
    )
    extra_left = 2.0 * extended_rows[:, 0:1] - extended_rows[:, 1:2]
    extra_right_one = 2.0 * extended_rows[:, -1:] - extended_rows[:, -2:-1]
    extra_right_two = 3.0 * extended_rows[:, -1:] - 2.0 * extended_rows[:, -2:-1]
    extended = np.concatenate(
        [extra_left, extended_rows, extra_right_one, extra_right_two], axis=1
    )

    # Fine corner i (0..H_target) corresponds to coarse-grid position
    # i / step_h. The integer part picks the cell; the fractional part
    # drives the kernel weights.
    i_fine = np.arange(height_target_corners, dtype=np.float64)
    j_fine = np.arange(width_target_corners, dtype=np.float64)
    u_position = i_fine / step_h
    v_position = j_fine / step_w
    u_floor = np.floor(u_position).astype(np.int64)
    v_floor = np.floor(v_position).astype(np.int64)
    u_fraction = u_position - u_floor
    v_fraction = v_position - v_floor

    weights_y = _catmull_rom_weights(u_fraction)
    weights_x = _catmull_rom_weights(v_fraction)

    # Separable application: contract along the row axis first to
    # produce an intermediate of shape (H_target + 1, W_coarse + 4, 2),
    # then along the column axis to reach (H_target + 1, W_target + 1, 2).
    # Looping over the 4 taps keeps peak memory at one full output-sized
    # buffer instead of growing it by a factor of 4.
    intermediate_shape = (height_target_corners, extended.shape[1], 2)
    intermediate = np.zeros(intermediate_shape, dtype=np.float64)
    for tap in range(4):
        intermediate += (
            weights_y[:, tap : tap + 1, None] * extended[u_floor + tap, :, :]
        )

    output = np.zeros(
        (height_target_corners, width_target_corners, 2), dtype=np.float64
    )
    for tap in range(4):
        output += (
            weights_x[None, :, tap : tap + 1]
            * intermediate[:, v_floor + tap, :]
        )
    return output

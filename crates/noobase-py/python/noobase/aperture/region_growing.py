"""Region-growing aperture mask construction (pure-Python wrapper).

Layers policy defaults on top of the typed Rust binding at
``noobase._core.aperture.grow_mask``:

- ``snr_threshold`` follows ``err``: when ``snr_threshold`` is not
  passed, it defaults to ``2.0`` if ``err`` is provided, else to
  ``None`` (SNR stop disabled). The auto-rule cannot be expressed in
  the typed Rust signature because PyO3 defaults are static. Pass
  ``snr_threshold=None`` explicitly to force-disable SNR even with
  ``err`` provided; pass a number to force-enable it.
- ``seed_pixels`` accepts a 2-D numpy ``int`` array of shape ``(N, 2)``
  in addition to the native sequence-of-``(row, col)`` form, so callers
  who already have coordinates as an array do not have to convert.
- ``shape_weight`` is interpreted in units of the noise level of the
  image that drives growth (``detection`` if given, else ``data``). The
  wrapper multiplies it by a robust ``mad_std`` estimate of that image
  before passing the absolute weight to the core, so the same
  ``shape_weight`` ports across images and bands and only biases growth
  near the noise floor — exactly where spurious tendrils form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from noobase._core.aperture import GrowthResult, grow_mask as _grow_mask_core

__all__ = ["GrowthResult", "grow_mask"]

_UNSET: object = object()


def _noise_scale(image: NDArray[np.float64]) -> float:
    """Robust scalar noise estimate (MAD-based sigma) of ``image``.

    Used to put the dimensionless ``shape_weight`` into the value units
    of the heap-key image. NaNs are ignored. A degenerate estimate
    (flat image, all-NaN) falls back to ``1.0`` so the shape term keeps
    its nominal magnitude rather than collapsing to zero.
    """
    median = np.nanmedian(image)
    scale = 1.4826 * float(np.nanmedian(np.abs(image - median)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def grow_mask(
    data: NDArray[np.float64],
    seed_pixels: Sequence[tuple[int, int]] | NDArray[np.integer],
    *,
    detection: Optional[NDArray[np.float64]] = None,
    err: Optional[NDArray[np.float64]] = None,
    label_map: Optional[NDArray[np.int32]] = None,
    label_allowed: Optional[Sequence[int]] = None,
    connectivity: str = "eight",
    shape_weight: float = 1.0,
    min_neighbor_support: int = 2,
    min_pixels_before_shape_gate: int = 8,
    snr_threshold: Optional[float] = _UNSET,  # type: ignore[assignment]
    snr_hysteresis: int = 3,
    gradient_ratio_threshold: Optional[float] = 1.0,
    gradient_hysteresis: int = 3,
    gradient_lo_percentile: float = 75.0,
    gradient_hi_percentile: float = 99.0,
    min_pixels_before_stop_check: int = 30,
    check_interval: int = 5,
    annulus_thickness: int = 2,
) -> GrowthResult:
    """Grow a boolean source mask outward from one or more seed pixels.

    Growth is a greedy region grow on a max-heap. Each candidate pixel's
    priority is its ``detection`` value plus a shape reward proportional
    to how many of its neighbours are already in the mask, so growth
    fills concavities before extending arms. A hard ``min_neighbor_support``
    floor (after a warm-up) refuses pixels with too few in-mask
    neighbours, forbidding one-pixel-wide tendrils. Two annulus stop
    criteria (SNR and radial gradient) decide when to stop, measured on
    ``data``.

    Parameters
    ----------
    data : ndarray
        2-D image of dtype ``float64``. Used for the stop statistics and
        returned to the caller for photometry. Also the default heap
        key when ``detection`` is not given.
    seed_pixels : sequence of (int, int) or ndarray of shape (N, 2)
        One or more ``(row, col)`` starting pixels. All must lie inside
        ``data``; when a label is given, all must sit on an allowed
        label. Seeds are admitted unconditionally.
    detection : ndarray, optional
        2-D image, same shape and dtype as ``data``, driving the heap
        priority. Pass a smoothed / matched-filter image here to keep
        growth from chasing single-pixel noise ridges; defaults to
        ``data`` (plain brightest-pixel growth).
    err : ndarray, optional
        2-D 1-sigma error image, same shape and dtype as ``data``.
        Required if (and only if) the SNR stop is enabled.
    label_map : ndarray, optional
        2-D ``int32`` segmentation map, same shape as ``data``. Must be
        paired with ``label_allowed`` (both or neither).
    label_allowed : sequence of int, optional
        Whitelisted labels the mask may occupy. Include the background
        label explicitly if growth into background is desired.
    connectivity : {"four", "eight"}, optional
        Pixel adjacency used for heap-neighbour expansion, neighbour
        support, and annulus dilation. Default ``"eight"``.
    shape_weight : float, optional
        Soft compactness bias, in units of the heap-key image noise
        level (see the module docstring). ``0`` disables the soft term.
        Default ``1.0``.
    min_neighbor_support : int, optional
        Hard floor on a candidate's in-mask neighbour count, enforced
        once ``min_pixels_before_shape_gate`` pixels are admitted. ``2``
        forbids one-pixel-wide tendrils under either connectivity; ``1``
        (or ``0``) disables the floor. Must not exceed the connectivity's
        neighbour count (4 or 8). Default ``2``.
    min_pixels_before_shape_gate : int, optional
        Admitted-pixel count before the ``min_neighbor_support`` floor
        activates, letting the seed core establish itself first. Default
        ``8``.
    snr_threshold : float, optional
        Threshold for the cumulative inner-annulus signal-to-noise stop.
        If unspecified, defaults to ``2.0`` when ``err`` is provided and
        to ``None`` otherwise (auto-follows ``err``). Pass ``None`` to
        force-disable the SNR stop.
    snr_hysteresis : int, optional
        Consecutive checks with SNR below threshold required to fire.
        Default ``3``.
    gradient_ratio_threshold : float, optional
        Threshold for the ``mean(outer) / mean(inner)`` radial-gradient
        stop. ``> 1`` corresponds to "outer is brighter than inner",
        i.e. the radial profile has flipped. Default ``1.0``. Pass
        ``None`` to disable the gradient stop.
    gradient_hysteresis : int, optional
        Consecutive checks with ratio above threshold required to fire.
        Default ``3``.
    gradient_lo_percentile, gradient_hi_percentile : float, optional
        Each annulus is summarised, in the gradient ratio, by the mean of
        its pixels within this percentile band rather than a plain mean.
        The lower bound (default ``75``) drops the sky pixels that
        otherwise dilute the ring and hide a rising neighbour; the upper
        bound (default ``99``) trims isolated hot pixels by count (a
        multi-pixel neighbour survives). Must satisfy
        ``0 <= lo < hi <= 100``; ``(0, 100)`` recovers the plain mean.
    min_pixels_before_stop_check : int, optional
        Lower bound on the admitted-pixel count before stop checks may
        fire. Default ``30``.
    check_interval : int, optional
        Evaluate stop criteria every ``check_interval`` admitted
        pixels. Must be ``>= 1``. Default ``5``.
    annulus_thickness : int, optional
        Per-ring dilation iteration count. Default ``2``.

    Returns
    -------
    GrowthResult
        ``.mask`` (bool ndarray), ``.stop_reason`` (``"snr_below"`` /
        ``"gradient_flip"`` / ``"filled"``), ``.n_iterations`` (int).
        ``"filled"`` is the failure outcome: the cutout was too small
        for the source.

    Raises
    ------
    ValueError
        For any invalid input — see the parameter descriptions for the
        constraints; the message identifies the specific violation.
    """
    if snr_threshold is _UNSET:
        snr_threshold = 2.0 if err is not None else None

    if isinstance(seed_pixels, np.ndarray):
        if seed_pixels.ndim != 2 or seed_pixels.shape[1] != 2:
            raise ValueError(
                "seed_pixels numpy array must be 2-D with shape (N, 2); "
                f"got shape {seed_pixels.shape}"
            )
        seed_pixels = [(int(row), int(col)) for row, col in seed_pixels]

    # ``shape_weight`` is dimensionless on this surface; scale it into
    # the value units of whichever image drives the heap.
    heap_image = detection if detection is not None else data
    shape_weight_absolute = shape_weight * _noise_scale(heap_image)

    return _grow_mask_core(
        data,
        seed_pixels,
        detection=detection,
        err=err,
        label_map=label_map,
        label_allowed=label_allowed,
        connectivity=connectivity,
        shape_weight=shape_weight_absolute,
        min_neighbor_support=min_neighbor_support,
        min_pixels_before_shape_gate=min_pixels_before_shape_gate,
        snr_threshold=snr_threshold,
        snr_hysteresis=snr_hysteresis,
        gradient_ratio_threshold=gradient_ratio_threshold,
        gradient_hysteresis=gradient_hysteresis,
        gradient_lo_percentile=gradient_lo_percentile,
        gradient_hi_percentile=gradient_hi_percentile,
        min_pixels_before_stop_check=min_pixels_before_stop_check,
        check_interval=check_interval,
        annulus_thickness=annulus_thickness,
    )

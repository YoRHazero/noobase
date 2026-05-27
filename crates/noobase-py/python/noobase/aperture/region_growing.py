"""Region-growing aperture mask construction (pure-Python wrapper).

Layers two policy defaults on top of the typed Rust binding at
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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from noobase._core.aperture import GrowthResult, grow_mask as _grow_mask_core

__all__ = ["GrowthResult", "grow_mask"]

_UNSET: object = object()


def grow_mask(
    data: NDArray[np.float64],
    seed_pixels: Sequence[tuple[int, int]] | NDArray[np.integer],
    *,
    err: Optional[NDArray[np.float64]] = None,
    label_map: Optional[NDArray[np.int32]] = None,
    label_allowed: Optional[Sequence[int]] = None,
    connectivity: str = "eight",
    snr_threshold: Optional[float] = _UNSET,  # type: ignore[assignment]
    snr_hysteresis: int = 3,
    gradient_ratio_threshold: Optional[float] = 1.0,
    gradient_hysteresis: int = 3,
    min_pixels_before_stop_check: int = 30,
    check_interval: int = 5,
    annulus_thickness: int = 2,
) -> GrowthResult:
    """Grow a boolean source mask outward from one or more seed pixels.

    Parameters
    ----------
    data : ndarray
        2-D image of dtype ``float64``. Pixel values are the heap key.
    seed_pixels : sequence of (int, int) or ndarray of shape (N, 2)
        One or more ``(row, col)`` starting pixels. All must lie inside
        ``data``; when a label is given, all must sit on an allowed
        label.
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
        Pixel adjacency used both for heap-neighbour expansion and for
        annulus dilation. Default ``"eight"``.
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

    return _grow_mask_core(
        data,
        seed_pixels,
        err=err,
        label_map=label_map,
        label_allowed=label_allowed,
        connectivity=connectivity,
        snr_threshold=snr_threshold,
        snr_hysteresis=snr_hysteresis,
        gradient_ratio_threshold=gradient_ratio_threshold,
        gradient_hysteresis=gradient_hysteresis,
        min_pixels_before_stop_check=min_pixels_before_stop_check,
        check_interval=check_interval,
        annulus_thickness=annulus_thickness,
    )

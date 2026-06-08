from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

SeedPixels = Union[Sequence[tuple[int, int]], NDArray[np.integer]]
StopReason = Literal["snr_below", "gradient_flip", "filled"]
Connectivity = Literal["four", "eight"]


class GrowthResult:
    @property
    def mask(self) -> NDArray[np.bool_]: ...
    @property
    def stop_reason(self) -> StopReason: ...
    @property
    def n_iterations(self) -> int: ...


def grow_mask(
    data: NDArray[np.float64],
    seed_pixels: SeedPixels,
    *,
    detection: Optional[NDArray[np.float64]] = None,
    err: Optional[NDArray[np.float64]] = None,
    label_map: Optional[NDArray[np.int32]] = None,
    label_allowed: Optional[Sequence[int]] = None,
    connectivity: Connectivity = "eight",
    shape_weight: float = 1.0,
    min_neighbor_support: int = 2,
    min_pixels_before_shape_gate: int = 8,
    fill_min_cardinal_support: Optional[int] = 3,
    # Auto-follows ``err`` when not passed (2.0 if err is not None, else None).
    snr_threshold: Optional[float] = ...,
    snr_hysteresis: int = 3,
    gradient_ratio_threshold: Optional[float] = 1.0,
    gradient_hysteresis: int = 3,
    gradient_lo_percentile: float = 75.0,
    gradient_hi_percentile: float = 99.0,
    min_pixels_before_stop_check: int = 30,
    check_interval: int = 5,
    annulus_thickness: int = 2,
) -> GrowthResult: ...

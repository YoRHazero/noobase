from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from . import psf as psf


def reproject_exact(
    image_in: NDArray[Any],
    pixel_corners: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def make_pixel_corners(
    target_shape: tuple[int, int],
    *,
    target_pixel_to_world: Callable[..., Any],
    source_world_to_pixel: Callable[..., Any],
    coarse_step: tuple[int, int] | None = None,
) -> NDArray[np.float64]: ...


class StampResult:
    @property
    def stamp(self) -> NDArray[np.float64]: ...
    @property
    def error(self) -> Optional[NDArray[np.float64]]: ...
    @property
    def valid(self) -> NDArray[np.bool_]: ...
    @property
    def delta(self) -> tuple[float, float]: ...
    @property
    def origin(self) -> tuple[int, int]: ...


def build_stamp(
    cutout: NDArray[Any],
    stamp_size: int,
    *,
    error: Optional[NDArray[Any]] = None,
    mask: Optional[NDArray[np.bool_]] = None,
    weight_fwhm: float = 3.0,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> Optional[StampResult]: ...

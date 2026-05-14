from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


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

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def reproject_exact(
    image_in: NDArray[Any],
    pixel_corners: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


def make_pixel_corners(
    output_shape: tuple[int, int],
    *,
    output_pixel_to_world: Callable[..., Any],
    input_world_to_pixel: Callable[..., Any],
) -> NDArray[np.float64]: ...

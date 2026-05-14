from typing import Any

import numpy as np
from numpy.typing import NDArray


def reproject_exact(
    image_in: NDArray[Any],
    pixel_corners: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

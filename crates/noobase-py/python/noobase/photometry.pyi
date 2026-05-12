from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import Grid

Convention = Literal["photon_counting", "energy_weighted"]


def synthetic(
    *,
    spectrum_grid: Union[Grid, NDArray[Any]],
    spectrum_flux: NDArray[Any],
    spectrum_error: Optional[NDArray[Any]] = None,
    transmission_grid: Union[Grid, NDArray[Any]],
    transmission_values: NDArray[Any],
    convention: Convention = "photon_counting",
) -> tuple[float, Optional[float], float]: ...


class SyntheticOperator:
    def __init__(
        self,
        *,
        spectrum_grid: Grid,
        transmission_grid: Union[Grid, NDArray[Any]],
        transmission_values: NDArray[Any],
        convention: Convention = "photon_counting",
    ) -> None: ...

    @property
    def coverage(self) -> float: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def apply(
        self,
        spectrum_flux: NDArray[Any],
        *,
        spectrum_error: Optional[NDArray[Any]] = None,
    ) -> tuple[float, Optional[float]]: ...

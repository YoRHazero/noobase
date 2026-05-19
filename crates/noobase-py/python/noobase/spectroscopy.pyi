from typing import Any, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from . import Grid, GridKind, Spacing


class Spectrum:
    @overload
    def __init__(
        self,
        *,
        wavelength: Grid,
        flux: NDArray[Any],
        error: Optional[NDArray[Any]] = None,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        wavelength: NDArray[Any],
        flux: NDArray[Any],
        error: Optional[NDArray[Any]] = None,
        mask: Optional[NDArray[np.bool_]] = None,
        spacing: Spacing = "linear",
        kind: GridKind = "centers",
    ) -> None: ...

    @property
    def wavelength(self) -> Grid: ...

    @property
    def flux(self) -> NDArray[Any]: ...

    @property
    def error(self) -> Optional[NDArray[Any]]: ...

    @property
    def mask(self) -> Optional[NDArray[np.bool_]]: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @overload
    def rebin(self, target: Grid) -> "Spectrum": ...

    @overload
    def rebin(
        self,
        target: NDArray[Any],
        *,
        spacing: Spacing = "linear",
        kind: GridKind = "centers",
    ) -> "Spectrum": ...

    def to_f_nu(self, speed_of_light: float) -> "Spectrum": ...
    def to_f_lambda(self, speed_of_light: float) -> "Spectrum": ...

    def synthetic_photometry(
        self,
        *,
        transmission_grid: Union[Grid, NDArray[Any]],
        transmission_values: NDArray[Any],
        convention: Literal["photon_counting", "energy_weighted"] = "photon_counting",
    ) -> tuple[float, Optional[float], float]: ...

    def convolve_lsf(
        self,
        *,
        spec: Literal["constant_r", "constant_velocity"],
        resolving_power: Optional[float] = None,
        sigma: Optional[float] = None,
        speed_of_light: Optional[float] = None,
    ) -> "Spectrum": ...

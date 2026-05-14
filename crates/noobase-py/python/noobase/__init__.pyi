from typing import Any, Literal, Optional, Union, overload

import numpy as np
from numpy.typing import NDArray

from . import image as image
from . import overlap as overlap
from . import photometry as photometry

Spacing = Literal["linear", "log"]
GridKind = Literal["centers", "edges"]

WavelengthArg = Union["Grid", NDArray[Any]]
TargetArg = Union["Grid", NDArray[Any]]


class Grid:
    def __init__(
        self,
        values: NDArray[Any],
        *,
        spacing: Spacing = "linear",
        kind: GridKind = "centers",
    ) -> None: ...

    @classmethod
    def linspace(
        cls,
        start: float,
        end: float,
        n: int,
        *,
        kind: GridKind = "centers",
        dtype: Optional[Any] = None,
    ) -> "Grid": ...

    @classmethod
    def logspace(
        cls,
        start: float,
        end: float,
        n: int,
        *,
        kind: GridKind = "centers",
        dtype: Optional[Any] = None,
    ) -> "Grid": ...

    @classmethod
    def from_array(
        cls,
        values: NDArray[Any],
        *,
        rel_tol: float = 1e-9,
        kind: GridKind = "centers",
    ) -> "Grid": ...

    @property
    def values(self) -> NDArray[Any]: ...

    @property
    def spacing(self) -> Spacing: ...

    @property
    def kind(self) -> GridKind: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def __len__(self) -> int: ...

    def to_edges(self) -> "Grid": ...

    def to_centers(self) -> "Grid": ...

    def is_uniform(self, rel_tol: float = 1e-9) -> bool: ...


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



from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import image as image
from . import overlap as overlap
from . import photometry as photometry
from . import spectroscopy as spectroscopy

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

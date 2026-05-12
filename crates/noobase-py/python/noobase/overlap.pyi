from typing import Any

from numpy.typing import NDArray

from . import Grid


def rebin(
    source_grid: Grid,
    source_values: NDArray[Any],
    target_grid: Grid,
) -> NDArray[Any]: ...


def rebin_variance(
    source_grid: Grid,
    source_variance: NDArray[Any],
    target_grid: Grid,
) -> NDArray[Any]: ...


def coverage(
    source_grid: Grid,
    target_grid: Grid,
) -> NDArray[Any]: ...

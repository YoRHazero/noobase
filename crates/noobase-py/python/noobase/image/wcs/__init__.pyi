from typing import Any

import numpy as np
from numpy.typing import NDArray

class WcsProgram:
    def __init__(self, spec: dict[str, Any]) -> None: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_outputs(self) -> int: ...
    def __call__(
        self, *inputs: float | NDArray[np.float64]
    ) -> tuple[float, ...] | tuple[NDArray[np.float64], ...]: ...

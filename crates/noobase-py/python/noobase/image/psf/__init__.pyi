from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

CombineMethod = Literal["clipped_mean", "median"]
ResidualReweight = Literal["none", "huber", "tukey"]


class RobustCombined:
    @property
    def combined(self) -> NDArray[np.float64]: ...
    @property
    def weight(self) -> NDArray[np.float64]: ...
    @property
    def count(self) -> NDArray[np.uint32]: ...


class FluxBackground:
    @property
    def flux(self) -> NDArray[np.float64]: ...
    @property
    def background(self) -> NDArray[np.float64]: ...
    @property
    def ok(self) -> NDArray[np.bool_]: ...


class BuildEpsf:
    @property
    def epsf(self) -> NDArray[np.float64]: ...
    @property
    def flux(self) -> NDArray[np.float64]: ...
    @property
    def background(self) -> NDArray[np.float64]: ...
    @property
    def delta(self) -> NDArray[np.float64]: ...
    @property
    def ok(self) -> NDArray[np.bool_]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...


class ExtendedPsf:
    @property
    def core(self) -> NDArray[np.float64]: ...
    @property
    def oversample(self) -> int: ...
    @property
    def wing(self) -> NDArray[np.float64]: ...
    @property
    def match_radius(self) -> float: ...
    @property
    def feather_width(self) -> float: ...
    @property
    def ee_aperture_radius(self) -> float: ...


class ExtendedPsfBuilt:
    @property
    def extended(self) -> ExtendedPsf: ...
    @property
    def star_flux(self) -> NDArray[np.float64]: ...
    @property
    def star_background(self) -> NDArray[np.float64]: ...
    @property
    def star_ok(self) -> NDArray[np.bool_]: ...
    @property
    def star_scale_from_core(self) -> NDArray[np.bool_]: ...


def robust_combine(
    stack: NDArray[Any],
    *,
    weight: Optional[NDArray[Any]] = None,
    method: CombineMethod = "clipped_mean",
    combine_kappa: float = 3.0,
    combine_max_iter: int = 5,
) -> RobustCombined: ...


def solve_flux_background(
    epsf: NDArray[np.float64],
    oversample: int,
    data: NDArray[Any],
    delta: NDArray[np.float64],
    *,
    weight: Optional[NDArray[Any]] = None,
) -> FluxBackground: ...


def build_epsf(
    data: NDArray[Any],
    delta_init: NDArray[np.float64],
    oversample: int,
    *,
    weight: Optional[NDArray[Any]] = None,
    psi_init: Optional[NDArray[np.float64]] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    step: float = 1.0,
    residual_reweight: ResidualReweight = "none",
    reweight_c: float = 4.0,
    nuisance_max_iter: int = 3,
    nuisance_tol: float = 1e-4,
) -> BuildEpsf: ...


def stitch_psf(
    core: NDArray[np.float64],
    oversample: int,
    wing: NDArray[np.float64],
    *,
    wing_confidence: Optional[NDArray[np.float64]] = None,
    match_radius: float = 8.0,
    feather_width: float = 4.0,
    ee_aperture_radius: float = 15.0,
) -> ExtendedPsf: ...


def build_extended_psf(
    wing_data: NDArray[Any],
    wing_delta: NDArray[np.float64],
    core: NDArray[np.float64],
    oversample: int,
    *,
    wing_weight: Optional[NDArray[Any]] = None,
    match_radius: float = 8.0,
    feather_width: float = 4.0,
    ee_aperture_radius: float = 15.0,
    combine: CombineMethod = "clipped_mean",
    combine_kappa: float = 3.0,
    combine_max_iter: int = 5,
    scale_aperture_radius: float = 6.0,
    scale_background_annulus: tuple[float, float] = (18.0, 24.0),
) -> ExtendedPsfBuilt: ...

"""Tests for ``noobase.image.wcs.WcsProgram``.

Programs are built by hand from the spec schema and checked against direct
numpy reference math; no WCS library is involved (compiling real gwcs trees
into specs is the caller's job and is tested there).
"""

import numpy as np
import pytest

from noobase.image.wcs import WcsProgram


def _poly2d_coeffs(degree: int, terms: dict[tuple[int, int], float]) -> list[float]:
    """Dense row-major ``(degree+1)**2`` coefficient list from sparse terms."""
    matrix = np.zeros((degree + 1, degree + 1))
    for (i, j), value in terms.items():
        matrix[i, j] = value
    return matrix.ravel().tolist()


def test_shift_scale_chain_scalar_and_array_agree() -> None:
    spec = {
        "n_regs": 3,
        "inputs": [0],
        "outputs": [2],
        "ops": [
            {"op": "shift", "in": [0], "out": [1], "offset": 1.5},
            {"op": "scale", "in": [1], "out": [2], "factor": -2.0},
        ],
    }
    program = WcsProgram(spec)
    assert program.n_inputs == 1
    assert program.n_outputs == 1

    (scalar,) = program(3.0)
    assert scalar == (3.0 + 1.5) * -2.0

    x = np.linspace(-5, 5, 10_001)
    (out,) = program(x)
    np.testing.assert_allclose(out, (x + 1.5) * -2.0, rtol=0, atol=0)


def test_poly2d_matches_numpy_reference() -> None:
    degree = 5
    rng = np.random.default_rng(7)
    matrix = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            matrix[i, j] = rng.normal()
    spec = {
        "n_regs": 3,
        "inputs": [0, 1],
        "outputs": [2],
        "ops": [
            {
                "op": "poly2d",
                "in": [0, 1],
                "out": [2],
                "degree": degree,
                "coeffs": matrix.ravel().tolist(),
            }
        ],
    }
    program = WcsProgram(spec)
    x = rng.uniform(-2, 2, size=(64, 64))
    y = rng.uniform(-2, 2, size=(64, 64))
    (out,) = program(x, y)
    want = sum(
        matrix[i, j] * x**i * y**j
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    )
    np.testing.assert_allclose(out, want, rtol=1e-13, atol=1e-13)
    assert out.shape == x.shape


def test_sphere_round_trip_through_rotation() -> None:
    # s2c -> rot3(R) -> c2s -> s2c -> rot3(R^T) -> c2s == identity.
    angle = np.deg2rad(30.0)
    rot = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    rot_t = [list(row) for row in np.array(rot).T]
    spec = {
        "n_regs": 12,
        "inputs": [0, 1],
        "outputs": [10, 11],
        "ops": [
            {"op": "sph2cart", "in": [0, 1], "out": [2, 3, 4]},
            {"op": "rot3", "in": [2, 3, 4], "out": [5, 6, 7], "matrix": rot},
            {"op": "cart2sph", "in": [5, 6, 7], "out": [8, 9], "wrap_lon_at": 360},
            {"op": "sph2cart", "in": [8, 9], "out": [2, 3, 4]},
            {"op": "rot3", "in": [2, 3, 4], "out": [5, 6, 7], "matrix": rot_t},
            {"op": "cart2sph", "in": [5, 6, 7], "out": [10, 11], "wrap_lon_at": 360},
        ],
    }
    program = WcsProgram(spec)
    rng = np.random.default_rng(11)
    lon = rng.uniform(0, 360, 1000)
    lat = rng.uniform(-89, 89, 1000)
    out_lon, out_lat = program(lon, lat)
    np.testing.assert_allclose(out_lon, lon, atol=1e-11)
    np.testing.assert_allclose(out_lat, lat, atol=1e-11)


def test_tan_projection_round_trip_and_affine() -> None:
    spec = {
        "n_regs": 6,
        "inputs": [0, 1],
        "outputs": [4, 5],
        "ops": [
            {"op": "tan_project", "in": [0, 1], "out": [2, 3]},
            {
                "op": "affine2",
                "in": [2, 3],
                "out": [2, 3],
                "matrix": [[2.0, 0.0], [0.0, 2.0]],
                "translation": [1.0, -1.0],
            },
            {
                "op": "affine2",
                "in": [2, 3],
                "out": [2, 3],
                "matrix": [[0.5, 0.0], [0.0, 0.5]],
                "translation": [-0.5, 0.5],
            },
            {"op": "tan_deproject", "in": [2, 3], "out": [4, 5]},
        ],
    }
    program = WcsProgram(spec)
    rng = np.random.default_rng(3)
    lon = rng.uniform(0, 360, 500)
    lat = rng.uniform(88.0, 89.9, 500)  # native TAN domain: near the pole
    out_lon, out_lat = program(lon, lat)
    np.testing.assert_allclose(np.mod(out_lon, 360), np.mod(lon, 360), atol=1e-9)
    np.testing.assert_allclose(out_lat, lat, atol=1e-10)


def _grism_specs(cubic: bool = False) -> tuple[dict, dict]:
    """A synthetic row-dispersion grism pair (forward and backward specs).

    lambda(t; x0, y0) = (3.9 + 1e-5 x0) + (1.1 + 1e-5 y0) t + 0.02 t^2
                        [+ (0.005 + 1e-6 x0) t^3 with ``cubic=True``,
                        mirroring current CRDS NIRCam GRISMR specwcs]
    dx(t) = -1200 + 3000 t   (t-only, matching NIRCam row dispersion)
    dy(t; x0, y0) = 0.05 + 1e-6 x0 + 0.4 t
    """
    lmodel_coeffs = [
        _poly2d_coeffs(1, {(0, 0): 3.9, (1, 0): 1e-5}),
        _poly2d_coeffs(1, {(0, 0): 1.1, (0, 1): 1e-5}),
        _poly2d_coeffs(1, {(0, 0): 0.02}),
    ]
    if cubic:
        lmodel_coeffs.append(_poly2d_coeffs(1, {(0, 0): 0.005, (1, 0): 1e-6}))
    lmodels = [{"kind": "spatial", "degree": 1, "coeffs": lmodel_coeffs}]
    xmodels = [{"kind": "t", "coeffs": [-1200.0, 3000.0]}]
    ymodels = [
        {"kind": "spatial", "degree": 1, "coeffs": [
            _poly2d_coeffs(1, {(0, 0): 0.05, (1, 0): 1e-6}),
            _poly2d_coeffs(1, {(0, 0): 0.4}),
        ]},
    ]
    forward = {
        "n_regs": 6,
        "inputs": [0, 1, 2, 3, 4],  # x, y, x0, y0, order
        "outputs": [2, 3, 5, 4],  # x0, y0, lambda, order (pass-through wiring)
        "ops": [
            {
                "op": "grism_forward",
                "in": [0, 1, 2, 3, 4],
                "out": [5],
                "axis": "row",
                "orders": [1],
                "alongdisp": xmodels,
                "lmodels": lmodels,
            }
        ],
    }
    backward = {
        "n_regs": 6,
        "inputs": [0, 1, 2, 3],  # x0, y0, wavelength, order
        "outputs": [4, 5, 0, 1, 3],  # x, y, x0, y0, order
        "ops": [
            {
                "op": "grism_backward",
                "in": [0, 1, 2, 3],
                "out": [4, 5],
                "orders": [1],
                "lmodels": lmodels,
                "xmodels": xmodels,
                "ymodels": ymodels,
            }
        ],
    }
    return forward, backward


@pytest.mark.parametrize("cubic", [False, True], ids=["quadratic", "cubic"])
def test_grism_backward_forward_round_trip(cubic: bool) -> None:
    forward_spec, backward_spec = _grism_specs(cubic=cubic)
    forward = WcsProgram(forward_spec)
    backward = WcsProgram(backward_spec)

    rng = np.random.default_rng(5)
    x0 = rng.uniform(100, 1900, 300)
    y0 = rng.uniform(100, 1900, 300)
    wavelength = rng.uniform(3.95, 4.9, 300)
    order = np.ones_like(x0)

    x, y, x0_out, y0_out, order_out = backward(x0, y0, wavelength, order)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    np.testing.assert_array_equal(x0_out, x0)
    np.testing.assert_array_equal(order_out, order)

    x0_rt, y0_rt, lam_rt, _ = forward(x, y, x0, y0, order)
    np.testing.assert_array_equal(x0_rt, x0)
    np.testing.assert_array_equal(y0_rt, y0)
    # Newton trace inversion: round trip to near machine precision.
    np.testing.assert_allclose(lam_rt, wavelength, atol=1e-10)


def test_grism_unknown_order_yields_nan() -> None:
    _, backward_spec = _grism_specs()
    backward = WcsProgram(backward_spec)
    x, y, *_ = backward(
        np.array([500.0]), np.array([500.0]), np.array([4.2]), np.array([9.0])
    )
    assert np.isnan(x).all()
    assert np.isnan(y).all()


def test_non_contiguous_input_is_handled() -> None:
    spec = {
        "n_regs": 1,
        "inputs": [0],
        "outputs": [0],
        "ops": [{"op": "shift", "in": [0], "out": [0], "offset": 1.0}],
    }
    program = WcsProgram(spec)
    base = np.arange(100.0).reshape(10, 10)
    view = base.T  # non-contiguous
    (out,) = program(view)
    np.testing.assert_array_equal(out, view + 1.0)


def test_spec_validation_errors() -> None:
    with pytest.raises(ValueError, match="missing key"):
        WcsProgram({"n_regs": 1, "inputs": [0], "outputs": [0]})
    with pytest.raises(ValueError, match="out of range"):
        WcsProgram({"n_regs": 1, "inputs": [0], "outputs": [4], "ops": []})
    with pytest.raises(ValueError, match="unknown op"):
        WcsProgram(
            {
                "n_regs": 1,
                "inputs": [0],
                "outputs": [0],
                "ops": [{"op": "warp9", "in": [0], "out": [0]}],
            }
        )
    program = WcsProgram({"n_regs": 2, "inputs": [0, 1], "outputs": [0], "ops": []})
    with pytest.raises(ValueError, match="input"):
        program(np.zeros(3))


def test_select_routes_by_label_array() -> None:
    # Two regions: label 1 -> x + 100, label 2 -> x * 2; label 0 -> NaN.
    labels = np.zeros((4, 8), dtype=np.int64)
    labels[:, 2:5] = 1
    labels[:, 5:] = 2
    case = lambda spec_ops: {  # noqa: E731
        "n_regs": 3,
        "inputs": [0, 1],
        "outputs": [2],
        "ops": spec_ops,
    }
    spec = {
        "n_regs": 3,
        "inputs": [0, 1],
        "outputs": [2],
        "ops": [
            {
                "op": "select",
                "in": [0, 1],
                "out": [2],
                "label": {"kind": "array", "data": labels},
                "cases": [
                    {"label": 1, "program": case([
                        {"op": "shift", "in": [0], "out": [2], "offset": 100.0}
                    ])},
                    {"label": 2, "program": case([
                        {"op": "scale", "in": [0], "out": [2], "factor": 2.0}
                    ])},
                ],
            }
        ],
    }
    program = WcsProgram(spec)
    yy, xx = np.mgrid[0:4, 0:8].astype(float)
    (out,) = program(xx, yy)
    assert np.isnan(out[:, :2]).all()
    np.testing.assert_array_equal(out[:, 2:5], xx[:, 2:5] + 100.0)
    np.testing.assert_array_equal(out[:, 5:], xx[:, 5:] * 2.0)


def test_select_dict_labels_and_tabular_logical() -> None:
    sub = {
        "n_regs": 2,
        "inputs": [0],
        "outputs": [1],
        "ops": [{"op": "shift", "in": [0], "out": [1], "offset": 1.0}],
    }
    spec = {
        "n_regs": 2,
        "inputs": [0],
        "outputs": [1],
        "ops": [
            {
                "op": "select",
                "in": [0],
                "out": [1],
                "label": {
                    "kind": "dict",
                    "keys": [0.5, 1.5],
                    "labels": [1, 1],
                    "key_input": 0,
                    "atol": 0.01,
                },
                "cases": [{"label": 1, "program": sub}],
            }
        ],
    }
    program = WcsProgram(spec)
    (out,) = program(np.array([0.5005, 1.499, 3.0, np.nan]))
    np.testing.assert_allclose(out[:2], [1.5005, 2.499])
    assert np.isnan(out[2:]).all()

    tab = WcsProgram(
        {
            "n_regs": 1,
            "inputs": [0],
            "outputs": [0],
            "ops": [
                {
                    "op": "tabular1d",
                    "in": [0],
                    "out": [0],
                    "points": [0.0, 1.0, 3.0],
                    "values": [10.0, 20.0, 40.0],
                    "fill": float("nan"),
                }
            ],
        }
    )
    (out,) = tab(np.array([0.5, 2.0, 3.0, -0.1, 3.1]))
    np.testing.assert_allclose(out[:3], [15.0, 30.0, 40.0])
    assert np.isnan(out[3:]).all()

    logic = WcsProgram(
        {
            "n_regs": 1,
            "inputs": [0],
            "outputs": [0],
            "ops": [
                {
                    "op": "logical",
                    "in": [0],
                    "out": [0],
                    "condition": "GT",
                    "compareto": 0.55,
                    "value": float("nan"),
                }
            ],
        }
    )
    (out,) = logic(np.array([0.5, 0.6, np.nan]))
    assert out[0] == 0.5
    assert np.isnan(out[1])
    assert np.isnan(out[2])

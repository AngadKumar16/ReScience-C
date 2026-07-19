import numpy as np

from src.grid import (
    P_MAX,
    P_MIN,
    P_RESOLUTION,
    V_MAX,
    V_MIN,
    V_RESOLUTION,
    arcsin_sqrt_to_p,
    build_grid,
    p_to_arcsin_sqrt,
    v_to_log,
    log_to_v,
)


def test_grid_axis_lengths():
    grid = build_grid(n_values=np.array([4, 5, 6, 7, 8]))
    assert grid.p_transformed.shape == (P_RESOLUTION,)
    assert grid.p_values.shape == (P_RESOLUTION,)
    assert grid.v_transformed.shape == (V_RESOLUTION,)
    assert grid.v_values.shape == (V_RESOLUTION,)
    assert grid.n_values.shape == (5,)


def test_grid_bounds_respected():
    grid = build_grid(n_values=np.array([6]))
    assert grid.p_values.min() >= P_MIN - 1e-9
    assert grid.p_values.max() <= P_MAX + 1e-9
    assert grid.v_values.min() >= V_MIN - 1e-9
    assert grid.v_values.max() <= V_MAX + 1e-9


def test_p_transform_round_trip():
    p = np.array([0.04, 0.1, 0.35, 0.5, 0.96])
    theta = p_to_arcsin_sqrt(p)
    p_recovered = arcsin_sqrt_to_p(theta)
    np.testing.assert_allclose(p, p_recovered, atol=1e-10)


def test_v_transform_round_trip():
    v = np.array([0.05, 0.1, 0.5, 1.0])
    log_v = v_to_log(v)
    v_recovered = log_to_v(log_v)
    np.testing.assert_allclose(v, v_recovered, atol=1e-10)


def test_build_grid_rejects_empty_n():
    try:
        build_grid(n_values=np.array([], dtype=int))
        assert False, "expected ValueError for empty n_values"
    except ValueError:
        pass


def test_build_grid_rejects_nonpositive_n():
    try:
        build_grid(n_values=np.array([0, 1, 2]))
        assert False, "expected ValueError for n_values containing 0"
    except ValueError:
        pass

import numpy as np

from src.q_model import q_function
from src.simulate_q import simulate_from_q_model


def test_q_function_integrates_to_one():
    x = np.linspace(-200, 1000, 4000)
    y = q_function(x, n=6, p=0.35, gamma=11.1, lam=9.0, eps2=625.0)
    integral = np.trapezoid(y, x)
    assert abs(integral - 1.0) < 1e-2


def test_q_function_rejects_invalid_params():
    x = np.array([0.0])
    try:
        q_function(x, n=6, p=1.5, gamma=11.1, lam=9.0, eps2=625.0)
        assert False, "expected ValueError for p > 1"
    except ValueError:
        pass


def test_simulate_from_q_model_shape_and_mean():
    rng = np.random.default_rng(0)
    n, p, gamma, lam, eps2 = 6, 0.35, 11.1, 9.0, 625.0
    n_obs = 100_000
    amps = simulate_from_q_model(n, p, gamma, lam, eps2, n_obs, rng)
    assert amps.shape == (n_obs,)
    expected_mean = n * p * (gamma * lam)  # mu = npq, q = gamma*lam
    assert abs(amps.mean() - expected_mean) / expected_mean < 0.02

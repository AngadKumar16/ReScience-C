import numpy as np

from src.simulate import simulate_responses


def test_simulate_responses_shape():
    rng = np.random.default_rng(0)
    result = simulate_responses(
        n_sites=6,
        p=0.35,
        q=100.0,
        cv_intra=0.3,
        sigma_baseline=25.0,
        n_obs=50,
        rng=rng,
    )
    assert result.amplitudes.shape == (50,)
    assert result.k_released.shape == (50,)
    assert np.all(result.k_released >= 0)
    assert np.all(result.k_released <= 6)


def test_simulate_responses_mean_converges():
    rng = np.random.default_rng(1)
    n_sites, p, q = 6, 0.35, 100.0
    result = simulate_responses(
        n_sites=n_sites,
        p=p,
        q=q,
        cv_intra=0.3,
        sigma_baseline=25.0,
        n_obs=200_000,
        rng=rng,
    )
    expected_mean = n_sites * p * q
    observed_mean = result.amplitudes.mean()
    assert abs(observed_mean - expected_mean) / expected_mean < 0.02


def test_simulate_responses_rejects_invalid_p():
    rng = np.random.default_rng(2)
    try:
        simulate_responses(6, 1.5, 100.0, 0.3, 25.0, 10, rng)
        assert False, "expected ValueError for p > 1"
    except ValueError:
        pass
